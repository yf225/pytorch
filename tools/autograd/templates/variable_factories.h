#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/MemoryFormat.h>
#include <ATen/core/EnableNamedTensor.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/ir.h>

#include <functional>
#include <initializer_list>
#include <utility>

#ifdef BUILD_NAMEDTENSOR
using at::DimnameList;
#endif

namespace torch {

namespace detail {
  struct ScalarWrapper {
#define TENSOR(T, S) \
    ScalarWrapper(T value) : scalar_(value), scalar_type_(at::k##S), is_scalar_(true) {} \
    ScalarWrapper(std::initializer_list<T> value) : scalar_(*value.begin()), scalar_type_(at::k##S), is_scalar_(false) {}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

    c10::Scalar scalar_;
    c10::ScalarType scalar_type_;
    bool is_scalar_;
  };

  enum class InitListTensorType { Scalar, InitList };

  // We use `InitListTensor` to support converting an arbitrarily nested braced-init-list
  // (e.g. {{1, 2}, {3, 4}}) into the equivalent Tensor, taking advantage of the fact that
  // the constructor will automatically be called recursively until it reaches all innermost
  // scalar values.
  //
  // At any time, a `InitListTensor` object represents either of the following:
  // 1. A scalar with value `scalar()` and type `scalar_type()`.
  // 2. A Tensor represented in `std::initializer_list<InitListTensor>` form, with value
  //    `init_list()`, Tensor scalar type `scalar_type()`, and Tensor sizes `sizes()`.
  struct InitListTensor {
    InitListTensor(ScalarWrapper scalar_wrapper) :
        scalar_(scalar_wrapper.scalar_), init_list_(),
        sizes_((scalar_wrapper.is_scalar_ ? std::vector<int64_t>({}) : std::vector<int64_t>({1}))),
        scalar_type_(scalar_wrapper.scalar_type_),
        type_(InitListTensorType::Scalar) {}
    InitListTensor(std::initializer_list<InitListTensor> init_list) :
        scalar_(),
        init_list_(init_list),
        sizes_(),
        scalar_type_(),
        type_(InitListTensorType::InitList) {
      TORCH_CHECK(
        init_list.size() > 0,
        "Empty init-list is not yet supported. We can create tensors with zero-size dimensions in the following way:\n",
        "1-D: `torch::randn({0})`\n",
        "N-D: `torch::randn({2, 3, 0})`");
      scalar_type_ = init_list.begin()->scalar_type_;
      const InitListTensor& first_elem = *(init_list.begin());
      for (const auto& elem : init_list) {
        TORCH_CHECK(elem.scalar_type_ == first_elem.scalar_type_,
          "Expected all elements of the tensor to have the same scalar type: ",
          first_elem.scalar_type_,
          ", but got element of scalar type: ",
          elem.scalar_type_);
        TORCH_CHECK(elem.sizes_ == first_elem.sizes_,
          "Expected all sub-lists to have sizes: ",
          first_elem.sizes_,
          " (e.g. ", first_elem, "), ",
          "but got sub-list ",
          elem,
          " with sizes: ",
          elem.sizes_);
      }
      sizes_.reserve(first_elem.sizes_.size() + 1);
      sizes_.push_back(init_list.size());
      sizes_.insert(sizes_.end(), first_elem.sizes_.begin(), first_elem.sizes_.end());
    }

    const c10::Scalar& scalar() const {
      return scalar_;
    }

    const std::initializer_list<InitListTensor>& init_list() const {
      return init_list_;
    }

    const std::vector<int64_t>& sizes() const {
      return sizes_;
    }

    const c10::ScalarType& scalar_type() const {
      return scalar_type_;
    }

    const InitListTensorType& type() const {
      return type_;
    }

    at::Tensor to_tensor(const at::TensorOptions& options) const {
      // NOTE: Here we explicitly choose to initialize the tensor on CPU first,
      // fill each element of the tensor, and then move the tensor to the desired
      // device. For CUDA device, this approach only involves 1 CUDA kernel launch,
      // and is much faster than initializing the tensor on CUDA first and then
      // filling each element of it (which involves `N` CUDA kernel launches where
      // `N` is the number of the elements in the tensor).
      at::Tensor tensor = ([&]() {
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        return at::empty(sizes_, at::TensorOptions(options).device(at::kCPU).is_variable(false));
      })();
      fill_tensor(tensor);
      return tensor.to(options.device());
    }

    void pretty_print_recursive(std::ostream& stream) const {
      if (type_ == InitListTensorType::Scalar) {
        AT_DISPATCH_ALL_TYPES_AND3(at::kBool, at::kHalf, at::kBFloat16, scalar_type_, "InitListTensor_pretty_print_scalar", [&] {
          stream << scalar_.to<scalar_t>();
        });
      } else if (type_ == InitListTensorType::InitList) {
        stream << "{";
        for (const InitListTensor* it = init_list_.begin(); it != init_list_.end(); it++) {
          it->pretty_print_recursive(stream);
          if (std::next(it) != init_list_.end()) stream << ", ";
        }
        stream << "}";
      }
    }

   private:
    void fill_tensor(at::Tensor tensor) const {
      size_t index = 0;
      for (const auto& elem : init_list_) {
        if (elem.type_ == InitListTensorType::Scalar) {
          at::NoGradGuard guard;
          tensor[index].fill_(elem.scalar());
        } else if (elem.type_ == InitListTensorType::InitList) {
          elem.fill_tensor(tensor[index]);
        } else {
          TORCH_INTERNAL_ASSERT(false, "Invalid InitListTensor");
        }
        index++;
      }
    }
    c10::Scalar scalar_;
    std::initializer_list<InitListTensor> init_list_;
    std::vector<int64_t> sizes_;
    c10::ScalarType scalar_type_;
    InitListTensorType type_;
  };

  inline std::ostream& operator<<(std::ostream& stream, const InitListTensor& init_list_tensor) {
    init_list_tensor.pretty_print_recursive(stream);
    return stream;
  }
} // namespace detail

#define TENSOR(T, S)                                                       \
  inline at::Tensor tensor(                                                \
      at::ArrayRef<T> values, const at::TensorOptions& options) {          \
    at::Tensor result = ([&]() {                                           \
      at::AutoNonVariableTypeMode non_var_type_mode(true);                 \
      return at::tensor(values, at::TensorOptions(options).is_variable(false)); \
    })();                                                                  \
    return autograd::make_variable(result, options.requires_grad());       \
  }                                                                        \
  inline at::Tensor tensor(                                                \
      std::initializer_list<T> values, const at::TensorOptions& options) { \
    return torch::tensor(at::ArrayRef<T>(values), options);                \
  }                                                                        \
  inline at::Tensor tensor(T value, const at::TensorOptions& options) {    \
    return torch::tensor(at::ArrayRef<T>(value), options);                 \
  }                                                                        \
  inline at::Tensor tensor(at::ArrayRef<T> values) {                       \
    return torch::tensor(std::move(values), at::dtype(at::k##S));          \
  }                                                                        \
  inline at::Tensor tensor(std::initializer_list<T> values) {              \
    return torch::tensor(at::ArrayRef<T>(values));                         \
  }                                                                        \
  inline at::Tensor tensor(T value) {                                      \
    return torch::tensor(at::ArrayRef<T>(value));                          \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

/// NOTE: `torch::tensor({})` doesn't work at the moment because we would need to solve the
/// ambiguous overload problem (see https://github.com/pytorch/pytorch/pull/26210#discussion_r325336686).
/// We can create tensors with zero-size dimensions in the following way instead:
/// - 1-D tensor: `torch::randn({0})`
/// - N-D tensor: `torch::randn({2, 3, 0})`
///
/// NOTE: Currently `torch::tensor(...)` doesn't support mixed data types
/// (i.e. `torch::tensor({{bool, 2.0}})` doesn't work). We might be able to
/// support it in the future by iterating over all sub-lists to find
/// the largest data type that can represent all of the elements, or by using
/// variadic templates.
inline at::Tensor tensor(detail::InitListTensor init_list_tensor, const at::TensorOptions& options) {
  return autograd::make_variable(init_list_tensor.to_tensor(options), options.requires_grad());
}

inline at::Tensor tensor(detail::InitListTensor init_list_tensor) {
  return torch::tensor(init_list_tensor, at::dtype(init_list_tensor.scalar_type()));
}

/// NOTE: We add `torch::tensor(std::initializer_list<detail::InitListTensor>)` function overload (and its options variant),
/// so that `torch::tensor({{1, 2}})` can take this overload instead of `torch::tensor(at::ArrayRef<T>)`.
inline at::Tensor tensor(std::initializer_list<detail::InitListTensor> init_list, const at::TensorOptions& options) {
  TORCH_INTERNAL_ASSERT(
    init_list.begin()->type() != detail::InitListTensorType::Scalar,
    "1D tensor construction such as `torch::tensor({1, 2, 3})` should never take the ",
    "torch::tensor(std::initializer_list<detail::InitListTensor>) function overload. ",
    "Please fix the code to avoid this regression.")
  return torch::tensor(detail::InitListTensor(init_list), options);
}

inline at::Tensor tensor(std::initializer_list<detail::InitListTensor> init_list) {
  return torch::tensor(init_list, at::dtype(init_list.begin()->scalar_type()));
}

/// A generic deleter function.
using Deleter = std::function<void(void*)>;
using at::MemoryFormat;

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `deleter` function (a
/// `std::function<void(void*)>`) will be called on the `data` when the Tensor
/// data would normally be deallocated. The `TensorOptions` specify additional
/// configuration options for the returned tensor, such as what type to
/// interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::from_blob(data, sizes, strides, deleter, options.is_variable(false));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `TensorOptions`
/// specify additional configuration options for the returned tensor, such as
/// what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options = at::TensorOptions()) {
  return torch::from_blob(
      data,
      sizes,
      strides,
      /*deleter=*/[](void*) {},
      options);
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The `deleter`
/// (a `std::function<void(void*)>`) function will be called on the `data` when
/// the Tensor data would normally be deallocated. The `TensorOptions` specify
/// additional configuration options for the returned tensor, such as what type
/// to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::from_blob(data, sizes, deleter, options.is_variable(false));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The
/// `TensorOptions` specify additional configuration options for the returned
/// tensor, such as what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const at::TensorOptions& options = at::TensorOptions()) {
  return torch::from_blob(data, sizes, /*deleter=*/[](void*) {}, options);
}

${function_definitions}

} // namespace torch
