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

/*
namespace detail {
  enum class InitListTensorType { Scalar, InitList };

  // We use `InitListTensor` to support converting an arbitrarily nested braced-init-list
  // (e.g. {{1, 2}, {3, 4}}) into the equivalent Tensor, taking advantage of the fact that
  // the constructor will automatically be called recursively until it reaches all innermost
  // scalar values.
  //
  // At any time, a `InitListTensor` object represents either of the following:
  // 1. A scalar with value `scalar()` and type `scalar_type()`.
  // 2. A Tensor represented in `std::vector<InitListTensor>` form, with value
  //    `vec()`, Tensor scalar type `scalar_type()`, and Tensor sizes `sizes()`.
  struct InitListTensor {
    InitListTensor() = default;
#define TENSOR(T, S) \
    InitListTensor(std::initializer_list<T> init_list) : \
        scalar_(), vec_(), \
        sizes_({init_list.size()}), \
        scalar_type_(at::k##S), \
        type_(InitListTensorType::InitList) { \
      for (const auto& elem : init_list) { \
        InitListTensor scalar_value; \
        scalar_value.scalar_ = elem; \
        scalar_value.scalar_type_ = at::k##S; \
        scalar_value.type_ = InitListTensorType::Scalar; \
        vec_.push_back(scalar_value); \
      } \
    }
AT_FORALL_SCALAR_TYPES_AND(Bool, TENSOR)
#undef TENSOR
    InitListTensor(std::initializer_list<InitListTensor> init_list) :
        scalar_(),
        vec_(init_list),
        sizes_(),
        scalar_type_(),
        type_(InitListTensorType::InitList) {
      TORCH_CHECK(
        vec_.size() > 0,
        "Empty init-list is not yet supported. We can create tensors with zero-size dimensions in the following way:\n",
        "1-D: `torch::randn({0})`\n",
        "N-D: `torch::randn({2, 3, 0})`");
      scalar_type_ = vec_.begin()->scalar_type_;
      const InitListTensor& first_elem = *(vec_.begin());
      for (const auto& elem : vec_) {
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
      sizes_.push_back(vec_.size());
      sizes_.insert(sizes_.end(), first_elem.sizes_.begin(), first_elem.sizes_.end());
    }

    const c10::Scalar& scalar() const {
      return scalar_;
    }

    const std::vector<InitListTensor>& vec() const {
      return vec_;
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
        for (std::vector<InitListTensor>::const_iterator it = vec_.begin() ; it != vec_.end(); it++) {
          it->pretty_print_recursive(stream);
          if (std::next(it) != vec_.end()) stream << ", ";
        }
        stream << "}";
      }
    }

   private:
    void fill_tensor(at::Tensor tensor) const {
      size_t index = 0;
      for (const auto& elem : vec_) {
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
    std::vector<InitListTensor> vec_;
    std::vector<int64_t> sizes_;
    c10::ScalarType scalar_type_;
    InitListTensorType type_;
  };

  inline std::ostream& operator<<(std::ostream& stream, const InitListTensor& init_list_tensor) {
    init_list_tensor.pretty_print_recursive(stream);
    return stream;
  }
} // namespace detail

*/

// #define TENSOR(T, S)                                                       \
//   inline at::Tensor tensor(                                                \
//       at::ArrayRef<T> values, const at::TensorOptions& options) {          \
//     at::Tensor result = ([&]() {                                           \
//       at::AutoNonVariableTypeMode non_var_type_mode(true);                 \
//       return at::tensor(values, at::TensorOptions(options).is_variable(false)); \
//     })();                                                                  \
//     return autograd::make_variable(result, options.requires_grad());       \
//   }                                                                        \
//   inline at::Tensor tensor(                                                \
//       std::initializer_list<T> values, const at::TensorOptions& options) { \
//     return torch::tensor(at::ArrayRef<T>(values), options);                \
//   }                                                                        \
//   inline at::Tensor tensor(T value, const at::TensorOptions& options) {    \
//     return torch::tensor(at::ArrayRef<T>(value), options);                 \
//   }                                                                        \
//   inline at::Tensor tensor(at::ArrayRef<T> values) {                       \
//     return torch::tensor(std::move(values), at::dtype(at::k##S));          \
//   }                                                                        \
//   inline at::Tensor tensor(std::initializer_list<T> values) {              \
//     return torch::tensor(at::ArrayRef<T>(values));                         \
//   }                                                                        \
//   inline at::Tensor tensor(T value) {                                      \
//     return torch::tensor(at::ArrayRef<T>(value));                          \
//   }
// AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
// #undef TENSOR

// /// NOTE: `torch::tensor({})` doesn't work at the moment because we would need to solve the
// /// ambiguous overload problem (see https://github.com/pytorch/pytorch/pull/26210#discussion_r325336686).
// /// We can create tensors with zero-size dimensions in the following way instead:
// /// - 1-D tensor: `torch::randn({0})`
// /// - N-D tensor: `torch::randn({2, 3, 0})`
// ///
// /// NOTE: Currently `torch::tensor(...)` doesn't support mixed data types
// /// (i.e. `torch::tensor({{bool, 2.0}})` doesn't work). We might be able to
// /// support it in the future by iterating over all sub-lists to find
// /// the largest data type that can represent all of the elements, or by using
// /// variadic templates.
// inline at::Tensor tensor(detail::InitListTensor init_list_tensor, const at::TensorOptions& options) {
//   return autograd::make_variable(init_list_tensor.to_tensor(options), options.requires_grad());
// }

// inline at::Tensor tensor(detail::InitListTensor init_list_tensor) {
//   return torch::tensor(init_list_tensor, at::dtype(init_list_tensor.scalar_type()));
// }

// /// NOTE: We add `torch::tensor(std::initializer_list<detail::InitListTensor>)` function overload (and its options variant),
// /// so that `torch::tensor({{1, 2}})` can take this overload instead of `torch::tensor(at::ArrayRef<T>)`.
// inline at::Tensor tensor(std::initializer_list<detail::InitListTensor> init_list, const at::TensorOptions& options) {
//   TORCH_INTERNAL_ASSERT(
//     init_list.begin()->type() != detail::InitListTensorType::Scalar,
//     "1D tensor construction such as `torch::tensor({1, 2, 3})` should never take the ",
//     "torch::tensor(std::initializer_list<detail::InitListTensor>) function overload. ",
//     "Please fix the code to avoid this regression.")
//   return torch::tensor(detail::InitListTensor(init_list), options);
// }

// inline at::Tensor tensor(std::initializer_list<detail::InitListTensor> init_list) {
//   return torch::tensor(init_list, at::dtype(init_list.begin()->scalar_type()));
// }

// yf225 TODO: let's test this 2D case first
/*
struct A
{
    A(int i_) : i (i_) {}
    A(std::initializer_list<int> il) : i (*il.begin() + 10) {}
    int i;
};

int main(int, char* []) {
  A a1 = 5; // a1.i == 5
  std::cout << a1.i << std::endl;

  A a2 = {5}; // a2.i = 6
  std::cout << a2.i << std::endl;

  return 0;
}
*/

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

struct ArrayRefWrapper {
#define TENSOR(T, S) \
  ArrayRefWrapper(at::ArrayRef<T> values) { \
    at::AutoNonVariableTypeMode non_var_type_mode(true); \
    at::Tensor tensor_ = at::tensor(values, at::TensorOptions().device(at::kCPU).is_variable(false)); \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

#define TENSOR(T, S) \
  ArrayRefWrapper(std::vector<T> values) : ArrayRefWrapper(at::ArrayRef<T>(values)) {}
AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TENSOR)
#undef TENSOR

  // yf225 TODO: we will handle std::vector<bool> separately

  at::Tensor tensor_;
};

inline at::Tensor tensor(ScalarWrapper scalar_wrapper, const at::TensorOptions& options = {}) {
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty({1}, at::TensorOptions(options).device(at::kCPU).is_variable(false));
  })();

  tensor.fill_(scalar_wrapper.scalar_);

  tensor = tensor.to(options.device());

  return autograd::make_variable(tensor, options.requires_grad());
}

inline at::Tensor tensor(ArrayRefWrapper array_ref_wrapper, const at::TensorOptions& options = {}) {
  auto tensor = array_ref_wrapper.tensor_;
  tensor = tensor.to(options);
  return autograd::make_variable(tensor, options.requires_grad());
}

// yf225 TODO: can we take advantage of this to support mixed type? Same question for higher dims
inline at::Tensor tensor(std::initializer_list<ScalarWrapper> init_list, const at::TensorOptions& options = {}) {
  std::vector<int64_t> tensor_sizes;
  if (init_list.begin()->is_scalar_) { // yf225 TODO: we need to prevent mixing is_scalar_=true and is_scalar_=false types
    tensor_sizes = {(int64_t)init_list.size()};
  } else {
    tensor_sizes = {(int64_t)init_list.size(), 1};
  }

  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty(tensor_sizes, at::TensorOptions(options).device(at::kCPU).is_variable(false));
  })();

  size_t index = 0;
  for (const auto& elem : init_list) {
    at::NoGradGuard guard;
    tensor[index].fill_(elem.scalar_);
    index++;
  }

  tensor = tensor.to(options.device());

  return autograd::make_variable(tensor, options.requires_grad());
}

inline at::Tensor tensor(std::initializer_list<std::initializer_list<ScalarWrapper>> init_list, const at::TensorOptions& options = {}) {
  std::vector<int64_t> tensor_sizes;
  if (init_list.begin()->begin()->is_scalar_) { // yf225 TODO: we need to prevent mixing is_scalar=true and is_scalar=false types
    tensor_sizes = {(int64_t)init_list.size(), (int64_t)(init_list.begin()->size())};
  } else {
    tensor_sizes = {(int64_t)init_list.size(), (int64_t)(init_list.begin()->size()), 1};
  }

  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty(tensor_sizes, at::TensorOptions(options).device(at::kCPU).is_variable(false));
  })();

  size_t outer_index = 0;
  for (const auto& outer_elem : init_list) {
    size_t inner_index = 0;
    for (const auto& inner_elem : outer_elem) {
      at::NoGradGuard guard;
      tensor[outer_index][inner_index].fill_(inner_elem.scalar_);
      inner_index++;
    }
    outer_index++;
  }

  tensor = tensor.to(options.device());

  return autograd::make_variable(tensor, options.requires_grad());
}


// yf225 TODO: we need this to make `torch::tensor({{1, 2}})` not ambiguous, because
// `tensor(std::initializer_list<ScalarWrapper> init_list)` and `tensor(ArrayRefWrapper array_ref_wrapper)
// are ambiguous for that
#define TENSOR(T, S) \
inline at::Tensor tensor(std::initializer_list<std::initializer_list<T>> init_list, const at::TensorOptions& options = {}) { \
  std::vector<int64_t> tensor_sizes = {(int64_t)init_list.size(), (int64_t)(init_list.begin()->size())}; \
  at::Tensor tensor = ([&]() { \
    at::AutoNonVariableTypeMode non_var_type_mode(true); \
    return at::empty(tensor_sizes, at::TensorOptions(options).dtype(at::k##S).device(at::kCPU).is_variable(false)); \
  })(); \
  size_t outer_index = 0; \
  for (const auto& outer_elem : init_list) { \
    size_t inner_index = 0; \
    for (const auto& inner_elem : outer_elem) { \
      at::NoGradGuard guard; \
      tensor[outer_index][inner_index].fill_(inner_elem); \
      inner_index++; \
    } \
    outer_index++; \
  } \
  tensor = tensor.to(options.device()); \
  return autograd::make_variable(tensor, options.requires_grad()); \
}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

inline at::Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<ScalarWrapper>>> init_list, const at::TensorOptions& options = {}) {
  std::vector<int64_t> tensor_sizes;
  if (init_list.begin()->begin()->begin()->is_scalar_) { // yf225 TODO: we need to prevent mixing is_scalar=true and is_scalar=false types
    tensor_sizes = {(int64_t)init_list.size(), (int64_t)(init_list.begin()->size()), (int64_t)(init_list.begin()->begin()->size())};
  } else {
    tensor_sizes = {(int64_t)init_list.size(), (int64_t)(init_list.begin()->size()), (int64_t)(init_list.begin()->begin()->size()), 1};
  }

  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty(tensor_sizes, at::TensorOptions(options).device(at::kCPU).is_variable(false));
  })();

  size_t dim0_index = 0;
  for (const auto& dim0_elem : init_list) {
    size_t dim1_index = 0;
    for (const auto& dim1_elem : dim0_elem) {
      size_t dim2_index = 0;
      for (const auto& dim2_elem : dim1_elem) {
        at::NoGradGuard guard;
        tensor[dim0_index][dim1_index][dim2_index].fill_(dim2_elem.scalar_);
        dim2_index++;
      }
      dim1_index++;
    }
    dim0_index++;
  }

  tensor = tensor.to(options.device());

  return autograd::make_variable(tensor, options.requires_grad());
}

// inline at::Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<c10::Scalar>> init_list, const at::TensorOptions& options) {
  
// }

// inline at::Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<c10::Scalar>> init_list, const at::TensorOptions& options) {
  
// }

// inline at::Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<c10::Scalar>> init_list, const at::TensorOptions& options) {
  
// }

// inline at::Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<c10::Scalar>> init_list, const at::TensorOptions& options) {
  
// }

// inline at::Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<c10::Scalar>> init_list, const at::TensorOptions& options) {
  
// }

// inline at::Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<c10::Scalar>> init_list, const at::TensorOptions& options) {
  
// }

// inline at::Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<c10::Scalar>> init_list, const at::TensorOptions& options) {

// }

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
