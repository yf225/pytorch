#pragma once

// ${generated_comment}

#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/EnableNamedTensor.h>

#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace c10 {
class Scalar;
}
namespace at {
struct Generator;
class Tensor;
struct Type;
} // namespace at

namespace at {
namespace native {
namespace detail {
  enum class ListInitTensorType { Scalar, InitList };

  // We use `ListInitTensor` to support converting an arbitrarily nested braced-init-list
  // (e.g. {{1, 2}, {3, 4}}) into the equivalent Tensor, taking advantage of the fact that
  // the constructor will automatically be called recursively until it reaches all innermost
  // scalar values.
  //
  // At any time, a `ListInitTensor` object represents either of the following:
  // 1. A scalar with value `scalar()` and type `scalar_type()`.
  // 2. A Tensor represented in `std::initializer_list<ListInitTensor>` form, with value
  //    `init_list()`, Tensor scalar type `scalar_type()`, and Tensor sizes `sizes()`.
  struct ListInitTensor {
#define TENSOR(T, S)                   \
    ListInitTensor(T scalar) :         \
        scalar_(scalar), init_list_(), \
        sizes_(),                      \
        scalar_type_(at::k##S),        \
        type_(ListInitTensorType::Scalar) {}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR
    ListInitTensor(std::initializer_list<ListInitTensor> init_list) :
        scalar_(),
        init_list_(init_list),
        sizes_(),
        scalar_type_(),
        type_(ListInitTensorType::InitList) {
      TORCH_CHECK(init_list.size() > 0, "Empty init-list is not supported");
      scalar_type_ = init_list.begin()->scalar_type_;
      const ListInitTensor& first_elem = *(init_list.begin());
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

    const std::initializer_list<ListInitTensor>& init_list() const {
      return init_list_;
    }

    const std::vector<int64_t>& sizes() const {
      return sizes_;
    }

    const c10::ScalarType& scalar_type() const {
      return scalar_type_;
    }

    const ListInitTensorType& type() const {
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
      if (type_ == ListInitTensorType::Scalar) {
        AT_DISPATCH_ALL_TYPES_AND3(at::kBool, at::kHalf, at::kBFloat16, scalar_type_, "ListInitTensor_pretty_print_scalar", [&] {
          stream << scalar_.to<scalar_t>();
        });
      } else if (type_ == ListInitTensorType::InitList) {
        stream << "{";
        for (const ListInitTensor* it = init_list_.begin(); it != init_list_.end(); it++) {
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
        if (elem.type_ == ListInitTensorType::Scalar) {
          at::NoGradGuard guard;
          tensor[index].fill_(elem.scalar());
        } else if (elem.type_ == ListInitTensorType::InitList) {
          elem.fill_tensor(tensor[index]);
        } else {
          TORCH_INTERNAL_ASSERT(false, "Invalid ListInitTensor");
        }
        index++;
      }
    }
    c10::Scalar scalar_;
    std::initializer_list<ListInitTensor> init_list_;
    std::vector<int64_t> sizes_;
    c10::ScalarType scalar_type_;
    ListInitTensorType type_;
  };

  inline std::ostream& operator<<(std::ostream& stream, const ListInitTensor& list_init_tensor) {
    list_init_tensor.pretty_print_recursive(stream);
    return stream;
  }
} // namespace detail

// These functions are defined in native/TensorFactories.cpp.
#define TENSOR(T, S)                                                          \
  CAFFE2_API Tensor tensor(ArrayRef<T> values, const TensorOptions& options); \
  inline Tensor tensor(                                                       \
      std::initializer_list<T> values, const TensorOptions& options) {        \
    return native::tensor(ArrayRef<T>(values), options);                      \
  }                                                                           \
  inline Tensor tensor(T value, const TensorOptions& options) {               \
    return native::tensor(ArrayRef<T>(value), options);                       \
  }                                                                           \
  inline Tensor tensor(ArrayRef<T> values) {                                  \
    return native::tensor(std::move(values), at::dtype(k##S));                \
  }                                                                           \
  inline Tensor tensor(std::initializer_list<T> values) {                     \
    return native::tensor(ArrayRef<T>(values));                               \
  }                                                                           \
  inline Tensor tensor(T value) {                                             \
    return native::tensor(ArrayRef<T>(value));                                \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

inline Tensor tensor(detail::ListInitTensor list_init_tensor, const at::TensorOptions& options) {
  return list_init_tensor.to_tensor(options);
}

inline Tensor tensor(detail::ListInitTensor list_init_tensor) {
  return native::tensor(list_init_tensor, at::dtype(list_init_tensor.scalar_type()));
}

${native_function_declarations}

} // namespace native
} // namespace at
