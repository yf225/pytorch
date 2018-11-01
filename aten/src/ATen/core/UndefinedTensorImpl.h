#pragma once

#include "ATen/core/TensorImpl.h"

namespace at {

struct CAFFE2_API UndefinedTensorImpl final : public TensorImpl {
 public:
  // Without this, we get:
  //  error: identifier "at::UndefinedTensorImpl::_singleton" is undefined in device code
  // (ostensibly because the constexpr tricks MSVC into trying to compile this
  // function for device as well).
#ifdef _WIN32
  static inline TensorImpl * singleton() {
#else
  static constexpr inline TensorImpl * singleton() {
#endif
    return &_singleton;
  }
private:
  UndefinedTensorImpl();
  static UndefinedTensorImpl _singleton;
public:
  friend struct UndefinedType;
};

} // namespace at
