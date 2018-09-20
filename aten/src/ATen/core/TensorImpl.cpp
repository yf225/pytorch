#include <ATen/core/TensorImpl.h>

#include <ATen/core/optional.h>
#include <ATen/core/Backend.h>
#include <ATen/core/WrapDimMinimal.h>
#include <ATen/core/LegacyTypeDispatch.h>

#include <ATen/core/VariableHooksInterface.h>

namespace at {

TensorImpl::TensorImpl(TensorTypeId type_id, const caffe2::TypeMeta& data_type, Allocator *allocator, bool is_variable)
    : TensorImpl({}, type_id, data_type, is_variable) {
  // UndefinedTensors and SparseTensors don't have storages.
  if (type_id != UndefinedTensorId() && data_type.id() != caffe2::TypeIdentifier::uninitialized()
      && type_id != SparseCPUTensorId() && type_id != SparseCUDATensorId()) {
    storage_ = Storage(data_type, 0, allocator, true);
  }
}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, bool is_variable)
    : TensorImpl(std::move(storage), type_id, storage.dtype(), is_variable) {}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, const caffe2::TypeMeta& data_type, bool is_variable)
    : storage_(std::move(storage)),
      storage_offset_(0),
      sizes_{0},
      strides_{1},
      is_contiguous_(true),
      numel_(0),
      type_id_(type_id),
      data_type_(data_type),
      is_variable_(is_variable) {}

// yf225 TODO: maybe this is bad API, and we should have parameters to specify whether we want to clone StorageImpl and VariableImpl
TensorImpl::TensorImpl(const TensorImpl& tensor_impl)
    : storage_offset_(tensor_impl.storage_offset()),
      sizes_(tensor_impl.sizes().begin(), tensor_impl.sizes().end()),
      strides_(tensor_impl.strides().begin(), tensor_impl.strides().end()),
      is_contiguous_(tensor_impl.is_contiguous()),
      numel_(tensor_impl.numel()),
      type_id_(tensor_impl.type_id()),
      data_type_(tensor_impl.dtype()),
      is_wrapped_number_(tensor_impl.is_wrapped_number()),
      is_variable_(tensor_impl.is_variable()) {
  // yf225 TODO: this could be wrong
  StorageImpl* storage_impl = tensor_impl.storage_.unsafeGetStorageImpl();
  c10::raw::intrusive_ptr::incref(storage_impl);
  storage_ = Storage(c10::intrusive_ptr<StorageImpl>::reclaim(storage_impl));

  // yf225 TODO: this could be wrong
  auto variable_impl = static_cast<intrusive_ptr_target*>(tensor_impl.variable_impl_);
  c10::raw::intrusive_ptr::incref(variable_impl);
  variable_impl_ = variable_impl;
}

// TensorImpl::TensorImpl(Storage&& storage,
//                        int64_t storage_offset,
//                        at::IntList sizes,
//                        at::IntList strides,
//                        bool is_contiguous,
//                        int64_t numel,
//                        TensorTypeId type_id,
//                        const caffe2::TypeMeta& data_type,
//                        bool is_wrapped_number,
//                        bool is_variable)
//     : storage_(std::move(storage)),
//       storage_offset_(storage_offset),
//       sizes_(sizes.begin(), sizes.end()),
//       strides_(strides.begin(), strides.end()),
//       is_contiguous_(is_contiguous),
//       numel_(numel),
//       type_id_(type_id),
//       data_type_(data_type),
//       is_wrapped_number_(is_wrapped_number),
//       is_variable_(is_variable) {}

IntList TensorImpl::sizes() const {
  return sizes_;
}

IntList TensorImpl::strides() const {
  return strides_;
}

bool TensorImpl::compute_contiguous() const {
  bool is_contiguous = true;
  if (is_empty())
    return is_contiguous;
  int64_t z = 1;
  for (int64_t d = dim() - 1; d >= 0; d--) {
    if (size(d) != 1) {
      if (stride(d) == z) {
        z *= size(d);
      } else {
        is_contiguous = false;
        break;
      }
    }
  }
  return is_contiguous;
}

void TensorImpl::release_resources() {
  if (storage_) {
    storage_ = {};
  }
  c10::raw::intrusive_ptr::decref(static_cast<intrusive_ptr_target*>(variable_impl_));
  variable_impl_ = nullptr;
}

int64_t TensorImpl::dim() const {
  return sizes_.size();
}

int64_t TensorImpl::size(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return sizes_[d];
}

int64_t TensorImpl::stride(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return strides_[d];
}

TensorImpl* TensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  bool set_zero_dim = condition_when_zero_dim && this->sizes().size() == 1 && this->size(0) == 1;
  if (set_zero_dim) {
    resize_dim(0);
  }
  return this;
}

const Storage& TensorImpl::storage() const {
  return storage_;
}

} // namespace at
