#pragma once

#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

// An "Opaque" TensorImpl -- there are no strides and (for now)
// even data() is not supported (thus no pointer arithmetic).

// NOTE: We could allow data() in the future, but would have to ensure pointer
// arithmetic code is properly guarded.
//
// NOTE: This does not support resize_ (and other metadata-changing ops) because of
// `shallow_copy_and_detach`. We would need to define an interface to  "shallow copy"
// in order to add support.

template <typename OpaqueHandle>
struct CAFFE2_API OpaqueTensorImpl : public TensorImpl {
  // public constructor for now...
  OpaqueTensorImpl(at::TensorTypeId type_id, const caffe2::TypeMeta& data_type, c10::Device device,
                   OpaqueHandle opaque_handle, c10::IntArrayRef sizes)
  :   TensorImpl(type_id, data_type, device),
      opaque_handle_(std::move(opaque_handle))
  {
    sizes_ = sizes.vec();
    refresh_numel();
  }

  void release_resources() override {
    TensorImpl::release_resources();
    opaque_handle_ = {};
  }

  IntArrayRef strides() const override {
    AT_ERROR("opaque tensors do not have strides");
  }

  bool is_contiguous() const override {
    AT_ERROR("opaque tensors do not have is_contiguous");
  }

  int64_t stride(int64_t d) const override {
    AT_ERROR("opaque tensors do not have strides");
  }

  void resize_dim(int64_t ndim) override {
    AT_ERROR("opaque tensors do not have resize_dim");
  }

  void set_size(int64_t dim, int64_t new_size) override {
    AT_ERROR("opaque tensors do not have set_size");
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    AT_ERROR("opaque tensors do not have set_stride");
  }

  void set_storage_offset(int64_t storage_offset) override {
    AT_ERROR("opaque tensors do not have set_storage_offset");
  }

  TensorImpl* maybe_zero_dim(bool condition_when_zero_dim) override {
      AT_ERROR("opaque tensors do not support maybe_zero_dim");
  }

  bool has_storage() const override {
    return false;
    }

  const Storage& storage() const override{
    AT_ERROR("opaque tensors do not have storage");
  }

  int64_t storage_offset() const override {
    AT_ERROR("opaque tensors do not have storage");
  }

/**
 * Return a TensorImpl that is a shallow-copy of this TensorImpl.
 * See NOTE [ TensorImpl Shallow-Copying ] for details.
 */
c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach() const override {
  //AT_ASSERT(false);
  auto impl = c10::make_intrusive<OpaqueTensorImpl<OpaqueHandle>>(
    type_id(), dtype(), device(), opaque_handle_, sizes_);
  // TensorImpl general fields
  // Note that some of these fields are not used in opaque tensor code,
  // and we copy them here only for completeness.
  impl->sizes_ = sizes_;
  impl->strides_ = strides_;
  impl->storage_offset_ = storage_offset_;
  impl->is_contiguous_ = is_contiguous_;
  impl->is_wrapped_number_ = is_wrapped_number_;
  impl->reserved_ = reserved_;

  // OpaqueTensorImpl-specific fields (none currently).
  return impl;
}

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   * See NOTE [ TensorImpl Shallow-Copying ] for details.
   */
  void shallow_copy_from(c10::intrusive_ptr<TensorImpl> impl) override {
    auto opaque_impl = static_cast<OpaqueTensorImpl<OpaqueHandle>*>(impl.get());
    type_id_ = opaque_impl->type_id();
    data_type_ = opaque_impl->dtype();
    device_opt_ = opaque_impl->device();
    opaque_handle_ = opaque_impl->opaque_handle_;
    sizes_ = opaque_impl->sizes_;

    // TensorImpl general fields
    // Note that some of these fields are not used in opaque tensor code,
    // and we copy them here only for completeness.
    strides_ = opaque_impl->strides_;
    storage_offset_ = opaque_impl->storage_offset_;
    is_contiguous_ = opaque_impl->is_contiguous_;
    is_wrapped_number_ = opaque_impl->is_wrapped_number_;
    reserved_ = opaque_impl->reserved_;
  }

  OpaqueHandle& unsafe_opaque_handle() {
    return opaque_handle_;
  }

private:
  OpaqueHandle opaque_handle_;
};

} // namespace at
