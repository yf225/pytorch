#include "ATen/core/UndefinedTensorImpl.h"
#include "c10/util/Exception.h"

namespace at {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensorImpl::UndefinedTensorImpl()
: TensorImpl(UndefinedTensorId(), caffe2::TypeMeta(), nullptr, /* is variable */ false) {
}

UndefinedTensorImpl UndefinedTensorImpl::_singleton;

}
