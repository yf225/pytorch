#pragma once

// ${generated_comment}

#include "torch/csrc/utils/auto_gil.h"

#include <ATen/ATen.h>

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using at::Tensor;
using at::Scalar;
using at::TensorList;
using at::IntArrayRef;
using at::Generator;
using at::SparseTensorRef;
using at::Storage;

inline Tensor dispatch_to_(Tensor & self, bool non_blocking, bool copy) {
  AutoNoGIL no_gil;
  return self.to_(options, non_blocking, copy);
}

inline Tensor dispatch_to_(Tensor & self, Device device, bool non_blocking, bool copy) {
  AutoNoGIL no_gil;
  // yf225 TODO: consider improving this comment!
  // NOTE: this is where we record aten::to_ in the graph during tracing. However, the behavior of aten::to_
  // is different with respect to TensorOptions fields that are not present: aten::to_ inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to_(self.options().device(device), non_blocking, copy);
}

inline Tensor dispatch_to_(Tensor & self, ScalarType dtype, bool non_blocking, bool copy) {
  AutoNoGIL no_gil;
  return self.to_(dtype, non_blocking, copy);
}

inline Tensor dispatch_to_(Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy) {
  AutoNoGIL no_gil;
  return self.to_(device, dtype, non_blocking, copy);
}

${py_method_dispatch}

}} // namespace torch::autograd
