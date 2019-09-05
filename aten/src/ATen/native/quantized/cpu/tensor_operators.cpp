#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

/*
All comparator operators will be named "<aten op name>_quantized_cpu".
'_out' will be appended for the 'out' variant of the op.

TODO: This is an inefficient implementation that uses `.dequantize`.
      Need a more efficient implementation.
*/

#define DEFINE_COMPARATOR(at_op) \
Tensor& at_op##_out_quantized_cpu(Tensor& out, const Tensor& self, \
                                Scalar other) { \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  auto self_dq = self.dequantize(); \
  return at:: at_op##_out(out, self_dq, other); \
} \
Tensor at_op##_quantized_cpu(const Tensor& self, Scalar other) { \
  auto self_dq = self.dequantize(); \
  return at:: at_op(self_dq, other); \
} \
Tensor& at_op##_out_quantized_cpu(Tensor& out, const Tensor& self, \
                                const Tensor& other) { \
  /* We infer size to make sure the tensors are compatible. */\
  infer_size(self.sizes(), other.sizes()); \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  auto self_dq = self.dequantize(); \
  auto other_dq = other.dequantize(); \
  return at:: at_op##_out(out, self_dq, other_dq); \
} \
Tensor at_op##_quantized_cpu(const Tensor& self, const Tensor& other) { \
  /* We infer size to make sure the tensors are compatible. */\
  infer_size(self.sizes(), other.sizes()); \
  auto self_dq = self.dequantize(); \
  auto other_dq = other.dequantize(); \
  return at:: at_op(self_dq, other_dq); \
}

#define AT_FORALL_OPERATORS(_) \
_(ne)                          \
_(eq)                          \
_(ge)                          \
_(le)                          \
_(gt)                          \
_(lt)                          \

AT_FORALL_OPERATORS(DEFINE_COMPARATOR)

#undef AT_FORALL_OPERATORS
#undef DEFINE_COMPARATOR

}}  // at::native
