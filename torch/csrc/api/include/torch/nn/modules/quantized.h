// yf225 TODO: implement https://github.com/pytorch/pytorch/blob/master/torch/jit/quantized.py in C++

/*
class QuantizedLinear(torch.jit.ScriptModule):
    __constants__ = ['scale', 'zero_point']

    def __init__(self, other):
        super(QuantizedLinear, self).__init__()
        self.in_features = other.in_features
        self.out_features = other.out_features
        # Quantize weight and discard the original
        self.weight, self.col_offsets, self.scale, self.zero_point = torch.fbgemm_linear_quantize_weight(
            other.weight.clone().float())
        self.weight = torch.nn.Parameter(self.weight, requires_grad=False)
        self.col_offsets = torch.nn.Parameter(self.col_offsets, requires_grad=False)
        assert other.bias is not None, 'QuantizedLinear requires a bias'
        self.bias = torch.nn.Parameter(other.bias.clone().float())

        self.register_buffer(
            'packed_tensor_ptr',
            torch.fbgemm_pack_quantized_matrix(self.weight.clone(), self.weight.size(1), self.weight.size(0)))

    @torch.jit.script_method
    def _unpack(self):
        self.packed_tensor_ptr.set_(
            torch.fbgemm_pack_quantized_matrix(
                self.weight, self.weight.size(1), self.weight.size(0)))

    @torch.jit.script_method
    def _pack(self):
        self.packed_tensor_ptr.set_(
            torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())

    @torch.jit.script_method
    def forward(self, input):
        out = torch.fbgemm_linear_int8_weight(
            input.float(), self.weight, self.packed_tensor_ptr, self.col_offsets,
            self.scale, self.zero_point, self.bias)
        return out.type_as(input)

    def extra_repr(self):
        repr = 'in_features={in_features}, out_features={out_features}, ' \
               'scale={scale}, zero_point={zero_point}'.format(**self.__dict__)
        return repr
*/

#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>

namespace torch {
namespace nn {
/// yf225 TODO: better comments
class TORCH_API QuantizedLinearImpl : public Cloneable<QuantizedLinearImpl> {
 public:
  QuantizedLinearImpl(Linear linear);

  // yf225 TODO: we are at here
  void reset() override;

  /// Pretty prints the `Linear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  Tensor forward(const Tensor& input);

  /// The options used to configure this module.
  LinearOptions options;

  /// The learned weight.
  Tensor weight;

  /// The learned bias. If `with_bias` is false in the `options`, this tensor is
  /// undefined.
  Tensor bias;
};

/// A `ModuleHolder` subclass for `LinearImpl`.
/// See the documentation for `LinearImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Linear);

} // namespace nn
} // namespace torch
