from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn.quantized as nnq
import torch.nn._intrinsic
import torch.nn._intrinsic.qat
from torch.nn.utils import fuse_conv_bn_weights
import torch

class ConvReLU2d(nnq.Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We adopt the same interface as :class:`torch.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.nn.quantized.Conv2d

    """
    _FLOAT_MODULE = torch.nn._intrinsic.ConvReLU2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(ConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode)

    def weight(self):
        return torch.ops.quantized.conv_unpack(self._packed_weight).permute([0, 3, 1, 2])

    def set_weight(self, w):
        self._packed_weight = torch.ops.quantized.conv_prepack(w.permute([0, 2, 3, 1]),
                                                               self.stride,
                                                               self.padding,
                                                               self.dilation,
                                                               self.groups)
        self.weight_scale = w.q_scale()

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        # Temporary work around for bias
        # see Issue:https://github.com/pytorch/pytorch/issues/23874
        bias = self.bias
        if bias is not None:
            bias = torch.quantize_linear(bias.dequantize(), float(self.weight_scale) * input.q_scale(), 0, torch.qint32)
        output = torch.ops.quantized.conv2d_relu(input.permute([0, 2, 3, 1]),
                                                 self._packed_weight, bias,
                                                 self.stride, self.padding,
                                                 self.dilation, self.groups,
                                                 float(self.scale), int(self.zero_point))
        return output.permute([0, 3, 1, 2])

    @classmethod
    def from_float(cls, mod):
        if type(mod) == torch.nn._intrinsic.qat.ConvBnReLU2d:
            mod.weight, mod.bias = \
                fuse_conv_bn_weights(mod.weight, mod.bias, mod.running_mean,
                                     mod.running_var, mod.eps, mod.gamma, mod.beta)
        return super(ConvReLU2d, cls).from_float(mod)
