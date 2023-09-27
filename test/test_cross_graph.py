import torch
import torch._dynamo.config
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch._dynamo import disable

# TORCH_LOGS="+dynamo,aot,inductor" TORCH_COMPILE_DEBUG=1 python test/test_cross_graph.py


class TestSubmodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, inp):
        return torch.add(self.sub_weight, inp)


@disable()
def g1_mutation_tuple(d, e):
    d.relu_()
    return d, e

@disable()
def g1_mutation_tensor(d, e):
    d.relu_()
    return d + e

@disable()
def g2(a, b):
    return torch.cat(torch.chunk(a * b, 2))

global_a = torch.randn(4, 4, device="cuda")

@disable()
def g2_read_global_var(a, b):
    return torch.cat(torch.chunk(a * b.div(torch.selu(global_a)), 2))

@torch._dynamo.disable()
def g2_read_global_var_simple(a, b):
    k = a * b.div(global_a)
    return torch.cat(torch.chunk(k, 2))

# def f(a, b):
#     return a + b + g2_read_global_var_simple(a, b)

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))  # torch.randn(4, 4))
        self.register_buffer('buf', torch.randn(1))  # torch.randn(4, 4))
        self.submod = TestSubmodule()

    @disable()
    def f_read_param_mutate_param(self, c):
        self.buf.relu_()
        return c * c * self.weight

    def forward(self, x, y):
        x.relu_()
        self.buf.relu_()
        y = torch.cat(torch.chunk(y, 2))
        z = torch.relu(x) + g1_mutation_tuple(x, y)[0]
        z = z + g1_mutation_tensor(x, x)
        z = z + g2(x, y)
        z = x + y
        z = z + g2_read_global_var(x, y)
        z = z + self.f_read_param_mutate_param(x)
        z = z + torch.tanh(self.weight)
        z = z + self.buf
        z = z + global_a
        return z


# class TestModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, y):
#         z = x + y
#         z = z + g2_read_global_var_simple(x, y)
#         return z

"""
var_2: x
var_3: y
var_4: self.buf
var_5: torch.relu(x)
var_6: y
var_10: z = torch.relu(x) + g1_mutation_tuple(x, y)[0]
var_14: g1_mutation_tensor(x, x)
var_15: z = z + g1_mutation_tensor(x, x)
var_19: g2(x, y)
var_20: z = z + g2(x, y)
var_24: g2_read_global_var(x, y)
var_25: z = z + g2_read_global_var(x, y)
var_28: self.f_read_param_mutate_param(x)
var_29: global_a
var_30: self.weight
"""

"""
>>> x = torch.randn(4, 4, device="cuda")
>>> x.stride()
(4, 1)
>>> x.data_ptr()
140626427379712
>>> x.storage().data_ptr()
<stdin>:1: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
140626427379712
>>> x.storage()._cdata
100713264
>>> x._cdata
1469156160
>>> y = x[1]
>>> y.data_ptr()
140626427379728
>>> y.storage().data_ptr()
140626427379712
>>> y.storage()._cdata
100713264
>>> y._cdata
115315808
"""

with (
    torch._dynamo.config.patch(
        dynamic_shapes=False,
        capture_dynamic_output_shape_ops=False,
        capture_scalar_outputs=False,
    ),
):
    torch._dynamo.reset()
    m = TestModule()
    m = m.cuda()
    compiled_m = torch.compile(m, fullgraph=False, dynamic=False)
    x = torch.randn(4, 4, device="cuda")
    y = torch.randn(4, 4, device="cuda")

    # ref = m(x, y)
    actual = compiled_m(x, y)
    # assert torch.allclose(ref, actual)
