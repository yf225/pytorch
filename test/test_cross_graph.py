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
def g1_no_mutation_tuple(d, e):
    d = d.relu()
    return d, e

@disable()
def g1_no_mutation_tensor(d, e):
    d = d.relu()
    return d + e

@disable()
def g2(a, b):
    return torch.cat(torch.chunk(a * b, 2))

global_a = torch.randn(4, 4)

@disable()
def g2_global_var(a, b):
    return torch.cat(torch.chunk(a * b * global_a, 2))


# TODO: cases to handle
# 1. eager function returning a tensor [DONE]
# 2. eager function returning a tuple [DONE]
# 3. eager function returning a scalar [Need unbacked symint support]
# 4. compiled function has mutation [DONE]
# 5. compiled function reads module param via `self.` [DONE]


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))  # torch.randn(4, 4))
        self.register_buffer('buf', torch.randn(1))  # torch.randn(4, 4))
        self.submod = TestSubmodule()

    @disable()
    def f_global_var(self, c):
        return c * c * self.weight

    @disable()
    def f(self, c):
        return c * c

    def forward(self, x):
        x.relu_()
        self.buf.relu_()
        # self.buf.relu_()
        # z = torch.sigmoid(y)
        # return g1_no_mutation(x, z) \
        #     + g2(self.weight, torch.sigmoid(y + 1)) \
        #  * x.sum().item()
        # return g1_no_mutation_tensor(x, x) + torch.relu(x) + g1_no_mutation_tensor(x * 2, x) \
        # return torch.relu(x) + g1_mutation_tuple(x, x)[0] + g1_mutation_tuple(x, x)[1] \
        # return torch.relu(x) + g1_no_mutation_tuple(x, x)[0] + g1_no_mutation_tuple(x, x)[1] \
        # return torch.selu(x) + g1_no_mutation_tensor(x, x) \
        # return torch.relu(x) + g2_global_var(x, x) \
        # return torch.relu(x) + g2(x, x) \
        # return torch.relu(x) + g1_no_mutation_tuple(x, x)[0] \
        # return torch.relu(x) + self.f_global_var(x) \
        # return torch.relu(x) + self.f(x) \
        # return torch.relu(x) * self.weight.sum().item() \
        y = torch.cat(torch.chunk(x, 2))
        return torch.relu(y) * self.weight \
            + torch.tanh(self.weight) \
            + x
            # + torch.selu(self.submod.sub_weight) \
            # + self.buf \
            # + self.f(x) \
            # + self.submod(x)


# # hack to observe behavior
# item_orig = torch.Tensor.item
# def _item(tensor):
#     import traceback
#     traceback.print_stack()
#     return item_orig(tensor)
# torch.Tensor.item = _item

with (
    torch._dynamo.config.patch(
        dynamic_shapes=False,
        capture_dynamic_output_shape_ops=False,
        capture_scalar_outputs=False,
    ),
):
    m = TestModule()
    compiled_m = torch.compile(m, fullgraph=False, dynamic=False)
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    # x = torch.randn(1)
    # y = torch.randn(1)

    # ref = m(x)
    actual = compiled_m(x)
    # assert torch.allclose(ref, actual)
