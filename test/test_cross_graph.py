import torch
import torch._dynamo.config
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch._dynamo.eval_frame import DisableContext, innermost_fn
from torch._dynamo.decorators import _disable
from torch._dynamo.utils import func_read_writes as frws

# TORCH_LOGS="+dynamo,aot,inductor" TORCH_COMPILE_DEBUG=1 python test/test_cross_graph.py


class TestSubmodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, inp):
        return torch.add(self.sub_weight, inp)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        self.register_buffer('buf', torch.randn(4, 4))
        self.submod = TestSubmodule()

    def g1(self, d, e):
        # TODO(yf225): since we own the Python interpreter (aka. Dynamo),
        # can we populate this automatically by walking through all branches?
        with _disable(mod=self, func=self.g1, reads=[d, e, self.weight, self.buf], mutations=[d]):
            d.relu_()
            return self.weight + self.buf + d + e
        # return d

    def g2(self, a, b):
        with _disable(mod=self, func=self.g2, reads=[a, b], mutations=[self.buf]):
            self.buf.relu_()
            return torch.cat(torch.chunk(a + b, 2))

    def f(self, c):
        return c * c

    def forward(self, x, y):
        x.relu_()
        self.buf.relu_()
        z = torch.sigmoid(y)
        return self.g1(x, z) \
            + self.g2(self.weight, torch.sigmoid(y + 1)) \
            + torch.relu(x) \
            + torch.tanh(self.weight) \
            + torch.selu(self.submod.sub_weight) \
            + self.buf \
            + self.f(x) \
            + self.submod(x)

with (
    torch._dynamo.config.patch(
        dynamic_shapes=False,
        capture_dynamic_output_shape_ops=False,
        capture_scalar_outputs=False,
    ),
):
    m = TestModule()
    compiled_m = torch.compile(m, backend="aot_eager", fullgraph=False, dynamic=False)
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    ref = m(x, y)
    actual = compiled_m(x, y)
    assert torch.allclose(ref, actual)

    assert frws[0].is_compiled_func()
    assert frws[0].reads == set(['y', 'x', 'l__self___buf'])
    assert frws[0].mutations == set(['x', 'l__self___buf'])
    assert frws[0].outputs == set(['graph_out_0_1'])

    assert frws[1].is_eager_func()
    assert frws[1].reads == set([
        'x',
        'graph_out_0_1',
        'l__self___weight',
        'l__self___buf',
    ])
    assert frws[1].mutations == set(['x'])
    assert frws[1].outputs == set(['___stack0'])

    assert frws[2].is_compiled_func()
    assert frws[2].reads == set(['y'])
    assert frws[2].mutations == set()
    assert frws[2].outputs == set(['graph_out_0_6'])

    assert frws[3].is_eager_func()
    assert frws[3].reads == set(['l__self__weight', 'graph_out_0_6', 'l__self___buf'])
    assert frws[3].mutations == set(['l__self___buf'])
    assert frws[3].outputs == set(['___stack1'])

    assert frws[4].is_compiled_func()
    assert frws[4].reads == set([
        '___stack0',
        '___stack1',
        'l__self___buf',
        'l__self___submod_sub_weight',
        'l__self___weight',
        'x'
    ])
    assert frws[4].mutations == set()
    assert frws[4].outputs == set()
