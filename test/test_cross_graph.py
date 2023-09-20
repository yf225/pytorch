import torch
import torch._dynamo.config
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch._dynamo.eval_frame import DisableContext, innermost_fn

# TORCH_LOGS="+dynamo,aot,inductor" TORCH_COMPILE_DEBUG=1 python test/test_cross_graph.py


# TODO(yf225): dedup with torch._dynamo.disable
def disable(param_reads=[], writes=[]):
    def _disable(fn=None, recursive=True):
        """
        Decorator and context manager to disable TorchDynamo

        If recursive=True, Dynamo is completely skipped on the decorated function
        frame as well as the recursively invoked functions.

        If recursive=False, Dynamo skips frames associated with the function code,
        but still process recursively invoked frames.
        """
        if recursive:
            if fn is not None:
                fn = innermost_fn(fn)
                assert callable(fn)
                return DisableContext(param_reads=param_reads, writes=writes)(fn)
            return DisableContext(param_reads=param_reads, writes=writes)
        else:
            return skip(fn)
    return _disable


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

    # NOTE: input args in `writes=[...]` must be in original order
    @disable(param_reads=["self.weight", "self.buf"], writes=["d"])
    def g1(self, d, e):
        d.relu_()
        return self.weight + self.buf + d + e
        # return d

    @disable(writes=["self.buf"])
    def g2(self, a, b):
        self.buf.relu_()
        return torch.cat(torch.chunk(a, 2))

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
    compiled_m = torch.compile(m, fullgraph=False, dynamic=False)
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    ref = m(x, y)
    actual = compiled_m(x, y)
    assert torch.allclose(ref, actual)
