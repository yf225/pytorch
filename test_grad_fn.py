"""
TORCH_COMPILE_DEBUG=1 python3 test_grad_fn.py >artifacts/run_output.txt 2>&1
"""
import copy
import functools
import contextlib
import logging
import os
import sys
import traceback

import torch
import torch._dynamo
import torch.nn as nn
from torch._dynamo import compiled_autograd


should_resize_storage = True
run_eager = False
run_compiled = True


def print_if_eager(msg):
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        print(msg)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(4, 4))
        self.buffer = torch.tensor(1.)

    def forward(self, x):
        out = torch.matmul(x, self.param)
        # out = x.mul(self.param)
        return out

grad_fns = []

def pre_backward_hook(module, grad) -> None:
    global called
    if any(
        torch._C._will_engine_execute_node(grad_fn)
        for grad_fn in grad_fns
    ):
        with torch.no_grad():
            module.buffer.add_(1)

def post_forward_hook(module, args, output):
    global grad_fns
    output.register_hook(functools.partial(pre_backward_hook, module))
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        assert output.grad_fn is not None
        print(f"output.grad_fn: {output.grad_fn}")
    grad_fns.append(output.grad_fn)

x = torch.randn(4, 4)

device = "cpu"

if __name__ == "__main__":
    x3 = torch.randn(4, 4, device=device, requires_grad=True)

    mod_ref = TestModule()
    mod_test = copy.deepcopy(mod_ref)
    mod_ref.register_forward_hook(post_forward_hook, prepend=False)
    mod_test.register_forward_hook(post_forward_hook, prepend=False)

    out = mod_ref(x3)
    out.sum().backward()
    print(f"eager done: mod_ref.param.grad: {mod_ref.param.grad}, mod_ref.buffer: {mod_ref.buffer}")

    def compiler_fn(gm):
        print("Compiling autograd?")
        return torch.compile(gm, backend="aot_eager", fullgraph=True)
    with compiled_autograd.enable(compiler_fn):
        compiled_mod_test = torch.compile(mod_test, backend="aot_eager", fullgraph=True)
        out = compiled_mod_test(x3)
        out.sum().backward()
    print(f"compiled done: compiled_mod_test.param.grad: {compiled_mod_test.param.grad}, compiled_mod_test.buffer: {compiled_mod_test.buffer}")
    assert torch.allclose(mod_ref.buffer, compiled_mod_test.buffer)
