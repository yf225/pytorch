"""
Adapted from fsdp.py in https://github.com/pytorch/pytorch/pull/110609.
"""

"""
TORCH_COMPILE_DEBUG=1 TORCH_LOGS_RANKS=0 torchrun --standalone --nproc_per_node=2 test/distributed/test_dynamo_fsdp.py

TORCH_COMPILE_DEBUG=1 TORCH_LOGS_RANKS=1 torchrun --standalone --nproc_per_node=2 test/distributed/test_dynamo_fsdp.py

TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=2 test/distributed/test_dynamo_fsdp.py
"""
import contextlib
import logging
import os
import sys
import traceback

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo import compiled_autograd
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

torch_log = logging.getLogger("torch")

hidden_dim = 1234


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    torch_log.error(
        "Uncaught exception\n%s",
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    )


sys.excepthook = handle_exception


def init():
    torch.manual_seed(0)
    fsdp_kwargs = {
        "use_orig_params": True,
        "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
    }
    # Expectation:
    # - FWD: 2 all-gathers
    # - BWD: 2 all-gathers + 2 reduce-scatters
    model = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim, device="cuda"),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim, device="cuda"),
    )
    model = FSDP(
        model,
        **fsdp_kwargs,
    )
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, optim


def printing_eager(gm, inputs):
    gm.graph.print_tabular()
    return gm.forward


gpu_id = int(os.environ["LOCAL_RANK"])


def run(model, optim):
    torch.manual_seed(42)
    losses = []
    for _ in range(1):
        optim.zero_grad(set_to_none=True)
        inp = torch.randn((2, hidden_dim), device="cuda", requires_grad=True)
        torch.storage.resize_count_and_loc = {}
        torch_log.warning("FORWARD")
        out = model(inp)
        torch_log.warning("END FORWARD")
        # torch.storage.resize_count_and_loc = {}
        loss = out.sum()
        losses.append(loss)
        torch.storage.resize_count_and_loc = {}
        torch_log.warning("BACKWARD")
        from torchviz import make_dot
        torch_log.warning("OUT GRAPH\n%s", make_dot(loss))
        loss.backward()
        torch_log.warning("END BACKWARD")
        optim.step()
    print(f"losses: {losses}")
    return losses


def main_compiled():
    model, optim = init()

    def compiler_fn(gm):
        torch_log.warning("Compiling autograd?")
        return torch.compile(gm, backend="aot_eager", fullgraph=True)

    torch._dynamo.config.trace_distributed = True

    # with ctx:
    with compiled_autograd.enable(compiler_fn):
        model = torch._dynamo.optimize("aot_eager")(model)
        res = run(model, optim)
    return res

def main_eager():
    model, optim = init()
    res = run(model, optim)
    return res


if __name__ == "__main__":
    import time
    import argparse

    dist.init_process_group(backend="nccl")
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    losses_compiled = main_compiled()
    losses_eager = main_eager()

    for loss_compiled, loss_eager in zip(losses_compiled, losses_eager):
        assert torch.allclose(loss_compiled, loss_eager, rtol=1e-3), f"{loss_compiled} vs {loss_eager}"
    torch_log.warning("res_compiled: %s", res_compiled)
