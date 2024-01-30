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

# ======== REMOVE WHEN READY TO MERGE ========
import argparse
import os
import subprocess
import sys
import urllib
import urllib.parse
import uuid

from typing import Optional

PERFETTO_UI_ROOT_URL = (
    "https://interncache-all.fbcdn.net/manifold/perfetto-artifacts/tree/ui/index.html"
)
MANIFOLD_FOLDER = "perfetto_internal_traces/tree/shared_trace"
DEFAULT_TTL_SEC = 28 * 24 * 60 * 60


def upload_trace_file(local_path: str, overwrite: bool = False) -> Optional[str]:
    file_name = os.path.basename(local_path)
    manifold_path = os.path.join(
        MANIFOLD_FOLDER, f"{os.getlogin()}_{str(uuid.uuid4())}_{file_name}"
    )
    cmd = [
        "manifold",
        "put",
        local_path,
        manifold_path,
        "--ttl",
        str(DEFAULT_TTL_SEC),
        "--userData",
        "false",
    ]
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if ret.returncode == 0:
        print("Uploaded trace successfully.")
        return manifold_path
    else:
        print("[ERROR] Upload failed, maybe the trace file exists.")
        return None


def print_perfetto_ui_url(manifold_path: str) -> None:
    url = (
        PERFETTO_UI_ROOT_URL
        + "#!/?url=https://interncache-all.fbcdn.net/manifold/"
        + urllib.parse.quote_plus(manifold_path)
    )
    print(f"The trace is accessible at:\n{url}")
# ======== REMOVE WHEN READY TO MERGE ========


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
        # "limit_all_gathers": False,
    }
    # Expectation:
    # - FWD: 2 all-gathers
    # - BWD: 2 all-gathers + 2 reduce-scatters
    model = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim, device="cuda"),
        nn.ReLU(),
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
        optim.zero_grad(set_to_none=False)
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


# def main(compiled_fwd, compiled_bwd, aot_eager):
def main_compiled():
    model, optim = init()

    def compiler_fn(gm):
        torch_log.warning("Compiling autograd?")
        return torch.compile(gm, backend="aot_eager", fullgraph=True)

    # ctx = (
    #     compiled_autograd.enable(compiler_fn)
    #     if compiled_bwd
    #     else contextlib.nullcontext()
    # )

    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.trace_distributed = True

    # with ctx:
    model = torch.compile(model, backend="aot_eager", fullgraph=True)
    with compiled_autograd.enable(compiler_fn):
        # if compiled_fwd:
        #     backend = "aot_eager" if aot_eager else "eager"
        #     torch_log.warning("RUNNING COMPILE with backend %s", backend)
        #     torch._dynamo.config.capture_dynamic_output_shape_ops = True
        #     torch._dynamo.config.capture_scalar_outputs = True
        #     torch._dynamo.config.trace_distributed = True
        #     model = torch._dynamo.optimize(backend, nopython=True, dynamic=False)(model)
        #     res = run(model, optim)
        # else:
        res = run(model, optim)
    return res

"""
Observations:
- DDP is "input hook", FSDP is "intermediate hook". See explanation in https://github.com/pytorch/pytorch/pull/109537 PR description or "Note: [On tensor.register_hook]"
- Both DDP and FSDP uses Dynamo to re-trace the compiled autograd graph (search for "torchdynamo start tracing forward <eval_with_key>").
- For FSDP, flat_param size is hidden_dim * hidden_dim + hidden_dim, i.e. weight size + bias size
- Current status:
    - FSDP forward Dynamo output has `hook_handle = torch__dynamo_variables_tensor__register_hook_trampoline(...)`
    - AOTAutograd joint graph has `torch__dynamo__trace_wrapped_higher_order_op_self_invoke`
    - compiled autograd graph has `torch__dynamo__trace_wrapped_higher_order_op_self_invoke`
    - Only 1 compiled autograd graph, which is good
    - "===== Compiled autograd graph =====" is the forward part of the CA graph? it only has 3 torch.mm for a 3 Linears model (shouldn't it be 6?) TODO figure out
    - For FSDP, `_reduce_grad` happens *after* `acc_grad` (see https://arxiv.org/pdf/2304.11277.pdf "Hooks on AccumulateGrad")
    - `.shard()` is called only once, during FSDP(...) init (outside of compile)
    - ._resize_(0) is called via: FSDP __init__ -> _init_param_handle_from_module -> _init_param_handle_from_params -> handle.shard()
    - Problem: There is no mention of `_reduce_grad` in compile log.
    - 1523990 is the full numel for an nn.Linear, 761995 is numel of one shard
    - At beginning of Dynamo tracing, _handle.flat_param is already of shape 761995
    - During compile, when we are tracing `needs_unshard()` and calling `_same_storage_size`, type(a): <class 'torch._subclasses.fake_tensor.FakeTensor'>
a.shape: torch.Size([1523990])
a.untyped_storage().size(): 6095960
b: 1523990, causing `_same_storage_size` to return True and thus causing `_all_gather_flat_param` to not be traced into.
    - During eager mode, calling `_same_storage_size`, a.shape: torch.Size([1523990]), a.untyped_storage().size(): 0, b: 1523990, hence `_same_storage_size` returns True
    - TODO: Voz's last public post is Dec 23. Let's try his commit around that date: https://github.com/pytorch/pytorch/pull/115410/commits
        - Jan 17: 40dfe77f4470fbc00d42f118f444514b9b4eadef -> accumulate_grad_ shape mismatch error (The size of tensor a (1523990) must match the size of tensor b (761995) at non-singleton dimension 0)
        - Dec 28: a527ebe83393f5fb8ae3e2652c3e722358cd02b0 -> doesn't work, assert low > 0 error in bytecode_analysis.py
        - Dec 22: 245977a6c1263dd952144885dbe10b96bdecccda -> doesn't work, assert low > 0 error in bytecode_analysis.py
        - Dec 8: 764c45d25d900e0d289759a3935516130d659ad9 -> doesn't work, `Failed running call_function <function _alloc_storage`
    - TODO: the acc_grad shape mismatch seems related to grad being always torch.zeros(). Why is gradient always 0?
    - We are hitting mid-`return` in `def unshard(self):` in _flat_param.py, and thus not scheduling any all-gather. Reason is Dynamo seems to record the wrong size.
    - TODO: it feels like we are not recording the correct shape
    - TODO: Proper fix sequence:
    - 1. Figure out how to let compile know that storage.size is resized to 0 during eager init, so that compile can trace unshard() properly. (maybe rerun the resize op in FSDP forward func?)
    - 2. Figure out why gradients are torch.zeros instead of proper computation like what DDP tracing does.
    - 3. Figure out why reduce_grad is not in the graph.
    # Expectation:
    # - FWD: 2 all-gathers
    # - BWD: 2 all-gathers + 2 reduce-scatters


4) typed_storage, _resize_(0), _resize_(k)
As part of the family of hacks that FSDP uses in order to get past limitations of how one can use tensors,
and partially abusing the view behaviors of  storage notions, is resize. Specifically, resizing the storage in FSDP
allows it to resize views in a way that a tensor resize does not. *it does not change the size/stride metadata of
all aliases of that tensor* (this is not something functionalization expects / knows how to deal with).
(Source: https://docs.google.com/presentation/d/1K_aeSIVVq5oxkqCUVi77LJJ9xcjVGsx3ZXVpDxrFENE/edit#slide=id.g247865e5b38_0_57)

This was solved with a custom op. This op is actually 150% broken and doing the wrong thing in CUDA right now,
leading to memory bugs, BUT it does let us capture a graph and the op is in the correct place with the seemingly correct schema - so,
it is a matter of fixing the cuda memory allocations, I think.
TODO: search for Voz's PR on `resize_storage_` op:
# File: /scratch/voz/work/pytorch/torch/distributed/utils.py:190, code: tensor.resize_storage_(0)
resize_storage_3: 32[16] = torch.ops.aten.resize_storage.default(copy_2, 0); copy_2 = None
# File: /scratch/voz/work/pytorch/torch/distributed/utils.py:167, code: tensor.resize_storage_(size.numel())
resize_storage_2: 32[16] = torch.ops.aten.resize_storage.default(resize_storage, 16); resize_storage = None


DDP:
- DDP compiled autograd Dynamo re-tracing is able to Dynamo trace into the torch__dynamo_external_utils_call_hook.
    (search for "torch._dynamo.symbolic_convert.__trace_source: [DEBUG]         call_hook = torch__dynamo_external_utils_call_hook(")

FSDP:
- See existing unit test for intermediary hook case: `test_intermediary_hooks_same_on_aot_eager` and `test_intermediary_hooks_same_on_inductor` and others in the same test file.
- For tensors without sources:
   - We don't generate any instructions for registering a hook.
   - Handles from intermediary hooks are NYI.
   - We produce a call function that utilizes the trace_wrapped higher order op, closing over it.
   - We then manually insert the call function above into the graph.
- The handle's exact user-specified name, "user_code_variable_name", is discerned and associated during STORE_FAST.

We use the HOP added in https://github.com/pytorch/pytorch/pull/109690/files (with unit tests), referred to as the HOP below.
- We intercept register_hook calls and wrap the user defined fn in the HOP
- We write a _register_hook_trampoline to the graph that is a local no-arg function that is invoked as a call_function in the dynamo graph
- aot_autograd inlines through it during its trace, and sees the HOP
- the HOP preserves itself in the graph - it does not get traced into
- During backwards, compiled_autograd installs the HOP under a hook call (the logic is likely added in https://github.com/pytorch/pytorch/pull/109537/files)
- When compiled_autograd enters compilation over its generated graph, dynamo traces the contents of the hook

Brian in https://github.com/pytorch/pytorch/pull/109690/files#r1333333919:
It seems like the approach we're taking w.r.t. functionalization for this op is:
(1) Don't functionalize it at all during the joint graph tracing that AOTAutograd runs
(because, well, during joint graph tracing the inner function is an opaque python blob)
(2) When compiled backward tries to compile the backward graph, rely on the backward compiler being another call to AOTAutograd,
that will then run functionalization on the backward graph (and at this point this higher order op has desugared away into ordinary torch ops)
"""

def main_eager():
    model, optim = init()
    res = run(model, optim)
    return res


if __name__ == "__main__":
    import time
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--compiled-fwd', action='store_true', default=True)
    # parser.add_argument('--no-compiled-fwd', action='store_false', dest='compiled_fwd')
    # parser.add_argument('--compiled-bwd', action='store_true', default=True)
    # parser.add_argument('--no-compiled-bwd', action='store_false', dest='compiled_bwd')
    # parser.add_argument('--aot-eager', action='store_true', default=True)
    # parser.add_argument('--no-aot-eager', action='store_false', dest='aot_eager')
    # args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    # res = main(compiled_fwd=args.compiled_fwd, compiled_bwd=args.compiled_bwd, aot_eager=args.aot_eager)

    losses_compiled = main_compiled()

    # profiler_trace_path = "eager_trace.json"
    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    # losses_eager = main_eager()
    # if dist.get_rank() == 0:
    #     prof.export_chrome_trace(profiler_trace_path)
    #     if not os.path.exists(profiler_trace_path):
    #         raise Exception(f"[ERROR] The trace file doesn't exist: {profiler_trace_path}")
    #     manifold_path = upload_trace_file(profiler_trace_path)
    #     if manifold_path:
    #         print_perfetto_ui_url(manifold_path)

    # for loss_compiled, loss_eager in zip(losses_compiled, losses_eager):
    #     assert torch.allclose(loss_compiled, loss_eager, rtol=1e-3), f"{loss_compiled} vs {loss_eager}"
    # torch_log.warning("res_compiled: %s", res_compiled)
