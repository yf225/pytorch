# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import torch
import torch.distributed as dist
import torch._dynamo
import torch.nn as nn
from torch.distributed._tensor import init_device_mesh, DeviceMesh, Shard, Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module, PrepareModuleInput
# from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
# AC related
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
# AC/selective AC
policy_fn = [
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
    # torch.ops.aten.bmm.default,
    # torch.ops.aten.baddbmm.default,
    # torch.ops.aten._scaled_dot_product_flash_attention.default,
    # torch.ops.aten._scaled_dot_product_efficient_attention.default,
]
class RMSNormPython(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x)
        return output * self.weight
# Directly use PTD FSDP AC wrapper, with selective AC
def sac_checkpoint_wrapper(module):
    from torch.utils.checkpoint import checkpoint, context_fn_gen
    def _get_custom_policy(no_recompute_list=None):
        def _custom_policy(mode, func, *args, **kwargs):
            return func in no_recompute_list
        return _custom_policy
    def selective_checkpointing_context_fn():
        return context_fn_gen(
            _get_custom_policy(no_recompute_list=policy_fn)
        )
    return ptd_checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT, checkpoint_fn=checkpoint, context_fn=selective_checkpointing_context_fn, use_reentrant=False, preserve_rng_state=False)
# Directly use PTD FSDP AC wrapper, with full AC
def checkpoint_wrapper(module):
    from torch.utils.checkpoint import checkpoint
    return ptd_checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT, checkpoint_fn=checkpoint, use_reentrant=False)
class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(16, 16)
        self.wk = nn.Linear(16, 16)
        self.wv = nn.Linear(16, 16)
        self.wo = nn.Linear(16, 16)
    def forward(self, x):
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        # fake attention:
        xo = xq + xk + xv
        return self.wo(xo)
class ToyMLP(nn.Module):
    def __init__(self):
        super(ToyMLP, self).__init__()
        self.net1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 16)
    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()
        self.mlp = ToyMLP()
    def forward(self, x):
        return self.mlp(self.attn(x))
class SimpleTransformer(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.norm = RMSNormPython(16)
        self.layers = nn.ModuleList()
        for layer in range(n_layers):
            block = TransformerBlock()
            self.layers.append(block)
    def forward(self, input):
        h = input
        for i, block in enumerate(self.layers):
            h = block(h)
        h = self.norm(h)
        return h
class TestDTensorCompile(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        fake_store = FakeStore()
        dist.init_process_group(
            "fake", store=fake_store, rank=0, world_size=self.world_size
        )
    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()
    @property
    def device_type(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    @property
    def world_size(self) -> int:
        return 4
    def test_2d_fsdp_tp_compile(self):
        model_parallel_size = 2
        n_layers = 5
        model = SimpleTransformer(n_layers).to(self.device_type)
        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        tp_mesh = mesh_2d["tp"]
        # apply sequence parallel
        for block in model.layers:
            parallel_plan = {
                "attn": PrepareModuleInput(
                    input_layouts=Shard(0),
                    output_layouts=Replicate()
                ),
                "attn.wq": ColwiseParallel(),
                "attn.wk": ColwiseParallel(),
                "attn.wv": ColwiseParallel(),
                "attn.wo": RowwiseParallel(output_layouts=Shard(0)),
                "mlp.net1": ColwiseParallel(input_layouts=Shard(0)),
                "mlp.net2": RowwiseParallel(output_layouts=Shard(0)),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=parallel_plan,
            )
        # model = checkpoint_wrapper(model)
        torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = True
        model = sac_checkpoint_wrapper(model)
        model = torch.compile(model, backend="aot_eager")
        fsdp_mod = FSDP(model, device_mesh=mesh_2d["dp"], use_orig_params=True)
        inp = torch.rand(20, 16).to(self.device_type)
        out = fsdp_mod(inp)
        out.sum().backward()
if __name__ == "__main__":
    run_tests()
