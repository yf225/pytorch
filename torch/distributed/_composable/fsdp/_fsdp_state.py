import functools

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.graph import register_multi_grad_hook

from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.utils import _to_kwargs
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils.hooks import RemovableHandle
from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import _cast_fp_tensor, TrainingState
from ._fsdp_param import FSDPParam
from ._fsdp_param_group import FSDPCommContext, FSDPParamGroup, param_group_pre_forward


class FSDPStateContext:
    """This has state shared across FSDP states."""

    def __init__(self):
        # All FSDP states in the root state's module tree
        self.all_states: List[FSDPState] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward
        self.iter_forward_root: Optional[FSDPState] = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False


def _fsdp_state_pre_forward(
    fsdp_state, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    # When composing with module-hook-based activation checkpointing, the
    # the pre-backward hook is responsible for the unshard
    if fsdp_state._training_state == TrainingState.PRE_BACKWARD:
        return args, kwargs
    fsdp_state._training_state = TrainingState.FORWARD
    args, kwargs = fsdp_state._root_pre_forward(module, args, kwargs)
    # TODO(yf225): "Dynamic control flow is not supported at the moment"
    # if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
    #     with torch.profiler.record_function("FSDP::cast_forward_inputs"):
    #         cast_fn = functools.partial(
    #             _cast_fp_tensor, self._mp_policy.param_dtype
    #         )
    #         args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
    # if self._fsdp_param_group:
    args, kwargs = param_group_pre_forward(fsdp_state._fsdp_param_group, module, args, kwargs)
    return args, kwargs


def _fsdp_state_post_forward(fsdp_state, module: nn.Module, input: Any, output: Any) -> Any:
    # When composing with module-hook-based activation checkpointing, the
    # post-backward hook is responsible for the reshard
    if fsdp_state._training_state == TrainingState.PRE_BACKWARD:
        return output
    # if self._fsdp_param_group:  # TODO(yf225): Dynamic control flow is not supported at the moment
    output = fsdp_state._fsdp_param_group.post_forward(module, input, output)
    output = fsdp_state._register_pre_backward_hook(output)
    fsdp_state._training_state = TrainingState.IDLE
    if fsdp_state._state_ctx.iter_forward_root is fsdp_state:
        if all_gather_state := fsdp_state._comm_ctx.all_gather_state:
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                fsdp_state._comm_ctx.all_gather_copy_in_stream.wait_event(
                    all_gather_state.event
                )
                fsdp_state._comm_ctx.all_gather_stream.wait_event(all_gather_state.event)
            fsdp_state._comm_ctx.all_gather_state = None  # free the all-gather result
        fsdp_state._state_ctx.iter_forward_root = None
    if fsdp_state._mp_policy.output_dtype is not None:
        with torch.profiler.record_function("FSDP::cast_forward_outputs"):
            output = tree_map(
                functools.partial(_cast_fp_tensor, fsdp_state._mp_policy.output_dtype),
                output,
            )
    return output


def _fsdp_state_pre_backward(fsdp_state, *unused: Any) -> None:
    fsdp_state._training_state = TrainingState.PRE_BACKWARD
    fsdp_state._register_root_post_backward_final_callback()
    if fsdp_state._fsdp_param_group:
        fsdp_state._fsdp_param_group.pre_backward(*unused)


class FSDPState(_State):
    def __init__(self):
        super().__init__()
        self._fsdp_param_group: Optional[FSDPParamGroup] = None
        self._is_root: Optional[bool] = None  # root set during lazy init
        self._state_ctx = FSDPStateContext()
        self._comm_ctx = FSDPCommContext()
        self._training_state: TrainingState = TrainingState.IDLE
        self._pre_forward_hook_handle: Optional[RemovableHandle] = None
        self._pre_backward_hook_handles: List[RemovableHandle] = []
        # Shared post-forward order for explicit backward prefetching
        self._post_forward_order: List[FSDPParamGroup] = []  # will cause ref cycles

    # Define a separate init since `__init__` is called in the contract
    def init(
        self, module: nn.Module, device: torch.device, mp_policy: MixedPrecisionPolicy
    ) -> None:
        _insert_module_state(module, self)
        self._module = module
        self._device = device
        self._mp_policy = mp_policy
        self._pre_forward_hook_handle = module.register_forward_pre_hook(
            functools.partial(_fsdp_state_pre_forward, self), prepend=True, with_kwargs=True
        )
        self._post_forward_hook_handle = module.register_forward_hook(
            functools.partial(_fsdp_state_post_forward, self), prepend=False
        )

    def _root_pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        self._state_ctx.iter_forward_root = self
        with torch.profiler.record_function("FSDP::root_pre_forward"):
            # Wait for optimizer before implicitly prefetched all-gathers
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                current_stream = torch.cuda.current_stream()
                self._comm_ctx.all_gather_copy_in_stream.wait_stream(current_stream)
                self._comm_ctx.all_gather_stream.wait_stream(current_stream)
            if self._device.type == "cuda":
                with torch.profiler.record_function("FSDP::inputs_to_device"):
                    args_tuple, kwargs_tuple = _to_kwargs(
                        args, kwargs, self._device, False
                    )  # same as DDP
                args, kwargs = args_tuple[0], kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(self, compile=False) -> None:
        """
        Lazy initialization logically represents when all modules' parallelisms
        have finalized (e.g. FSDP has been applied to all desired modules).
        This means that we can determine root state. We identify the root by
        the 1st state to run forward.
        """
        if self._is_root is not None:
            return  # no-op: already initialized
        self._is_root = True
        root_module = self._module
        for module_name, module in root_module.named_modules():
            if (state := _get_module_fsdp_state(module)) is None:
                continue
            if module is not root_module:
                if state._is_root is not None:
                    raise RuntimeError(
                        "FSDP state has already been lazily initialized for "
                        f"{module_name}\nFSDP requires running forward through "
                        "the root module first"
                    )
                state._is_root = False
            self._state_ctx.all_states.append(state)
            if state._fsdp_param_group:
                state._fsdp_param_group.lazy_init()
        if self._fsdp_param_group:
            # For the root, do not reshard after forward since for training,
            # the parameters would be freed and all-gathered immediately
            self._fsdp_param_group.post_forward_mesh_info = None
        self._init_fqns()
        self._init_shared_state(compile=compile)

    def _init_shared_state(self, compile=False) -> None:
        self._comm_ctx.init(compile=compile)
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._comm_ctx = self._comm_ctx
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.comm_ctx = self._comm_ctx

    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
        assert self._is_root
        root_module = self._module
        param_to_fsdp_param: Dict[nn.Parameter, FSDPParam] = {}
        module_to_fsdp_param_group: Dict[nn.Module, FSDPParamGroup] = {}
        for state in self._state_ctx.all_states:
            if fsdp_param_group := state._fsdp_param_group:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    param_to_fsdp_param[fsdp_param.sharded_param] = fsdp_param
                module_to_fsdp_param_group[fsdp_param_group.module] = fsdp_param_group
        for param_name, param in root_module.named_parameters():
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        for module_name, module in root_module.named_modules():
            if module in module_to_fsdp_param_group:
                module_to_fsdp_param_group[module]._module_fqn = module_name

    def _root_post_backward_final_callback(self) -> None:
        with torch.profiler.record_function("FSDP::root_post_backward_callback"):
            self._training_state = TrainingState.IDLE
            for state in self._state_ctx.all_states:
                state._training_state = TrainingState.IDLE
                if state._fsdp_param_group:
                    state._fsdp_param_group.finalize_backward()
            self._state_ctx.post_backward_final_callback_queued = False
            for handle in self._pre_backward_hook_handles:
                handle.remove()
            self._pre_backward_hook_handles.clear()
            self._comm_ctx.post_forward_order.clear()

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output

        flat_outputs, _ = tree_flatten(output)
        # tensors = tuple(t for t in flat_outputs if t.requires_grad)  # TODO(yf225): Dynamic control flow is not supported at the moment
        tensors = flat_outputs
        if tensors:
            # TODO(yf225): Error: "call_function args: ListVariable() FunctoolsPartialVariable() ConstantVariable(str)"
            # handle = register_multi_grad_hook(tensors, functools.partial(_fsdp_state_pre_backward, self), mode="any")
            for tensor in tensors:
                handle = tensor.register_hook(functools.partial(_fsdp_state_pre_backward, self))
                self._pre_backward_hook_handles.append(handle)
            if self._fsdp_param_group:
                self._fsdp_param_group.expected_backward_unshard_count += 1
        return output

    def _register_root_post_backward_final_callback(self):
        if self._state_ctx.post_backward_final_callback_queued:
            return
        self._state_ctx.post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )


def _get_module_fsdp_state(module: nn.Module) -> Optional[FSDPState]:
    state = _get_module_state(module)
    if isinstance(state, FSDPState):
        return state
    return None
