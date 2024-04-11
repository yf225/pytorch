# Owner(s): ["module: inductor"]

import torch
import torch._export
import torch._inductor
import torch.export._trace
import torch.fx._pytree as fx_pytree

from torch.testing._internal.common_utils import IS_FBCODE

from torch.utils import _pytree as pytree


class WrapperModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class AOTIRunnerUtil:
    @classmethod
    def compile(
        cls,
        model,
        example_inputs,
        options=None,
        dynamic_shapes=None,
        disable_constraint_solver=False,
    ):
        if not isinstance(model, torch.nn.Module):
            model = WrapperModule(model)
        # The exact API is subject to change
        if torch._inductor.config.is_predispatch:
            ep = torch.export._trace._export(
                model, example_inputs, dynamic_shapes=dynamic_shapes, pre_dispatch=True
            )
            gm = ep.module()
        else:
            gm = torch.export._trace._export_to_torch_ir(
                model,
                example_inputs,
                dynamic_shapes=dynamic_shapes,
                disable_constraint_solver=disable_constraint_solver,
                # Disabling this flag, because instead we can rely on the mapping
                # dynamo_flat_name_to_original_fqn which is coming from Dynamo.
                restore_fqn=False,
            )

        with torch.no_grad():
            so_path = torch._inductor.aot_compile(gm, example_inputs, options=options)  # type: ignore[arg-type]

        return so_path

    @classmethod
    def load_runner(cls, device, so_path):
        if IS_FBCODE:
            from .fb import test_aot_inductor_model_runner_pybind

            return test_aot_inductor_model_runner_pybind.Runner(
                so_path, device == "cpu"
            )
        else:
            return (
                torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
                if device == "cpu"
                else torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)
            )

    @classmethod
    def load(cls, device, so_path):
        # TODO: unify fbcode and oss behavior to only use torch._export.aot_load
        if IS_FBCODE:
            runner = AOTIRunnerUtil.load_runner(device, so_path)

            def optimized(*args, **kwargs):
                call_spec = runner.get_call_spec()
                in_spec = pytree.treespec_loads(call_spec[0])
                out_spec = pytree.treespec_loads(call_spec[1])
                flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
                flat_outputs = runner.run(flat_inputs)
                return pytree.tree_unflatten(flat_outputs, out_spec)

            return optimized
        else:
            return torch._export.aot_load(so_path, device)

    @classmethod
    def run(
        cls,
        device,
        model,
        example_inputs,
        options=None,
        dynamic_shapes=None,
        disable_constraint_solver=False,
    ):
        so_path = AOTIRunnerUtil.compile(
            model,
            example_inputs,
            options=options,
            dynamic_shapes=dynamic_shapes,
            disable_constraint_solver=disable_constraint_solver,
        )
        optimized = AOTIRunnerUtil.load(device, so_path)
        return optimized(*example_inputs)

    @classmethod
    def run_multiple(
        cls,
        device,
        model,
        list_example_inputs,
        options=None,
        dynamic_shapes=None,
    ):
        so_path = AOTIRunnerUtil.compile(
            model,
            list_example_inputs[0],
            options=options,
            dynamic_shapes=dynamic_shapes,
        )
        optimized = AOTIRunnerUtil.load(device, so_path)
        list_output_tensors = []
        for example_inputs in list_example_inputs:
            list_output_tensors.append(optimized(*example_inputs))
        return list_output_tensors
