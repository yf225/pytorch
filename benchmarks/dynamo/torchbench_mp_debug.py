"""
cd /fsx/users/willfeng2/pytorch_yf225/benchmarks/dynamo
PYTHONPATH=/fsx/users/willfeng2/benchmark:$PYTHONPATH HUGGING_FACE_HUB_TOKEN=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 python torchbench_mp_debug.py --performance --training --only llama_v2_7b_16h

PYTHONPATH=/fsx/users/willfeng2/benchmark:$PYTHONPATH HUGGING_FACE_HUB_TOKEN=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 python torchbench_mp_debug.py --performance --training --only stable_diffusion

"""

import argparse
import contextlib

import os
from typing import List, Optional, Tuple

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import _functional_collectives as collectives
from torch.nn.parallel import DistributedDataParallel as DDP
import torchbenchmark

from torchbench import TorchBenchmarkRunner
from common import parse_args, patch_torch_manual_seed, cast_to_fp16

def bench1(
    iter_func,
    iters=20,
    warmup=5,
    profile=False,
    device=None,
    model_name="",
):
    assert device is not None

    dynamo_config.suppress_errors = False

    f_ = iter_func

    repeat = 5
    f = lambda: [(f_(), torch.cuda.synchronize()) for _ in range(repeat)]
    import time

    for _ in range(warmup):
        f()
    torch.cuda.reset_peak_memory_stats(device)
    f()
    torch.cuda.synchronize()
    f_gb = torch.cuda.max_memory_allocated(device) / 1e9

    if profile:
        if dist.get_rank() == 0:
            prof = torch.profiler.profile()
        else:
            prof = contextlib.nullcontext()
        with prof:
            f()
        if dist.get_rank() == 0:
            prof.export_chrome_trace(f"{model_name}.json")
    f_times = []

    for _ in range(iters):
        # Calculate the elapsed time
        torch.cuda.synchronize(device)
        begin = time.time()
        f()
        torch.cuda.synchronize(device)
        f_times.append(time.time() - begin)

    avg_time = sum(f_times)*1000/repeat/len(f_times)
    print(f"{model_name}: avg_time    : {avg_time} ms \t{f_gb}GB")

    return None

def bench2(
    eager_iter_func,
    compiled_iter_func,
    iters=20,
    warmup=5,
    profile=False,
    device=None,
    model_name="",
):
    assert device is not None

    dynamo_config.suppress_errors = False

    # # ====== DEBUG ======
    # g_ = torch._dynamo.explain(f_)
    # if dist.get_rank() == 0:
    #     print(g_)
    # # ====== DEBUG ======

    f_ = eager_iter_func
    g_ = compiled_iter_func

    # g_ = torch.compile(f_)
    repeat = 5
    f = lambda: [(f_(), torch.cuda.synchronize()) for _ in range(repeat)]
    g = lambda: [(g_(), torch.cuda.synchronize()) for _ in range(repeat)]
    import time

    for _ in range(warmup):
        f()
        g()
    torch.cuda.reset_peak_memory_stats(device)
    f()
    torch.cuda.synchronize()
    f_gb = torch.cuda.max_memory_allocated(device) / 1e9
    torch.cuda.reset_peak_memory_stats(device)
    g()
    torch.cuda.synchronize()
    g_gb = torch.cuda.max_memory_allocated(device) / 1e9

    if profile:
        if dist.get_rank() == 0:
            prof = torch.profiler.profile()
        else:
            prof = contextlib.nullcontext()
        with prof:
            f()
        if dist.get_rank() == 0:
            prof.export_chrome_trace(f"{model_name}_eager.json")
        with prof:
            g()
        if dist.get_rank() == 0:
            prof.export_chrome_trace(f"{model_name}_compiled.json")
    f_times = []
    g_times = []

    for _ in range(iters):
        # Calculate the elapsed time
        torch.cuda.synchronize(device)
        begin = time.time()
        f()
        torch.cuda.synchronize(device)
        f_times.append(time.time() - begin)
        torch.cuda.synchronize(device)
        begin = time.time()
        g()
        torch.cuda.synchronize(device)
        g_times.append(time.time() - begin)

    eager_avg_time = sum(f_times)*1000/repeat/len(f_times)
    compiled_avg_time = sum(g_times)*1000/repeat/len(g_times)
    speedup = (eager_avg_time - compiled_avg_time) / eager_avg_time

    print(f"eager    : {eager_avg_time} ms \t{f_gb}GB")
    print(f"compiled : {compiled_avg_time} ms \t{g_gb}GB")
    print(f"speedup: {eager_avg_time/compiled_avg_time}x")

    return None


def run_one_rank(
    my_rank,
    args,
    runner,
):
    print(f"RANK {my_rank} started")

    torch.cuda.set_device(my_rank)
    device = torch.device(f"cuda:{my_rank}")

    os.environ["RANK"] = f"{my_rank}"
    os.environ["WORLD_SIZE"] = f"{args.world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    nccl_options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", pg_options=nccl_options
    )

    (
        _,
        name,
        model,
        example_inputs,
        batch_size,
    ) = runner.load_model("cuda", args.only, batch_size=args.batch_size)

    model, example_inputs = cast_to_fp16(model, example_inputs)

    if args.accuracy:
        torch._inductor.config.fallback_random = True
        if args.only not in {
            "alexnet",
            "Background_Matting",
            "pytorch_CycleGAN_and_pix2pix",
            "pytorch_unet",
            "Super_SloMo",
            "vgg16",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForPreTraining",
            "sam",
        }:
            # some of the models do not support use_deterministic_algorithms
            torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

        # Remove randomeness when torch manual seed is called
        patch_torch_manual_seed()

    model_eager = DDP(
        model,
        device_ids=[my_rank],
        output_device=my_rank,
        bucket_cap_mb=25,  # DDP default value
    )

    # # NOTE: throws `daemonic processes are not allowed to have children` error at `AsyncCompile.warm_pool() -> pool._adjust_process_count()` if we don't set this to 1.
    inductor_config.compile_threads = 1
    torch._inductor.config.triton.cudagraphs = not args.disable_cudagraphs
    if not args.disable_cudagraphs:
        torch.profiler._utils._init_for_cuda_graphs()

    model_compiled = DDP(
        torch.compile(model),
        device_ids=[my_rank],
        output_device=my_rank,
        bucket_cap_mb=args.ddp_bucket_cap_mb_for_compiled
    )

    runner.init_optimizer(name, device, model.parameters())
    runner.model_iter_fn = runner.forward_and_backward_pass

    # bench2(
    #     lambda: runner.model_iter_fn(model_eager, example_inputs),
    #     lambda: runner.model_iter_fn(model_compiled, example_inputs),
    #     profile=args.export_profiler_trace,
    #     device=device,
    #     model_name=f"{args.only}",
    # )
    bench1(
        lambda: runner.model_iter_fn(model_eager, example_inputs, collect_outputs=False),
        profile=args.export_profiler_trace,
        device=device,
        model_name=f"{args.only}_eager"
    )
    bench1(
        lambda: runner.model_iter_fn(model_compiled, example_inputs),
        profile=args.export_profiler_trace,
        device=device,
        model_name=f"{args.only}_compiled"
    )
    torch.cuda.synchronize()
    torch._dynamo.reset()
    print("done!")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--world-size", type=int, default=8)
    args = parse_args()
    args.world_size = 8

    runner = TorchBenchmarkRunner()
    runner.args = args

    processes = []
    for rank in range(args.world_size):
        p = torch.multiprocessing.get_context("spawn").Process(
            target=run_one_rank,
            args=(
                rank,
                args,
                runner,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    print("LAUNCHED")

    for rank, p in enumerate(processes):
        p.join()
        print(f"Rank {rank} exited with {p.exitcode}")

    print("JOINED")


if __name__ == "__main__":
    main()
