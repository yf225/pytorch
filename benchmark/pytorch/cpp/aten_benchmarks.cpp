#include <iostream>
#include <chrono>

#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include "benchmark_utils.h"

static int random_in_range(int min, int max) {
  static bool first = true;
  if (first) {
    srand( time(NULL) ); //seeding for the first time only!
    first = false;
  }
  return min + rand() % (( max + 1 ) - min);
}

static void BM_TensorDim(benchmark::State& state) {
  benchmark_context_setup(state, "BM_TensorDim");
  for (auto _ : state) {
    // Workload setup
    // NOTE: We always need a temporary variable to save the intermediate result of the workload,
    // otherwise repeated calls to the same function might be optimized away by the compiler.
    int64_t res = 0;
    std::vector<int64_t> tensor_sizes = {};
    for (int i = 0; i < random_in_range(1, 5); i++) {
      tensor_sizes.push_back(2);
    }
    auto tmp = at::empty(tensor_sizes, at::TensorOptions(at::kCPU));

    benchmark_iteration_setup(state, "BM_TensorDim");
    if (benchmark_should_run_workload(state)) {
      // Workload
      benchmark::DoNotOptimize(res = tmp.dim());
    }
    benchmark_iteration_teardown(state, "BM_TensorDim");

    // Workload teardown
    std::ostream cnull(0);
    cnull << res;
  }
  benchmark_context_teardown(state, "BM_TensorDim");
}

// yf225 TODO: how do we hide this part and do the runtime subtraction automatically?
// yf225 TODO: how to deal with negative runtime after subtraction? Answer: there is no good way to deal with it. Just warn user that it might happen
// No wipe
BENCHMARK(BM_TensorDim)->Args({WIPE_NO_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
BENCHMARK(BM_TensorDim)->Args({WIPE_NO_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000000);
// Wipe L1
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(100000);
// Wipe L1 + L2 + L3
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_L2_L3_CACHE, DONT_RUN_WORKLOAD})->UseManualTime()->Iterations(1000);
BENCHMARK(BM_TensorDim)->Args({WIPE_L1_L2_L3_CACHE, RUN_WORKLOAD})->UseManualTime()->Iterations(1000);

BENCHMARK_MAIN();
