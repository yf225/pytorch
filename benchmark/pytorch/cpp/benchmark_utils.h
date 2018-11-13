#include <iostream>
#include <chrono>

#include <benchmark/benchmark.h>
#include <torch/torch.h>

static int WIPE_NO_CACHE = 0;
static int WIPE_L1_CACHE = 1;
static int WIPE_L1_L2_L3_CACHE = 3;

static int DONT_RUN_WORKLOAD = 0;
static int RUN_WORKLOAD = 1;

static std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> testNameToStartTime;
static std::map<std::string, uint32_t*> testNameToWipeBuffer;
static std::map<std::string, double> testNameToOverheadRuntime;

static size_t get_cache_size(int wipe_level) {
  /*
    On c1.small.x86 (E3-1240 V5) (Packet)
    L1 Data 32K (x4)
    L1 Instruction 32K (x4)
    L2 Unified 256K (x4)
    L3 Unified 8192K (x1)
  */
  if (wipe_level == WIPE_NO_CACHE) {
    return 0;
  } else if (wipe_level == WIPE_L1_CACHE) {  // Wipe L1 cache
    return 32 * 1024;
  } else if (wipe_level == WIPE_L1_L2_L3_CACHE) {  // Wipe L1 + L2 + L3 cache
    return (32 + 256 + 8192) * 1024;
  }
}

static uint32_t* wipe_dcache_setup(int wipe_level) {
  uint32_t* wipe_buffer = nullptr;
  size_t wipe_size = get_cache_size(wipe_level);

  if (wipe_buffer == nullptr) {
    wipe_buffer = static_cast<uint32_t*>(malloc(wipe_size));
    AT_ASSERT(wipe_buffer != nullptr);
  }
  uint32_t hash = 0;
  for (uint32_t i = 0; i * sizeof(uint32_t) < wipe_size; i += 8) {
    hash ^= std::rand();
    wipe_buffer[i] = hash;
  }
  /* Make sure compiler doesn't optimize the loop away */
  return wipe_buffer;
}

static void wipe_dcache_teardown(int wipe_level, uint32_t* wipe_buffer) {
  size_t wipe_size = get_cache_size(wipe_level);

  if (wipe_size > 0) {
    std::ostream cnull(0);
    for (uint32_t i = 0; i * sizeof(uint32_t) < wipe_size; i += 8) {
      cnull << wipe_buffer[i];
    }
    free(wipe_buffer);
  }
}

static void benchmark_setup(benchmark::State& state, std::string benchmark_name) {
  int wipe_level = state.range(0);
  testNameToWipeBuffer[benchmark_name] = wipe_dcache_setup(wipe_level);
  testNameToStartTime[benchmark_name] = std::chrono::high_resolution_clock::now();
}

static void benchmark_teardown(benchmark::State& state, std::string benchmark_name) {
  auto end   = std::chrono::high_resolution_clock::now();
  auto start = testNameToStartTime[benchmark_name];
  auto elapsed_seconds =
    std::chrono::duration_cast<std::chrono::duration<double>>(
      end - start);
  state.SetIterationTime(elapsed_seconds.count());
  uint32_t* wipe_buffer = testNameToWipeBuffer[benchmark_name];
  int wipe_level = state.range(0);
  wipe_dcache_teardown(wipe_level, wipe_buffer);
}

static void benchmark_setup(benchmark::State& state, std::string benchmark_name) {

}

static void benchmark_teardown(benchmark::State& state, std::string benchmark_name) {

}

static void benchmark_should_run_workload(benchmark::State& state) {
  return state.range(1) == RUN_WORKLOAD;
}
