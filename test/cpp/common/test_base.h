#include <gtest/gtest.h>

#include <torch/cuda.h>

using namespace c10::cuda;

class TestBase : public ::testing::Test {
 protected:
  virtual void SetUp() {
  	// yf225 TODO: need to gate this with CUDA existence check!
    cuda_memory_allocated_ = CUDACachingAllocator::currentMemoryAllocated(current_device());
  }

  virtual void TearDown() {
  	// yf225 TODO: need to gate this with CUDA existence check!
    ASSERT_EQ(cuda_memory_allocated_, CUDACachingAllocator::currentMemoryAllocated(current_device()));
    std::cout << "HERE1!" << std::endl;
  }

  static uint64_t cuda_memory_allocated_;
};