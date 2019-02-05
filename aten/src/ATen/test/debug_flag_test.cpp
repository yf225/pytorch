#include <gtest/gtest.h>

void testDebugFlag() {
  if (std::getenv("DEBUG") && std::string(std::getenv("DEBUG")) == "1") {
#if !(defined(DEBUG))
    throw std::runtime_error("DEBUG is defined as env var but not C macro constant.")
#endif
  }
  if (std::getenv("NDEBUG") && std::string(std::getenv("NDEBUG")) == "1") {
#if !(defined(NDEBUG))
    throw std::runtime_error("NDEBUG is defined as env var but not C macro constant.")
#endif
  }
};

TEST(TestDebugFlag, TestDebugFlag) {
  testDebugFlag();
}
