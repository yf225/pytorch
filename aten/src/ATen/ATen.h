#pragma once

#include <cstdlib>

constexpr bool test_debug_flag() {
  if (std::getenv("DEBUG") && std::string(std::getenv("DEBUG")) == "1") {
#if !(defined(DEBUG))
    return false;
#endif
  }
  return true;
};

constexpr bool test_ndebug_flag() {
  if (std::getenv("NDEBUG") && std::string(std::getenv("NDEBUG")) == "1") {
#if !(defined(NDEBUG))
    return false;
#endif
  }
  return true;
};

static_assert(test_debug_flag(), "DEBUG is defined as env var but not C macro constant.");
static_assert(test_ndebug_flag(), "NDEBUG is defined as env var but not C macro constant.");

#include <c10/core/Allocator.h>
#include <ATen/CPUGeneral.h>
#include <ATen/Context.h>
#include <ATen/Device.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DimVector.h>
#include <ATen/Dispatch.h>
#include <ATen/Formatting.h>
#include <ATen/Functions.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/TensorGeometry.h>
#include <ATen/TensorOperators.h>
#include <ATen/Type.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <c10/core/Layout.h>
#include <ATen/core/Scalar.h>
#include <c10/core/Storage.h>
#include <ATen/core/TensorMethods.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
