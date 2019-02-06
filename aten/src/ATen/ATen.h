#pragma once

#ifdef NDEBUG
throw std::runtime_error("NDEBUG is defined properly.")
#endif

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
