#pragma once

#include <torch/csrc/autograd/grad_mode.h>

#include <cstdint>

namespace torch {

using NoGradGuard = at::NoGradGuard;

/// Sets the global random seed for all newly created CPU and CUDA tensors.
using at::manual_seed;
} // namespace torch
