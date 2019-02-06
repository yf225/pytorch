#pragma once

#include <ATen/core/grad_mode.h>

namespace torch {

typedef at::NoGradGuard NoGradGuard;

/// Sets the global random seed for all newly created CPU and CUDA tensors.
using at::manual_seed;
} // namespace torch
