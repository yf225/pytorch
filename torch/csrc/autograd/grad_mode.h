#pragma once

#include <ATen/core/grad_mode.h>

namespace torch { namespace autograd {

typedef at::GradMode GradMode;
typedef at::AutoGradMode AutoGradMode;

}}
