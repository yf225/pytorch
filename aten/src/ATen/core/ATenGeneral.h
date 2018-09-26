#pragma once

#include "ATen/core/Macros.h"

// TODO: Merge the *_API macros.
#define AT_API AT_CORE_API
#define AT_EXPORT AT_CORE_EXPORT
#define AT_IMPORT AT_CORE_IMPORT

// yf225 TODO: move this to VariableHooksInterface.h

namespace at {
struct AT_API GradMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};
}