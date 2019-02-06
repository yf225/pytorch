#pragma once

#include <c10/macros/Macros.h>

namespace at {

struct CAFFE2_API GradMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
//
// This guard also controls whether we are in non-Variable-type mode.
// See NOTE [ Treating Variables as non-Variables in type dispatch ] for details.
struct CAFFE2_API AutoGradMode {
  AutoGradMode(bool enabled) : prev_mode(GradMode::is_enabled()) {
    GradMode::set_enabled(enabled);
  }
  ~AutoGradMode() {
    GradMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

// A RAII, thread local (!) guard that stops future operations from building
// gradients.
struct CAFFE2_API NoGradGuard : public AutoGradMode {
  NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};

} // at
