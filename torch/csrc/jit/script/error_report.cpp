#include <torch/csrc/jit/script/error_report.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/script/tree.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {
namespace script {

thread_local std::vector<Call> calls;

ErrorReport::ErrorReport(const ErrorReport& e)
    : ss(e.ss.str()),
      context(e.context),
      the_message(e.the_message),
      error_stack(e.error_stack.begin(), e.error_stack.end()) {}

ErrorReport::ErrorReport()
    : context(c10::nullopt), error_stack(calls.begin(), calls.end()) {}
ErrorReport::ErrorReport(SourceRange r)
    : context(std::move(r)), error_stack(calls.begin(), calls.end()) {}

void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {
  calls.back().caller_range = range;
}

ErrorReport::CallStack::CallStack(const std::string& name) {
  calls.push_back({name, c10::nullopt});
}

ErrorReport::CallStack::~CallStack() {
  calls.pop_back();
}

const char* ErrorReport::what() const noexcept {
  std::stringstream msg;
  msg << "\n" << ss.str();
  if (context) {
    msg << ":\n";
    context->highlight(msg);
  } else {
    msg << ".\n";
  }

  if (error_stack.size() > 0) {
    for (auto it = error_stack.rbegin(); it != error_stack.rend() - 1; ++it) {
      auto callee = it + 1;

      msg << "'" << it->fn_name
          << "' is being compiled since it was called from '" << callee->fn_name
          << "'\n";
      if (callee->caller_range) {
        callee->caller_range->highlight(msg);
      } else {
        msg << "<no range>\n";
      }
    }
  }

  the_message = msg.str();
  return the_message.c_str();
}

} // namespace script
} // namespace jit
} // namespace torch
