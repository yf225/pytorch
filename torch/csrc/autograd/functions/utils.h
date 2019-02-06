#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/ATen.h>

#include <functional>
#include <memory>
#include <vector>

namespace torch { namespace autograd {

using function_constructor = std::function<std::shared_ptr<Function>(edge_list&&)>;

/**
 * Wraps the tensor outputs in variables and creates the grad_fn and sets the
 * grad_fn if necessary.
 */
TORCH_API variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                                     const function_constructor& ctr);

///  Checks that inputs contains exactly `args` items and that the first `required_args`
/// items are not nullptr. If not specified, `required_args` defaults to `args`.
TORCH_API void check_input_variables(const char* name, const variable_list& inputs, int args, int required_args=-1);

inline void set_history(
    at::Tensor& variable,
    const std::shared_ptr<Function>& grad_fn) {
  if (grad_fn) {
    if (variable.defined()) {
      auto output_nr =
          grad_fn->add_input_metadata(variable);
      as_variable_ref(variable).set_gradient_edge({grad_fn, output_nr});
    } else {
      grad_fn->add_input_metadata(Function::undefined_input());
    }
  }
}

inline void set_history(
    std::vector<Variable>&& variables,
    const std::shared_ptr<Function>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn);
  }
}

inline void set_history(
    std::vector<Variable>& variables,
    const std::shared_ptr<Function>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn);
  }
}
}}
