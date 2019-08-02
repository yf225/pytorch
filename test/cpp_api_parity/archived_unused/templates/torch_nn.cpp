#include <torch/extension.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

/*
#pragma once

#include <torch/torch.h>
#include <test/cpp/api/parity/macros.h>

CPP_API_PARITY torch::Tensor test_torch__nn__Linear(torch::Tensor tensor);

${type_derived_method_declarations}
*/

// yf225 TODO: needs improvement (especially figure out how we bind the custom ops. Can we do it without JIT? Should we use the cpp-extensions approach instead?)

/*
std::vector<torch::Tensor> custom_op(
    torch::Tensor tensor,
    double scalar,
    int64_t repeat) {
  std::vector<torch::Tensor> output;
  output.reserve(repeat);
  for (int64_t i = 0; i < repeat; ++i) {
    output.push_back(tensor * scalar);
  }
  return output;
}

int64_t custom_op2(std::string s1, std::string s2) {
  return s1.compare(s2);
}

static auto registry =
    torch::jit::RegisterOperators()
        // We parse the schema for the user.
        .op("custom::op", &custom_op)
        .op("custom::op2", &custom_op2)
        // User provided schema. Among other things, allows defaulting values,
        // because we cannot infer default values from the signature. It also
        // gives arguments meaningful names.
        .op("custom::op_with_defaults(Tensor tensor, float scalar = 1, int repeat = 1) -> Tensor[]",
            &custom_op);

${type_derived_method_definitions}

static auto& registerer = globalATenDispatch()
  ${wrapper_registrations};
}} // namespace torch::autograd
*/
