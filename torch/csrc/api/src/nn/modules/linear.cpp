#include <torch/nn/modules/linear.h>

#include <torch/types.h>
#include <torch/utils.h>
#include <torch/nn/init.h>

#include <cmath>
#include <cstdint>

namespace torch {
namespace nn {
LinearOptions::LinearOptions(int64_t in, int64_t out) : in_(in), out_(out) {}

LinearImpl::LinearImpl(LinearOptions options) : options(options) {
  reset();
}

void LinearImpl::reset() {
  weight =
      register_parameter("weight", torch::empty({options.out_, options.in_}));
  if (options.with_bias_) {
    bias = register_parameter("bias", torch::empty(options.out_));
  }

/*
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
if self.bias is not None:
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)
*/
  init::kaiming_uniform_(weight, /*a=*/std::sqrt(5));
  if (bias.defined()) {
    int64_t fan_in, fan_out;
    std::tie(fan_in, fan_out) = init::_calculate_fan_in_and_fan_out(weight);
    const auto bound = 1 / std::sqrt(fan_in);
    init::uniform_(bias, -bound, bound);
  }
}

void LinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Linear(in=" << options.in_
         << ", out=" << options.out_ << ", with_bias=" << options.with_bias_
         << ")";
}

Tensor LinearImpl::forward(const Tensor& input) {
  AT_ASSERT(!options.with_bias_ || bias.defined());
  return torch::linear(input, weight, bias);
}
} // namespace nn
} // namespace torch
