#include <torch/nn/init.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>

namespace torch {
namespace nn {
namespace init {

double calculate_gain(Nonlinearity nonlinearity, double param) {
  if (nonlinearity == Nonlinearity::Tanh) {
    return 5.0 / 3.0;
  } else if (nonlinearity == Nonlinearity::ReLU) {
    return std::sqrt(2.0);
  } else if (nonlinearity == Nonlinearity::LeakyReLU) {
    return std::sqrt(2.0 / (1 + pow(param, 2)));
  }

  return 1.0;
}

/*
def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out
*/

std::pair<int64_t, int64_t> _calculate_fan_in_and_fan_out(Tensor tensor) {
  const auto dimensions = tensor.dim();
  TORCH_CHECK(
      dimensions >= 2,
      "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");

  int64_t fan_in, fan_out;
  if (dimensions == 2) {  // Linear
    fan_in = tensor.size(1);
    fan_out = tensor.size(0);
  } else {
    const auto num_input_fmaps = tensor.size(1);
    const auto num_output_fmaps = tensor.size(0);
    auto receptive_field_size = 1;
    if (tensor.dim() > 2) {
      receptive_field_size = tensor[0][0].numel();
    }
    fan_in = num_input_fmaps * receptive_field_size;
    fan_out = num_output_fmaps * receptive_field_size;
  }
  return {fan_in, fan_out};
}

/*
def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out
*/

int64_t _calculate_correct_fan(Tensor tensor, FanMode mode) {
  int64_t fan_in, fan_out;
  std::tie(fan_in, fan_out) = _calculate_fan_in_and_fan_out(tensor);
  return (mode == FanMode::FanIn) ? fan_in : fan_out;
}

Tensor constant_(Tensor tensor, Scalar value) {
  NoGradGuard guard;
  return tensor.fill_(value);
}

Tensor dirac_(Tensor tensor) {
  NoGradGuard guard;

  TORCH_CHECK(
      tensor.ndimension() >= 3 && tensor.ndimension() <= 5,
      "Only tensors with 3, 4, or 5 dimensions are supported");

  const auto sizes = tensor.sizes();
  const auto min_dim = std::min(sizes[0], sizes[1]);

  tensor.zero_();
  for (int64_t d = 0; d < min_dim; ++d) {
    switch (tensor.ndimension()) {
      case 3: // Temporal convolution
        tensor[d][d][sizes[2] / 2] = 1;
        break;
      case 4: // Spatial convolution
        tensor[d][d][sizes[2] / 2][sizes[3] / 2] = 1;
        break;
      case 5: // Volumetric convolution
        tensor[d][d][sizes[2] / 2][sizes[3] / 2][sizes[4] / 2] = 1;
        break;
    }
  }

  return tensor;
}

Tensor eye_(Tensor matrix) {
  NoGradGuard guard;
  TORCH_CHECK(
      matrix.ndimension() == 2, "Only tensors with 2 dimensions are supported");
  return torch::eye_out(matrix, matrix.size(0), matrix.size(1));
}

Tensor normal_(Tensor tensor, double mean, double std) {
  NoGradGuard guard;
  return tensor.normal_(mean, std);
}

Tensor ones_(Tensor tensor) {
  NoGradGuard guard;
  return tensor.fill_(1);
}

Tensor orthogonal_(Tensor tensor, double gain) {
  NoGradGuard guard;

  TORCH_CHECK(
      tensor.ndimension() >= 2,
      "Only tensors with 2 or more dimensions are supported");

  const auto rows = tensor.size(0);
  const auto columns = tensor.numel() / rows;
  auto flattened = torch::randn({rows, columns});

  if (rows < columns) {
    flattened.t_();
  }

  // Compute the qr factorization
  Tensor q, r;
  std::tie(q, r) = torch::qr(flattened);
  // Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
  auto d = torch::diag(r, 0);
  auto ph = d.sign();
  q *= ph;

  if (rows < columns) {
    q.t_();
  }

  tensor.view_as(q).copy_(q);
  tensor.mul_(gain);

  return tensor;
}

Tensor sparse_(Tensor tensor, double sparsity, double std) {
  NoGradGuard guard;

  TORCH_CHECK(
      tensor.ndimension() == 2, "Only tensors with 2 dimensions are supported");

  const auto rows = tensor.size(0);
  const auto columns = tensor.size(1);
  const int64_t num_zeros = std::ceil(sparsity * rows);
  tensor.normal_(0, std);
  for (int64_t column = 0; column < columns; ++column) {
    auto row_indices = torch::randperm(rows, tensor.options().dtype(kLong));
    auto zero_indices =
        row_indices.slice(/*dim=*/0, /*start=*/0, /*end=*/num_zeros);
    tensor.index_put_(
        {zero_indices, torch::tensor(column, tensor.options().dtype(kLong))},
        torch::zeros(num_zeros, tensor.options()));
  }

  return tensor;
}

Tensor uniform_(Tensor tensor, double low, double high) {
  NoGradGuard guard;
  return tensor.uniform_(low, high);
}

Tensor kaiming_uniform_(
    Tensor tensor,
    double a,
    FanMode mode,
    Nonlinearity nonlinearity) {
/*
fan = _calculate_correct_fan(tensor, mode)
gain = calculate_gain(nonlinearity, a)
std = gain / math.sqrt(fan)
bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
with torch.no_grad():
    return tensor.uniform_(-bound, bound)
*/

  const auto fan = _calculate_correct_fan(tensor, mode);
  const auto gain = calculate_gain(nonlinearity, a);
  const auto std = gain / std::sqrt(fan);
  const auto bound = std::sqrt(3.0) * std;  // Calculate uniform bounds from standard deviation
  NoGradGuard guard;
  return tensor.uniform_(-bound, bound);
}

Tensor kaiming_normal_(
    Tensor tensor,
    double a,
    FanMode mode,
    Nonlinearity nonlinearity) {
/*
fan = _calculate_correct_fan(tensor, mode)
gain = calculate_gain(nonlinearity, a)
std = gain / math.sqrt(fan)
with torch.no_grad():
    return tensor.normal_(0, std)
*/
  
  const auto fan = _calculate_correct_fan(tensor, mode);
  const auto gain = calculate_gain(nonlinearity, a);
  const auto std = gain / std::sqrt(fan);
  NoGradGuard guard;
  return tensor.normal_(0, std);
}

Tensor xavier_normal_(Tensor tensor, double gain) {
/*
fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

return _no_grad_normal_(tensor, 0., std)
*/
  int64_t fan_in, fan_out;
  std::tie(fan_in, fan_out) = _calculate_fan_in_and_fan_out(tensor);
  const auto std = gain * std::sqrt(2.0 / (fan_in + fan_out));

  NoGradGuard guard;
  return tensor.normal_(0, std);
}

Tensor xavier_uniform_(Tensor tensor, double gain) {
/*
fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

return _no_grad_uniform_(tensor, -a, a)
*/
  int64_t fan_in, fan_out;
  std::tie(fan_in, fan_out) = _calculate_fan_in_and_fan_out(tensor);
  const auto std = gain * std::sqrt(2.0 / (fan_in + fan_out));
  const auto a = std::sqrt(3.0) * std;  // Calculate uniform bounds from standard deviation

  NoGradGuard guard;
  return tensor.uniform_(-a, a);
}

Tensor zeros_(Tensor tensor) {
  NoGradGuard guard;
  return tensor.zero_();
}

} // namespace init
} // namespace nn
} // namespace torch
