import torch
import torch.nn.functional as F

from cpp_api_parity import wrap_functional, torch_nn_functionals, TorchNNFunctionalMetadata


def sample_functional(input, has_parity, int_option=0, double_option=0.1,
                    bool_option=False, string_option='0', tensor_option=torch.zeros(1),
                    int_or_tuple_option=0):
    if has_parity:
        return input + 1
    else:
        return input + 2

SAMPLE_FUNCTIONAL_CPP_SOURCE = """\n
namespace torch {
namespace nn{
namespace functional {

struct C10_EXPORT sample_functional_options {
  TORCH_ARG(int64_t, int_option) = 0;
  TORCH_ARG(double, double_option) = 0.1;
  TORCH_ARG(bool, bool_option) = false;
  TORCH_ARG(std::string, string_option) = "0";
  TORCH_ARG(torch::Tensor, tensor_option) = torch::zeros({1});
  TORCH_ARG(ExpandingArray<2>, int_or_tuple_option) = 0;
};

torch::Tensor sample_functional(torch::Tensor input, sample_functional_options options = {}) {
  return input + torch::ones_like(input);
}

} // namespace functional
} // namespace nn
} // namespace torch
"""

F.sample_functional = sample_functional

functional_tests = [
    dict(
        fullname='sample_functional_has_parity',
        constructor=wrap_functional(lambda i: F.sample_functional(i, has_parity=True)),
        cpp_options='torch::nn::functional::sample_functional_options()',
        input_size=(3, 4),
        has_parity=True,
    ),
    dict(
        fullname='sample_functional_no_parity',
        constructor=wrap_functional(lambda i: F.sample_functional(i, has_parity=False)),
        cpp_options='torch::nn::functional::sample_functional_options()',
        input_size=(3, 4),
        has_parity=False,
    ),
]

torch_nn_functionals.functional_metadata_map['sample_functional'] = TorchNNFunctionalMetadata(
    cpp_sources=SAMPLE_FUNCTIONAL_CPP_SOURCE,
)
