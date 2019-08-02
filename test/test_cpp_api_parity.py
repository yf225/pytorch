# See https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions.py#L173-L305
# for how to use inline cpp extension compilation

import os
import shutil
import sys
import tempfile
from string import Template

import torch
import common_utils as common
# from common_nn import module_tests
import torch.utils.cpp_extension


# yf225 TODO: in order to test all branches in forward logic, should we have a `input_fn` lambda / a `module_init_fn` lambda (that takes a module) for initializing params/buffers?
# yf225 TODO: how do we test non-forward methods? What else needs to be tested? Can we define custom cpp code block using Python string (and reuse the code template) and with matching Python test statements? How do we make sure initialization logic (which can involve rand gen) have parity?
# e.g. reset_parameters / forward
# yf225 TODO: can we make sure to use module_tests from common_nn?? And we can have our own additional tests
torch_nn_tests = [
    dict(
        module_name='Linear',
        python_constructor_args=(20, 5),
        cpp_constructor_args='20, 5',
        input_size=(2, 20),
    ),
    # dict(
    #     module_name='Linear',
    #     python_constructor_args=(10, 8, False),
    #     cpp_constructor_args='torch::nn::LinearOptions(10, 8).with_bias(false)',
    #     input_size=(4, 10),
    #     desc='no_bias',
    # )
]

class TestCppApiParity(common.TestCase):
    # yf225 TODO: do we need this? why do we need to wipe? And does this work on Windows?
    def setUp(self):
        default_build_root = torch.utils.cpp_extension.get_default_build_root()
        if os.path.exists(default_build_root):
            shutil.rmtree(default_build_root)

# yf225 TODO: how to transfer parameters and buffers from Python to C++?
# [Preferred] Option 1: trace as JIT model, and load from C++ side
# Option 2: use `torch.nn.cpp.ModuleWrapper`?
# Option 3: directly pass parameters / buffers / state_dict to C++ modules?
    def test_torch_nn(self):
        # yf225 TODO: do we need to `#include <torch/extension.h>`?
        TORCH_NN_MODULE_WRAPPER = Template("""\
// yf225 TODO: why does this test pass? are we checking the right thing?
void ${module_test_name}_check_reset(const std::string& saved_module_path) {
  torch::nn::${module_name} m1(${cpp_constructor_args});  // NOTE: this should already call reset()

  torch::nn::${module_name} m2(${cpp_constructor_args});
  torch::load(m2, saved_module_path);

  // Check that all parameters are equal
  auto params1 = m1->named_parameters();
  auto params2 = m2->named_parameters();
  assert(params1.size() == params2.size());
  for (auto& param : params1) {
    std::cout << param.key() << std::endl;
    std::cout << param.value() << std::endl;
    assert(param.value().allclose(params2[param.key()]));
  }

  // Check that all buffers are equal
  auto buffers1 = m1->named_buffers();
  auto buffers2 = m2->named_buffers();
  assert(buffers1.size() == buffers2.size());
  for (auto& buffer : buffers1) {
    std::cout << buffer.key() << std::endl;
    std::cout << buffer.value() << std::endl;
    assert(buffer.value().allclose(buffers2[buffer.key()]));
  }
}

void ${module_test_name}_check_forward(
    const std::string& saved_module_path,
    torch::Tensor input,
    torch::Tensor expected_output) {
  torch::nn::${module_name} module(${cpp_constructor_args});
  torch::load(module, saved_module_path);
  assert(module(input).allclose(expected_output));
}
""")
#         TEST_TENSOR_WRAPPER = """\
# torch::Tensor test_tensor_rand() {
#   // yf225 TODO: we can put `manual_seed(2)` here, if needed
#   return torch::randn({2, 3});
# }
# """
        cpp_source = ''
        functions = []
        for test_params in torch_nn_tests:
            if test_params.get('module_name') != 'Linear':
                continue
            module_name = test_params.get('module_name')
            desc = test_params.get('desc', None)
            module_test_name = module_name + (('_' + desc) if desc else '')
            python_constructor_args = test_params.get('python_constructor_args')
            cpp_constructor_args = test_params.get('cpp_constructor_args')
            cpp_source += TORCH_NN_MODULE_WRAPPER.substitute(
                module_test_name=module_test_name,
                module_name=module_name,
                cpp_constructor_args=cpp_constructor_args)
            functions.append(module_test_name+'_check_reset')
            functions.append(module_test_name+'_check_forward')

        # cpp_source += TEST_TENSOR_WRAPPER
        # functions.append('test_tensor_rand')

        print(cpp_source)
        cpp_module = torch.utils.cpp_extension.load_inline(
            name="test_torch_nn",
            cpp_sources=cpp_source,
            functions=functions,
            verbose=True,
        )

        for test_params in torch_nn_tests:
            # yf225 TODO: avoid copy pasta!!
            module_name = test_params.get('module_name')
            desc = test_params.get('desc', None)
            module_test_name = module_name + (('_' + desc) if desc else '')
            input_size = test_params.get('input_size')
            python_constructor_args = test_params.get('python_constructor_args')
            example_input = torch.randn(input_size)

            # Test reset()
            with common.freeze_rng_state():
                torch.manual_seed(2)
                module = getattr(torch.nn, module_name)(*python_constructor_args)
                traced_script_module = torch.jit.trace(module, example_input)
                with tempfile.NamedTemporaryFile() as f:
                    traced_script_module.save(f.name)
                    torch.manual_seed(2)
                    getattr(cpp_module, module_test_name+'_check_reset')(f.name)

            # Test forward()
            # yf225 TODO: combine this case with the reset() test code, and also `manual_seed(2)` when testing forward parity
            module = getattr(torch.nn, module_name)(*python_constructor_args)
            python_output = module(example_input)
            traced_script_module = torch.jit.trace(module, example_input)
            with tempfile.NamedTemporaryFile() as f:
                traced_script_module.save(f.name)
                getattr(cpp_module, module_test_name+'_check_forward')(f.name, example_input, python_output)


        # # yf225 TODO: can use this for param/buffer initialization tests
        # # yf225 TODO: Question: how to compare Python / C++ module params / buffers after initialization?
        # with common.freeze_rng_state():
        #     torch.manual_seed(2)
        #     x = torch.randn(2, 3)
        #     torch.manual_seed(2)
        #     y = getattr(cpp_module, 'test_tensor_rand')()
        #     self.assertEqual(x, y)


'''
def test_inline_jit_compile_extension_with_functions_as_list(self):
    cpp_source = """
    torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
      return x.tanh() + y.tanh();
    }
    """

    module = torch.utils.cpp_extension.load_inline(
        name="inline_jit_extension_with_functions_list",
        cpp_sources=cpp_source,
        functions="tanh_add",
        verbose=True,
    )

    self.assertEqual(module.tanh_add.__doc__.split("\n")[2], "tanh_add")

    x = torch.randn(4, 4)
    y = torch.randn(4, 4)

    z = module.tanh_add(x, y)
    self.assertEqual(z, x.tanh() + y.tanh())
'''

if __name__ == "__main__":
    common.run_tests()