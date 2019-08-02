import sys
import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

CXX_FLAGS = [] if sys.platform == 'win32' else ['-g', '-Werror']

ext_modules = [
    CppExtension(
    	# yf225 TODO: maybe it doesn't support nested modules
        'cpp_api_parity_test.torch_nn.cpp', ['generated/torch_nn.cpp'],
        extra_compile_args=CXX_FLAGS),
]

# yf225 TODO: make sure to test CUDA tests
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'cpp_api_parity_test.torch_nn.cuda', [
            'generated/torch_nn_cuda.cpp',
            # 'cuda_extension_kernel.cu',
            # 'cuda_extension_kernel2.cu',
        ],
        extra_compile_args={'cxx': CXX_FLAGS,
                            'nvcc': ['-O2']})
    ext_modules.append(extension)

setup(
    name='cpp_api_parity_test',
    packages=['cpp_api_parity_test'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
