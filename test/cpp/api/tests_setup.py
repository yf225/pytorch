import sys
import os
import torch

# yf225 TODO: how do we simplify this file??

def TensorSerializationInteropWithPythonFrontend_setup():
    x1 = torch.ones(5, 5)
    torch.cpp.save(x1, 'tensor_python_saved.pt')
    x2 = torch.ones(5, 5, requires_grad=True)
    torch.cpp.save(x2, 'tensor_python_requires_grad_saved.pt')

    
def TensorSerializationInteropWithPythonFrontend_post_cpp_test():
    x1 = torch.ones(5, 5)
    y1 = torch.cpp.load('tensor_cpp_saved.pt')
    assert x1.shape == y1.shape
    assert torch.allclose(x1, y1)
    assert x1.device == y1.device
    assert not y1.requires_grad

    x2 = torch.ones(5, 5, requires_grad=True)
    y2 = torch.cpp.load('tensor_cpp_requires_grad_saved.pt')
    assert x2.shape == y2.shape
    assert torch.allclose(x2, y2)
    assert x2.device == y2.device
    assert y2.requires_grad


def TensorSerializationInteropWithPythonFrontend_shutdown():
    if os.path.exists('tensor_python_saved.pt'):
        os.remove('tensor_python_saved.pt')
    if os.path.exists('tensor_python_requires_grad_saved.pt'):
        os.remove('tensor_python_requires_grad_saved.pt')
    if os.path.exists('tensor_cpp_saved.pt'):
        os.remove('tensor_cpp_saved.pt')
    if os.path.exists('tensor_cpp_requires_grad_saved.pt'):
        os.remove('tensor_cpp_requires_grad_saved.pt')


def TensorSerializationInteropWithPythonFrontend_CUDA_setup():
    x1 = torch.ones(5, 5).cuda()
    torch.cpp.save(x1, 'tensor_python_cuda_saved.pt')
    x2 = torch.ones(5, 5, requires_grad=True).cuda()
    torch.cpp.save(x2, 'tensor_python_requires_grad_cuda_saved.pt')


def TensorSerializationInteropWithPythonFrontend_CUDA_post_cpp_test():
    x1 = torch.ones(5, 5).cuda()
    y1 = torch.cpp.load('tensor_cpp_cuda_saved.pt')
    assert x1.shape == y1.shape
    assert torch.allclose(x1, y1)
    assert x1.device == y1.device
    assert not y1.requires_grad

    x2 = torch.ones(5, 5, requires_grad=True).cuda()
    y2 = torch.cpp.load('tensor_cpp_requires_grad_cuda_saved.pt')
    assert x2.shape == y2.shape
    assert torch.allclose(x2, y2)
    assert x2.device == y2.device
    assert y2.requires_grad


def TensorSerializationInteropWithPythonFrontend_CUDA_shutdown():
    if os.path.exists('tensor_python_cuda_saved.pt'):
        os.remove('tensor_python_cuda_saved.pt')
    if os.path.exists('tensor_python_requires_grad_cuda_saved.pt'):
        os.remove('tensor_python_requires_grad_cuda_saved.pt')
    if os.path.exists('tensor_cpp_cuda_saved.pt'):
        os.remove('tensor_cpp_cuda_saved.pt')
    if os.path.exists('tensor_cpp_requires_grad_cuda_saved.pt'):
        os.remove('tensor_cpp_requires_grad_cuda_saved.pt')


def setup():
    TensorSerializationInteropWithPythonFrontend_setup()
    TensorSerializationInteropWithPythonFrontend_CUDA_setup()


def post_cpp_test():
    TensorSerializationInteropWithPythonFrontend_post_cpp_test()
    TensorSerializationInteropWithPythonFrontend_CUDA_post_cpp_test()


def shutdown():
    TensorSerializationInteropWithPythonFrontend_shutdown()
    TensorSerializationInteropWithPythonFrontend_CUDA_shutdown()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "setup":
        setup()
    elif command == "post_cpp_test":
        post_cpp_test()
    elif command == "shutdown":
        shutdown()
