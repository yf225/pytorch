import sys
import os
import torch


def TensorSerializationInteropWithPythonFrontend_setup():
    x = torch.ones(5, 5)
    torch.save(x, 'tensor_python.pt')

    
def TensorSerializationInteropWithPythonFrontend_post_cpp_test():
    x = torch.ones(5, 5)
    y = torch.load('tensor_cpp.pt')
    assert x.shape == y.shape
    assert torch.allclose(x, y)


def TensorSerializationInteropWithPythonFrontend_shutdown():
    if os.path.exists('tensor_python.pt'):
        os.remove('tensor_python.pt')
    if os.path.exists('tensor_cpp.pt')
        os.remove('tensor_cpp.pt')


def setup():
    TensorSerializationInteropWithPythonFrontend_setup()


def post_cpp_test():
    TensorSerializationInteropWithPythonFrontend_post_cpp_test()


def shutdown():
    TensorSerializationInteropWithPythonFrontend_shutdown()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "setup":
        setup()
    elif command == "post_cpp_test":
        post_cpp_test()
    elif command == "shutdown":
        shutdown()
