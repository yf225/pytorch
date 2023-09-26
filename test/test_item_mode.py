from torch.utils._python_dispatch import TorchDispatchMode
import torch

a = torch.randn(1)

@torch._dynamo.disable()
def g_disabled(a):
    a.relu_()
    return a + a

def func(a):
    # with TrackingMode():
    # return a.item()
    b = g_disabled(a) * a
    # a.relu_()
    return b + a

print("eager: ")
# print(func(a))

print("compiled: ")
print(torch.compile(func)(a))
