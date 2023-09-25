from torch.utils._python_dispatch import TorchDispatchMode
import torch

class TrackingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print("here1")
        print(f"func: {func}")
        breakpoint()
        return func(*args, **kwargs)


a = torch.randn(1)

@torch._dynamo.disable()
def g_disabled(a):
    return a + a

def func(a):
    # with TrackingMode():
    # return a.item()
    return g_disabled(a)

print("eager: ")
print(func(a))

print("compiled: ")
print(torch.compile(func)(a))
