from torch.utils._python_dispatch import TorchDispatchMode
import torch

class TrackingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print("here1")
        print(f"func: {func}")
        breakpoint()
        return func(*args, **kwargs)


a = torch.randn(1)

def func(a):
    # with TrackingMode():
    return a.item() * a

print("eager: ")
print(func(a))

print("compiled: ")
print(torch.compile(func, backend="aot_eager")(a))
