import torch

def save(obj, f):
    # yf225 TODO: add comment here!
    return torch._C.cpp.save(obj, f)

def load(f):
    # yf225 TODO: add comment here!
    return torch._C.cpp.load(f)
