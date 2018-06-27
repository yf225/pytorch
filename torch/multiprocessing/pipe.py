import multiprocessing
from .connection import ConnectionWrapper


class Pipe(object):
    """Proxy class for multiprocessing.Pipe which uses ConnectionWrapper to
    wrap connections"""

    def __init__(self, *args, **kwargs):
        c1, c2 = multiprocessing.Pipe(*args, **kwargs)
        return ConnectionWrapper(c1), ConnectionWrapper(c2)
