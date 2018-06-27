import io
import multiprocessing
from multiprocessing.reduction import ForkingPickler
import pickle


class ConnectionWrapper(object):
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects"""

    def __init__(self, conn):
        self.conn = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        if 'conn' in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, 'conn'))


def Pipe(*args, **kwargs):
    """Proxy class for multiprocessing.Pipe which uses ConnectionWrapper to
    wrap connections"""

    c1, c2 = multiprocessing.Pipe(*args, **kwargs)
    return ConnectionWrapper(c1), ConnectionWrapper(c2)
