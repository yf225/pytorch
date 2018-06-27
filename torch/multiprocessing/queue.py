import io
import multiprocessing
import multiprocessing.queues
from .connection import ConnectionWrapper


class Queue(multiprocessing.queues.Queue):

    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, **kwargs)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(multiprocessing.queues.SimpleQueue):

    def _make_methods(self):
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader = ConnectionWrapper(self._reader)
            self._writer = ConnectionWrapper(self._writer)
        super(SimpleQueue, self)._make_methods()
