from multiprocessing.shared_memory import SharedMemory
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import multiprocessing
from dataclasses import dataclass

class RingBufferCore:
    def __init__(self, shape, dtype=float):
        self.buffer = np.zeros(shape,dtype=dtype)
        self.length = shape[0]
        self.absWriteHead = 0 #the absolute write head
        self.absRecordHead = 0

    @property
    def buffer(self):
        return self.buffer

    @buffer.setter
    def buffer(self,value):
        self.buffer = value

    @property
    def absWriteHead(self):
        return self.absWriteHead

    @absWriteHead.setter
    def absWriteHead(self,v):
        self.absWriteHead = v

    @property
    def absRecordHead(self):
        return self.absRecordHead

    @absRecordHead.setter
    def absRecordHead(self,v):
        self.absReadHead = v


