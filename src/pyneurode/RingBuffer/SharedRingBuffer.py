from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory
from pyneurode.RingBuffer.RingBuffer import *
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
import numpy as np 
import time
from dataclasses import dataclass
from multiprocessing import Value

@dataclass
class SharedRingBufferInfo:
    buffer_shm: SharedMemory
    timestamp_shm: SharedMemory
    absWriteHead: Value
    absRecordHead: Value
    num_channel : Value
    length: Value
    is_valid: Value #whether buffer is in its value form
    dtype: np.dtype

    @property
    def shape(self):
        return (self.length.value, self.num_channel.value)

def allocateSharedMemory(shape, sm_manager, manager, dtype=np.float):

    # need to keep a ref to the shared memory, otherwise when it goes out of scope, the shared memory 
    # address will be deleted
    num_channel = shape[1]
    length = shape[0]

    buffer = np.zeros((length, num_channel),dtype=dtype)
    timestamps = np.zeros((length,1),dtype=dtype)
    buffer_shm = sm_manager.SharedMemory(size=buffer.nbytes)
    timestamp_shm = sm_manager.SharedMemory(size=timestamps.nbytes)


    absWriteHead = manager.Value('i', 0) #have to use manager to create the shared value
    absRecordHead = manager.Value('i', 0)
    num_channel = manager.Value('i',num_channel)
    length = manager.Value('i', length)
    is_valid = manager.Value('i', True)

    # absWriteHead = np.array(1,dtype=np.int)
    # absWriteHead_shm = sm_manager.SharedMemory(size=absWriteHead.nbytes) #the actual size allocated may depends on the memory page size

    # absRecordHead = np.array(1,dtype=np.int)
    # absRecordHead_shm = sm_manager.SharedMemory(size=absRecordHead.nbytes)

    return SharedRingBufferInfo(buffer_shm,timestamp_shm, absWriteHead,
         absRecordHead, num_channel, length, is_valid, dtype)


class SharedRingBuffer(RingBuffer):
    # A ring buffer that uses shared memory, allowing multiple process to access simultaneously

    def __init__(self, ringBuffer_info:SharedRingBufferInfo, writeLock=None, dt=1,start_timestamp=0) -> None:
        #shape is the shape of the ring buffer, as in the shape argument to np.zeros
        # if there will be more than one process writing to the buffer, the writelock must be provided 
        # to ensure safety

        # self.length = ringBuffer_info.length

        # need to keep a ref to the shared memory, otherwise when it goes out of scope
        # it will be deleted
        self.buffer_shm = SharedMemory(ringBuffer_info.buffer_shm.name)
        # self.absWriteHead_shm = SharedMemory(ringBuffer_info.absWriteHead_shm.name)
        # self.absRecordHead_shm = SharedMemory(ringBuffer_info.absRecordHead_shm.name)
        self.timestamp_shm = SharedMemory(ringBuffer_info.timestamp_shm.name)

        self.buffer = np.ndarray((ringBuffer_info.length.value, ringBuffer_info.num_channel.value), dtype=ringBuffer_info.dtype, buffer=self.buffer_shm.buf)
        self.timestamps = np.ndarray((ringBuffer_info.length.value,), dtype=ringBuffer_info.dtype, buffer=self.timestamp_shm.buf)
        # self.absWriteHead = np.ndarray((),dtype=np.int,buffer=self.absWriteHead_shm.buf)
        # self.absRecordHead = np.ndarray((), dtype=np.int, buffer = self.absRecordHead_shm.buf) #record on write
        self._absWriteHead = ringBuffer_info.absWriteHead #avoid circular referencing
        self._absRecordHead = ringBuffer_info.absRecordHead
        self._num_channel = ringBuffer_info.num_channel
        self._length = ringBuffer_info.length
        self._is_valid = ringBuffer_info.is_valid
        self.absReadHead = 0

        # print(self.buffer.shape)
        # self.timestamps = np.zeros((self.length,)) #time stamp of the data, will be tracked automatically
        self.timestamps[0] = start_timestamp

        self.dt = dt #dt of each data point
        self.start_timestamp = start_timestamp
        self.latest_time = self.start_timestamp
        self.isRecordingEnabled = False
        self.dataFile = None
        self.recordingBlockSize = self.length//10 #how many data to write to the disk each time
        self.dtype = ringBuffer_info.dtype
        self.writeLock = writeLock

    @property
    def is_valid(self):
        return self._is_valid.value

    @is_valid.setter
    def is_valid(self,value):
        self._is_valid.value = value

    @property
    def length(self):
        return self._length.value

    @length.setter
    def length(self, value):
        self._length.value = value
    
    @property
    def num_channel(self):
        return self._num_channel.value

    @num_channel.setter
    def num_channel(self, value):
        self._num_channel.value = value

    @property
    def absWriteHead(self):
        #need special to handle the memory mapped value
        return self._absWriteHead.value
    
    @absWriteHead.setter
    def absWriteHead(self,value):
        self._absWriteHead.value = value

    @property
    def absRecordHead(self):
        return self._absRecordHead.value
    
    @absRecordHead.setter
    def absRecordHead(self, value):
        self._absRecordHead.value = value

    def print_buffer(self):
        print(self.buffer)

    def write(self,xs):
        if self.writeLock is not None:
        # Ensure process-safe
            self.writeLock.acquire()
            super().write(xs)
            self.writeLock.release()
        else:
            super().write(xs)

    
    def set_buffer_size(self,shape,dtype=float,dt=1,start_timestamp = 0):
        # reset the buffer size
        self.buffer = np.ndarray(shape, dtype=dtype, buffer=self.buffer_shm.buf)
        self.num_channel = shape[1]
        self.length = shape[0]
        self.buffer[:] = 0
        self.start_timestamp = start_timestamp #clear buffer
        self.timestamps[:] = 0
        self.dt = dt 
        self.length = shape[0]
        self.absWriteHead = 0 #the absolute write head
        self.absReadHead = 0
        self.absRecordHead = 0
        self.isRecordingEnabled = False
        self.dataFile = None
        self.recordingBlockSize = self.length//10 #how many data to write to the disk each time
        self.dtype = dtype

