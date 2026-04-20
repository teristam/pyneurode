'''
Read data from a numpy file
'''

from typing import Any
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from multiprocessing import Event
import random
import pickle
import time
import logging
import numpy as np 
from pyneurode.processor_node.Message import Message
from pyneurode.spike_sorter import simpleDownSample
ADC_CHANNEL = 20

class NumpyMessage(Message):
    dtype = 'numpy_message'
    def __init__(self, data: np.ndarray, timestamp: Optional[float] = None):
        '''
        data should be in the form of time x channel
        '''
        assert isinstance(data, np.ndarray), "Error, data must be a np.ndarray"
        super().__init__(NumpyMessage.dtype, data, timestamp)


class NumpyFileSource(TimeSource):
    '''
    Read from data from a file. It will read data in a batch and send them out
    '''
    def __init__(self, filename:str, interval:float=0, batch_size:int=10):
        super().__init__(interval)
        self.filename = filename
        self.read_idx = 0
        self.batch_size = batch_size
        
    def startup(self):
        self.data = np.load(self.filename)
        self.log(logging.INFO, f'Loaded {self.data.shape}')

    def process(self) -> NumpyMessage :
        if self.read_idx> self.data.shape[0]:
            self.read_idx = 0
        msg = NumpyMessage(self.data[self.read_idx:(self.read_idx+self.batch_size),:])
        self.read_idx += self.batch_size
        
        return msg



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    start  = time.time()

    with ProcessorContext() as ctx:

        fileReader = NumpyFileSource('data/openfield_cells.npy', interval=0.01)
        echo = EchoSink()
        fileReader.connect(echo, [NumpyMessage.dtype])
