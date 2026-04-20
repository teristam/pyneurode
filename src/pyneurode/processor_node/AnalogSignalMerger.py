# from processor_node.BatchProcessor import BatchProcessor
import logging
import time
from collections import Counter

import numpy as np
from scipy import signal

from pyneurode.processor_node.AnalogVisualizer import AnalogVisualizer
from pyneurode.processor_node.BatchProcessor import BatchProcessor
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.RingBuffer.RingBuffer import RingBuffer


class AnalogSignalMerger(BatchProcessor):
    '''
    Class AnalogSignalMerger synchronizes two sources of an analog signal. The lower-sampled signal will be upsampled to match the other source’s data rate. 
    The data of the input message must be a numpy array  
    '''

    def __init__(self, message_type_info:Dict[Message, Tuple[int,int]],
                 interval:float=0.001, internal_buffer_size:int=10000,
                 verbose:bool=True, resample_method:str='resample_poly'):
        '''
        Args:
            message_type_info (Dict[Message, Tuple[int,int]]): A dictionary of messages each containing information on its up and down sampling rates.
            interval (float): Time difference between function calls in seconds. Defaults to 0.001
            internal_buffer_size(int): The size of ring buffer. Defaults to 10000.
            verbose(bool): If True enables logging, else disables. Defaults to True.
            resample_method(str): Resampling method to be used. Can be 'resample_poly' or 'upfirdn'. Defaults to 'resample_poly'.
            
        '''
        
        
        super().__init__(interval, internal_buffer_size, verbose)
        self.message_type_info = message_type_info # store the type of the message that we can receive
        self.message_type_dict = {k:[] for k in message_type_info.keys()} # dictionary of list that store the incoming messages
        self.data_list = {k:None for k in self.message_type_info.keys()} #store the ringbuffer for each data type
        self.resample_method = resample_method
        self.internal_buffer_size = internal_buffer_size
        self.read_head = 0
    
    def startup(self):
        super().startup()

    def process(self,data):
        # read from two sources of data

        # prcoess each message
        for d in data:
            if type(d) in self.message_type_dict:
                self.message_type_dict[type(d)].append(d)
        
        
        # At each analysis interval, oversample every message receive so that they are of the same length
        
        for msg_type, msg_list in self.message_type_dict.items():
            if len(msg_list)>0:
                # First combine all data of the same type

                d = [msg.data for msg in msg_list]
                d = np.concatenate(d)
                
                # upsample
                up, down  = self.message_type_info[msg_type]
                if self.resample_method == 'resample_poly':
                    d = signal.resample_poly(d, up, down, axis=0, padtype='line')
                else:
                    # sample-and-hold
                    h = np.ones(up)
                    d = signal.upfirdn(h, d, up, down,axis=0)
        
                
                # self.log(logging.DEBUG, f'{msg_type} : {d.shape=}')
                
                # write to ring buffer
                if self.data_list[msg_type] is None:
                    self.log(logging.DEBUG, 'creating new ring buffer')
                    self.data_list[msg_type] = RingBuffer((self.internal_buffer_size, d.shape[1]))
                
                # self.log(logging.DEBUG, f'{d.sum(axis=0)=} {msg_type}')
                self.data_list[msg_type].write(d)
                # self.log(logging.DEBUG, f'{self.data_list[msg_type].buffer.sum(axis=0)} {msg_type}')

                self.message_type_dict[msg_type] = [] # clear queue after processing

        # make sure all ring buffer is initiated before procedding
        if any([buffer is None for buffer in self.data_list.values()]):
            return

        # Read from the ring buffers
        curReadHead = min([buffer.absWriteHead for buffer in self.data_list.values()])
        data2read = curReadHead - self.read_head 

        if data2read >0:
            buf_size =[buffer.buffer.sum(axis=0) for buffer in self.data_list.values()]
            spike_train_data = []
            
            for msg_type in self.message_type_dict.keys():
                data = self.data_list[msg_type].readContinous(data2read)
                spike_train_data.append(data)

                # self.log(logging.DEBUG, f'{data} {msg_type} {self.data_list[msg_type].absWriteHead}')
                
            msg_data = np.hstack(spike_train_data)

            self.read_head += data2read

            return Message('synced_data', msg_data)

        

if __name__ == '__main__':
    start  = time.time()

    with ProcessorContext() as ctx:
    
        # simulate change of message data shape
        sine1 = SineTimeSource(0.1, 10, 2, msg_type='message1', sampling_frequency=50)
        sine2 = SineTimeSource(0.5, 5, 3, msg_type='message2', sampling_frequency=100)
        merger = AnalogSignalMerger( {Message1:(2,1), Message2:(1,1)}, interval=0.4)
        
        gui = GUIProcessor()
        visualizer = AnalogVisualizer('Sync data', scale=5)
        gui.register_visualizer(visualizer,['synced_data'])
        
        sink = EchoSink()
        
        sine1.connect(merger)
        sine2.connect(merger)
        merger.connect(sink)
        merger.connect(gui)
        
