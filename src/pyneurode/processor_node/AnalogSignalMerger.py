# from processor_node.BatchProcessor import BatchProcessor
import logging
import time
from collections import Counter

import numpy as np
from pyneurode.processor_node.ADCFileReaderSource import ADCFileReaderSource
from pyneurode.processor_node.AnalogVisualizer import AnalogVisualizer
from pyneurode.processor_node.BatchProcessor import BatchProcessor
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.TemplateMatchProcessor import SpikeTrainMessage
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from pyneurode.RingBuffer.RingBufferG import RingBufferG
from scipy import signal

class AnalogSignalMerger(BatchProcessor):
    '''
    Synchronize two source of analoge signal. The lower-sampled signal will be upsampled. The two input message must of different type 
    The data of the input message must be a numpy array  

    This class synchronizes two sources of an analog signal. The lower-sampled signal will be upsampled to match the other source’s data rate.
    '''

    
    def __init__(self, message_type_info:Dict[Message, Tuple[int,int]],
                 interval:float=0.001, internal_buffer_size:int=10000,
                 verbose:bool=True, resample_method:str='resample_poly'):
        # format of message_list: Dict(message_type, Tuple(up, down)
        # resample method can be resamply_poly, or upfirdn
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
        
                
                # print(f'{msg_type} : {d.shape=}')
                
                # write to ring buffer
                if self.data_list[msg_type] is None:
                    self.data_list[msg_type] = RingBuffer((self.internal_buffer_size, d.shape[1]))
                
                self.data_list[msg_type].write(d)
                
                self.message_type_dict[msg_type] = [] # clear queue after processing

        # make sure all ring buffer is initiated before procedding
        if any([buffer is None for buffer in self.data_list.values()]):
            return

        # Read from the ring buffers
        curReadHead = min([buffer.absWriteHead for buffer in self.data_list.values()])
        data2read = curReadHead - self.read_head 

        if data2read >0:

            spike_train_data = [buffer.readContinous(data2read) for buffer in self.data_list.values()]

            msg_data = np.hstack(spike_train_data)

            self.read_head += data2read

            return Message('sync_data', msg_data)

        

if __name__ == '__main__':
    start  = time.time()

    with ProcessorContext() as ctx:
    
        # simulate change of message data shape
        sine1 = SineTimeSource(0.1, 10, 2, msg_type='message1', sampling_frequency=50)
        sine2 = SineTimeSource(0.5, 5, 3, msg_type='message2', sampling_frequency=100)
        merger = AnalogSignalMerger( {Message1:(2,1), Message2:(1,1)}, interval=0.4)
        
        gui = GUIProcessor()
        visualizer = AnalogVisualizer('Sync data', scale=5)
        gui.register_visualizer(visualizer,['sync_data'])
        
        sink = EchoSink()
        
        sine1.connect(merger)
        sine2.connect(merger)
        merger.connect(sink)
        merger.connect(gui)
        
