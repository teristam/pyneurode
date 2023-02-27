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
    '''
    
    def __init__(self, message_list:List[Message], interval:float=0.001, internal_buffer_size:int=10000,  verbose:bool=True):
        super().__init__(interval, internal_buffer_size, verbose)
        self.message_type_list = {m:[] for m in message_list }
        self.data_list = [[] for _ in range(len(self.message_type_list))]
    
    def startup(self):
        super().startup()
        self.buffer_length = 100*60*5
        self.agg_buffer = None
        self.adc_buffer = None
        self.spike_buffer = None
        self.readHead = 0
        
    def reset_buffer(self, adc_channels:int=None, spike_train_channels:int=None):
        # reset the internal buffer, usually used when the input shape has changed 
        
        if self.adc_buffer is not None:
            if adc_channels is not None:
                self.adc_buffer = RingBuffer((self.buffer_length, adc_channels))
            else:
                self.adc_buffer = RingBuffer((self.buffer_length, self.adc_buffer.buffer.shape[1]))
        
        if self.spike_buffer is not None:
            if spike_train_channels is not None and self.spike_buffer is not None:
                self.spike_buffer = RingBuffer((self.buffer_length, spike_train_channels))
            else:
                self.spike_buffer = RingBuffer((self.buffer_length, self.spike_buffer.buffer.shape[1]))
            
        self.readHead = 0

    def process(self,data):
        # read from two sources of data

        # prcoess each message
        for d in data:
            if type(d) in self.message_type_list:
                self.message_type_list[type(d)].append(d)
    
        # At each analysis interval, oversample every message receive so that they are of the same length
        
        # First combine all data of the same type
        for i, msg_list in enumerate(self.message_type_list.values()):
            if len(msg_list) > 0:
                d = [msg.data for msg in msg_list]
                self.data_list[i] = np.concatenate(d)
        
        # oversample every message receive so that they are of the same length
        max_len = max([len(x) for x in self.data_list])
        
        for i in range(len(self.data_list)):
            if len(self.data_list[i]) < max_len:
                self.data_list[i] = signal.resample(self.data_list[i], max_len)
                
        # make sure data are now of the same length, and the concatenate them together 
        #TODO work on the case when one of the data is missing

        assert len(set([len(d) for d in self.data_list])) == 1, 'Error: data not of the same length'      
        data_concat = np.hstack(self.data_list)
        
        # clear the current data
        for k in self.message_type_list.keys():
            self.message_type_list[k] = []

        return Message('sync_data', data_concat)
        
        # if adc_data is not None: 
        #     if self.adc_buffer is None:
        #         self.adc_buffer = RingBuffer((self.buffer_length,adc_data.shape[1]))
        #         self.adc_buffer.write(adc_data) #format should be in time x channel
        #     else:
        #         try:
        #             self.adc_buffer.write(adc_data)
        #         except ValueError:
        #             self.log(logging.DEBUG, 'Data shape has changed. Initializing a new buffer')
        #             self.reset_buffer(adc_channels=adc_data.shape[1])
        #             self.adc_buffer.write(adc_data)
            
        # if len(spike_data_list)>0:
        #     # self.log(logging.DEBUG, f'Concatenating {len(spike_data_list)} {spike_data_list[0].shape}')
        #     # check if the spike data match
        #     try:
        #         spike_train = np.concatenate(spike_data_list,axis=0)
        #         # self.log(logging.DEBUG, spike_train.shape)
        #     except ValueError:
        #         self.log(logging.DEBUG, 'Array mismatch, skipping')
                
 
        # # Write to internal buffer
        # if spike_train is not None:
        #     # self.log(logging.DEBUG, f'spike_train shape: {spike_train.shape}')
        #     if self.spike_buffer is None:
        #          # create the a new buffer
        #         self.spike_buffer = RingBuffer((self.buffer_length, spike_train.shape[1]))

        #     try:
        #         self.spike_buffer.write(spike_train)
        #     except ValueError:
        #         # spike train shape has changed, properly the data is resorted, discard the current content
        #         self.log(logging.DEBUG, f'Data shape has changed. Initializing a new buffer : {spike_train.shape} old buffer shape: {self.spike_buffer.buffer.shape}')
        #         self.reset_buffer(spike_train_channels=spike_train.shape[1])
        #         self.spike_buffer.write(spike_train) # time x channel
        
        # msg_data_types = [d.type for d in data]
        # # self.log(logging.DEBUG, f'Processing data: {Counter(msg_data_types)}')

        # if self.spike_buffer is not None:

        #     if self.ignore_adc:
        #         curReadHead = self.spike_buffer.absWriteHead
        #         data2read = curReadHead - self.readHead
                
        #         if data2read > 0:
        #             msg_data = self.spike_buffer.readContinous(data2read)
        #             self.readHead += data2read
        #             return Message('synced_data', msg_data)
  
        #     else:
        #         # Find the minimum of the write head, and then send them
        #         curReadHead = min(self.spike_buffer.absWriteHead, self.adc_buffer.absWriteHead)
        #         data2read = curReadHead - self.readHead 

        #         if data2read >0:

        #             spike_train_data = self.spike_buffer.readContinous(data2read)
        #             adc_data = self.adc_buffer.readContinous(data2read)

        #             msg_data = np.hstack([spike_train_data, adc_data])

        #             self.readHead += data2read

        #             return Message('synced_data', msg_data)


if __name__ == '__main__':
    start  = time.time()

    with ProcessorContext() as ctx:
    
        # simulate change of message data shape
        sine1 = SineTimeSource(0.05, 10, 2, msg_type='message1')
        sine2 = SineTimeSource(0.1, 5, 3, msg_type='message2')
        merger = AnalogSignalMerger( [Message1, Message2], interval=0.2)
        
        gui = GUIProcessor()
        visualizer = AnalogVisualizer('Sync data')
        gui.register_visualizer(visualizer,['sync_data'])
        
        
        
        sink = EchoSink()
        
        sine1.connect(merger)
        sine2.connect(merger)
        merger.connect(sink)
        merger.connect(gui)
