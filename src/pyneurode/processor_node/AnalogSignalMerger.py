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
    
    def __init__(self, message_list:List[Message], interval:float=0.001, internal_buffer_size:int=10000,
                 verbose:bool=True, resample_method='resample_poly'):
        # resample method can be resample, resamply_poly, or upfirdn
        super().__init__(interval, internal_buffer_size, verbose)
        self.message_type_list = message_list # store the type of the message that we can receive
        self.message_type_dict = {msg:[] for msg in message_list} # dictionary of list that store the incoming messages
        self.data_list = [[] for _ in range(len(self.message_type_list))] # data from the incoming messages, in the order of the message_type_list
        self.resample_method = resample_method
    
    def startup(self):
        super().startup()

    def process(self,data):
        # read from two sources of data

        # prcoess each message
        for d in data:
            if type(d) in self.message_type_dict:
                self.message_type_dict[type(d)].append(d)
        
        
    
        # At each analysis interval, oversample every message receive so that they are of the same length
        
        # First combine all data of the same type
        for i, msg_type in enumerate(self.message_type_list):
            if len(self.message_type_dict[msg_type]) > 0:
                d = [msg.data for msg in self.message_type_dict[msg_type]]
                self.data_list[i] = np.concatenate(d)
                print(f'data_list {i} {self.data_list[i].shape}')
        
        # oversample every message receive so that they are of the same length
        max_len = max([x.shape[0] for x in self.data_list])
        
        for i in range(len(self.data_list)):
            if len(self.data_list[i]) < max_len:
                if self.resample_method == 'resample_poly':
                    self.data_list[i] = signal.resample_poly(self.data_list[i], max_len, len(self.data_list[i]), axis=0, padtype='line')
                elif self.resample_method =='resample':
                    self.data_list[i] = signal.resample(self.data_list[i], max_len, axis=0)
                else:
                    h = np.ones((max_len,))
                    self.data_list[i] = signal.upfirdn(h, self.data_list[i], max_len, self.data_list[i].shape[0],axis=0)
                    if self.data_list[i].shape[0] < max_len:
                        self.data_list[i] = np.pad(self.data_list[i], ((0, max_len-self.data_list[i].shape[0]),(0,0)), mode='maximum')
                
        # make sure data are now of the same length, and the concatenate them together 
        #TODO work on the case when one of the data is missing
        sample_lens = set([len(d) for d in self.data_list])
        assert len(sample_lens) == 1, f'Error: data not of the same length {sample_lens}'      
        data_concat = np.hstack(self.data_list)
        print(f'data_concat: {data_concat.shape}')
        
        # clear the current data
        for k in self.message_type_dict.keys():
            self.message_type_dict[k] = []

        return Message('sync_data', data_concat)
        
        # if data_concat is not None: 
        #     if self.adc_buffer is None:
        #         self.adc_buffer = RingBuffer((self.buffer_length,data_concat.shape[1]))
        #         self.adc_buffer.write(data_concat) #format should be in time x channel
        #     else:
        #         try:
        #             self.adc_buffer.write(data_concat)
        #         except ValueError:
        #             self.log(logging.DEBUG, 'Data shape has changed. Initializing a new buffer')
        #             self.reset_buffer(adc_channels=data_concat.shape[1])
        #             self.adc_buffer.write(data_concat)
                    
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
        sine1 = SineTimeSource(0.25, 10, 2, msg_type='message1', sampling_frequency=50)
        sine2 = SineTimeSource(0.5, 5, 3, msg_type='message2', sampling_frequency=100)
        merger = AnalogSignalMerger( [Message1, Message2], interval=1)
        
        gui = GUIProcessor()
        visualizer = AnalogVisualizer('Sync data')
        gui.register_visualizer(visualizer,['sync_data'])
        
        sink = EchoSink()
        
        sine1.connect(merger)
        sine2.connect(merger)
        merger.connect(sink)
        merger.connect(gui)
        
