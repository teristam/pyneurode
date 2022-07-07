# from processor_node.BatchProcessor import BatchProcessor
import logging
import time
from collections import Counter

import numpy as np
from pyneurode.processor_node.ADCFileReaderSource import ADCFileReaderSource
from pyneurode.processor_node.BatchProcessor import BatchProcessor
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.TemplateMatchProcessor import SpikeTrainMessage
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from pyneurode.RingBuffer.RingBufferG import RingBufferG


class SyncDataProcessor(BatchProcessor):
    '''
    Synchronize two sources of time series data
    '''
    
    def __init__(self, interval=0.001, internal_buffer_size=10000, ignore_adc=False, verbose=True):
        super().__init__(interval, internal_buffer_size, verbose)
        self.ignore_adc = ignore_adc # whether to ignore ADC signal, mainly used for debugging
    
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
        adc_data_list = []
        spike_data_list =[]
        spike_train = None
        adc_data = None

        # prcoess each messages
        for d in data:
            if d.type == 'adc_data':
                adc_data_list.append(d.data)
            elif isinstance(d,SpikeTrainMessage):
                spike_data_list.append(d.data)
                
        # concatenate the mssage
        if len(adc_data_list)>0:
            try:
                adc_data = np.concatenate(adc_data_list)
            except ValueError:
                self.log(logging.DEBUG, 'ADC array mistmatch, skipping')
        
        if adc_data is not None: 
            if self.adc_buffer is None:
                self.adc_buffer = RingBuffer((self.buffer_length,adc_data.shape[1]))
                self.adc_buffer.write(adc_data) #format should be in time x channel
            else:
                try:
                    self.adc_buffer.write(adc_data)
                except ValueError:
                    self.log(logging.DEBUG, 'Data shape has changed. Initializing a new buffer')
                    self.reset_buffer(adc_channels=adc_data.shape[1])
                    self.adc_buffer.write(adc_data)
            
        if len(spike_data_list)>0:
            # self.log(logging.DEBUG, f'Concatenating {len(spike_data_list)} {spike_data_list[0].shape}')
            # check if the spike data match
            try:
                spike_train = np.concatenate(spike_data_list,axis=0)
                # self.log(logging.DEBUG, spike_train.shape)
            except ValueError:
                self.log(logging.DEBUG, 'Array mismatch, skipping')
                
 
        # Write to internal buffer
        if spike_train is not None:
            # self.log(logging.DEBUG, f'spike_train shape: {spike_train.shape}')
            if self.spike_buffer is None:
                 # create the a new buffer
                self.spike_buffer = RingBuffer((self.buffer_length, spike_train.shape[1]))

            try:
                self.spike_buffer.write(spike_train)
            except ValueError:
                # spike train shape has changed, properly the data is resorted, discard the current content
                self.log(logging.DEBUG, f'Data shape has changed. Initializing a new buffer : {spike_train.shape} old buffer shape: {self.spike_buffer.buffer.shape}')
                self.reset_buffer(spike_train_channels=spike_train.shape[1])
                self.spike_buffer.write(spike_train) # time x channel
        
        msg_data_types = [d.type for d in data]
        # self.log(logging.DEBUG, f'Processing data: {Counter(msg_data_types)}')

        if self.spike_buffer is not None:

            if self.ignore_adc:
                curReadHead = self.spike_buffer.absWriteHead
                data2read = curReadHead - self.readHead
                
                if data2read > 0:
                    msg_data = self.spike_buffer.readContinous(data2read)
                    self.readHead += data2read
                    return Message('synced_data', msg_data)
  
            else:
                # Find the minimum of the write head, and then send them
                curReadHead = min(self.spike_buffer.absWriteHead, self.adc_buffer.absWriteHead)
                data2read = curReadHead - self.readHead 

                if data2read >0:

                    spike_train_data = self.spike_buffer.readContinous(data2read)
                    adc_data = self.adc_buffer.readContinous(data2read)

                    msg_data = np.hstack([spike_train_data, adc_data])

                    self.readHead += data2read

                    return Message('synced_data', msg_data)


if __name__ == '__main__':
    start  = time.time()
    with ProcessorContext() as ctx:
        # simulate change of message data shape
        adc_data = Message('adc_data', np.random.rand(5,2)) 
        adc_data2 = Message('adc_data', np.random.rand(5,4)) 
        spike_data = SpikeTrainMessage(np.random.rand(5,2)) # time x channel
        spike_data2 = SpikeTrainMessage(np.random.rand(5,3))
        
        msg_list = [adc_data if i%2==0 else spike_data for i in range(30)]
        msg_list += [adc_data if i%2==0 else spike_data2 for i in range(30)]
        msg_list += [adc_data2 if i%2==0 else spike_data2 for i in range(30)]

        msgtime = MsgTimeSource(0.2, msg_list)
        syncDataProcessor = SyncDataProcessor(interval=0.5)
        sink = EchoSink()
        
        msgtime.connect(syncDataProcessor)
        syncDataProcessor.connect(sink)

        ctx.register_processors(msgtime, syncDataProcessor, sink)

        ctx.start()
