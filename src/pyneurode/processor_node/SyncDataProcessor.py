# from processor_node.BatchProcessor import BatchProcessor
from pyneurode.RingBuffer.RingBufferG import RingBufferG
from pyneurode.RingBuffer.RingBuffer import RingBuffer
import logging
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.SpikeSortProcessor import SpikeSortProcessor
from pyneurode.processor_node.BatchProcessor import BatchProcessor
from pyneurode.processor_node.ADCFileReaderSource import ADCFileReaderSource
import numpy as np
import time
from collections import Counter

class SyncDataProcessor(BatchProcessor):
    '''
    Synchronize two sources of time series data
    '''
    def startup(self):
        super().startup()
        self.buffer_length = 100*60*5
        self.agg_buffer = None
        self.adc_buffer = RingBuffer((self.buffer_length,1))
        self.spike_buffer = None
        self.readHead = 0

    def process(self,data):
        # read from two sources of data
        adc_data_list = []
        spike_data_list =[]
        spike_train = None

        # prcoess each messages
        for d in data:
            if d.type == 'adc_data':
                adc_data_list.append(d.data)
            elif d.type == 'spike_train':
                spike_data_list.append(d.data)

        # concatenate the mssage
        if len(adc_data_list)>0:
            adc_data = np.concatenate(adc_data_list)
            # self.log(logging.DEBUG, f'{adc_data.shape}')
            self.adc_buffer.write(adc_data) #format should be in time x channel
        
        if len(spike_data_list)>0:
            # self.log(logging.DEBUG, f'Concatenating {len(spike_data_list)} {spike_data_list[0].shape}')
            spike_train = np.concatenate(spike_data_list,axis=1)

        # Write to internal buffer
        if spike_train is not None:
            if self.spike_buffer is None:
            # create the a new buffer
                self.spike_buffer = RingBuffer((self.buffer_length, spike_train.shape[0]))
                self.spike_buffer.write(spike_train.T) # ringbuffer accept data in time x col format
            else:
                self.spike_buffer.write(spike_train.T)
        
        msg_data_types = [d.type for d in data]
        # self.log(logging.DEBUG, f'Processing data: {Counter(msg_data_types)}')

        if self.spike_buffer is not None:
            # self.log(logging.DEBUG, f'spike train write head: {self.spike_buffer.absWriteHead}')
            # self.log(logging.DEBUG, f'adc train write head: {self.adc_buffer.absWriteHead}')

            # Find the minimum of the write head, and then send them
            curReadHead = min(self.spike_buffer.absWriteHead, self.adc_buffer.absWriteHead)
            data2read = curReadHead - self.readHead 

            if data2read >0:

                spike_train_data = self.spike_buffer.readContinous(data2read)
                adc_data = self.adc_buffer.readContinous(data2read)

                # self.log(logging.DEBUG, f'{spike_train_data.shape} {adc_data.shape}')
                msg_data = np.hstack([spike_train_data, adc_data])

                self.readHead += data2read

                return Message('synced_data', msg_data)


if __name__ == '__main__':
    start  = time.time()
    echofile = FileEchoSink.get_temp_filename()
    # echofile = 'test/data/syncfile.pkl'

    with ProcessorContext() as ctx:

        fileReader = FileReaderSource('data/data_packets_M2_D23_1910.pkl')
        spikeSortProcessor = SpikeSortProcessor(interval=0.02)
        syncDataProcessor = SyncDataProcessor(interval=0.02)
        fileEchoSink = FileEchoSink(echofile)

        fileReader.connect(spikeSortProcessor, filters='spike')
        spikeSortProcessor.connect(syncDataProcessor)
        fileReader.connect(syncDataProcessor, 'adc_data')
        syncDataProcessor.connect(fileEchoSink)

        ctx.register_processors(fileReader, spikeSortProcessor, syncDataProcessor, fileEchoSink)

        ctx.start()

        time.sleep(20)
        ctx.shutdown_event.set()

    time.sleep(1)  
    data = FileEchoSink.open_data_dump(echofile)
    print(data)