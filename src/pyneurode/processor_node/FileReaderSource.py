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



class FileReaderSource(TimeSource):
    '''
    Read from data from a file. It will read data in a batch and send them out
    '''
    def __init__(self, filename:str, interval:float=0, batch_size:int=10, 
                random_batch:bool=False, filter:list=None, adc_channel:int=ADC_CHANNEL, Fs:int = 30000, time_bin:float = 0.1):



        super().__init__(interval)
        self.filename = filename
        self.random_batch = random_batch
        self.batch_size = batch_size
        self.adc_channel = adc_channel
        self.filter = filter #only send this kind of message
        self.start_timestamp = None
        self.data_leftover = []
        self.segment_size = int(Fs*time_bin)
        self.start_spike_timestamp = 0 #used to shift the spike timing when wrapped to the beginning of recording
        self.last_spike_timestamp = 0

    def startup(self):
        super().startup()
        self.file = open(self.filename,'rb')


    def downsample_adc(self, data):
        if self.adc_channel is not None:
            d = data[[self.adc_channel],:].T
        else:
            d = data.T

        # concatenate the leftover data from previous run
        # d should be in time x channel
        if (len(self.data_leftover) >0):
            d = np.vstack([self.data_leftover, d])
        
        (data_ds, self.data_leftover) = simpleDownSample(d, self.segment_size)

        return data_ds

    def process(self) -> Message :
        msg_list = []

        if self.random_batch:
            batch_size = random.random(1,self.batch_size)
        else:
            batch_size = self.batch_size

        try:
            for b in range(batch_size):
                try:
                    data = pickle.load(self.file)
                except EOFError:
                    # restart the file
                    self.log(logging.DEBUG, 'Reach the end of file. Restart from beginning')
                    self.file = open(self.filename,'rb')
                    data = pickle.load(self.file)

                    # update the timestamp shift so that the spiketrain calculation is correct later
                    self.start_spike_timestamp = self.last_spike_timestamp

                # self.log(logging.DEBUG, f'Loaded {data.keys()}')
                # data here is a dictionary, with data, spike and data_timestamp

                if self.filter is None:
                    keys = list(data.keys())
                    timestamp = None
                    if 'data_timestamp' in keys:
                        timestamp = data['data_timestamp']

                    # mark the first data
                    if self.start_timestamp is None:
                        self.start_timestamp = timestamp

                    if 'spike' in keys:
                        spike = data['spike']
                        spike.acq_timestamp = time.perf_counter()
                        spike.timestamp += self.start_spike_timestamp #shift the timestamp when going back to the beginning
                        self.last_spike_timestamp = spike.timestamp

                        msg_list.append(Message('spike', spike, timestamp))   

                    if 'data' in keys:
                        # need to do some special processing to downsample the data
                        data_ds = self.downsample_adc(data['data'])
                        if data_ds is not None:
                            msg_list.append(Message('adc_data', data_ds, timestamp))   

                else:
                    # only return a subset of the data
                    timestamp = None
                    if 'data_timestamp' in data.keys():
                        timestamp = data['data_timestamp']

                    for filter in self.filter:
                        if filter in data.keys() and data[filter] is not None:

                            if filter == 'data':
                                data_ds = self.downsample_adc(data['data'])
                                if data_ds is not None:
                                    msg = Message('adc_data', data_ds, timestamp)
                            else:
                                msg = Message(filter, data[filter], timestamp)

                            msg_list.append(msg)

            # self.log(logging.DEBUG, f'Sending {data}')
            return msg_list

        except EOFError:
            return msg_list



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    start  = time.time()
    echofile = FileEchoSink.get_temp_filename()

    with ProcessorContext() as ctx:

        fileReader = FileReaderSource('data/data_packets.pkl', interval=0.01)
        echo = FileEchoSink(echofile)
        fileReader.connect(echo,'adc_data')

        ctx.register_processors(fileReader, echo)

        # ctx.makeProcessor('Echo', FileEchoSink, echofile)
        # ctx.connect('FileReader','Echo','data')
        ctx.start()

        time.sleep(5)
        ctx.shutdown_event.set()

    time.sleep(1)  
    data = FileEchoSink.open_data_dump(echofile)
    print(len(data))
    assert len(data) > 5
