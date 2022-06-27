from multiprocessing import Event

import numpy as np
from pyneurode.spike_sorter import *

from pyneurode.processor_node.FileReaderSource import *
from pyneurode.processor_node.Message import Message

ADC_CHANNEL = 20

class ADCFileReaderSource(FileReaderSource):
    '''
    Read ADC data from file
    '''
    def __init__(self, proc_name: str, shutdown_event: Event, filename, 
                channel=ADC_CHANNEL, interval=0, batch_size=10, random_batch=False, Fs = 30000, time_bin = 0.1, filter=['data']):
        super().__init__(proc_name, shutdown_event, filename, interval=interval, batch_size=batch_size, random_batch=random_batch, filter=filter)
        self.start_timestamp = None
        self.channel = channel
        self.data_leftover = [] # data left over from previous downsampling
        self.segment_size = int(Fs*time_bin)

    def process(self):
        msg = super().process()
        data = msg.data # a list of numpy array

        data = np.concatenate(data,axis=1) #convert list of array to one array
        # self.log(logging.DEBUG, data.shape)

        
        if self.start_timestamp is None:
            self.start_timestamp = msg.timestamp

        if self.channel is not None:
            d = data[[self.channel],:].T
        else:
            d = data.T

        # concatenate the leftover data from previous run
        # d should be in time x channel
        if (len(self.data_leftover) >0):
            d = np.vstack([self.data_leftover, d])
        
        (data_ds, self.data_leftover) = simpleDownSample(d, self.segment_size)

        return Message('adc_data', data_ds, msg.timestamp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    with ProcessorContext() as ctx:
        ctx.makeProcessor('ADCFileReader', ADCFileReaderSource, 'data/data_packets_M2_D23_1910.pkl', 
            interval=0.1, channel = ADC_CHANNEL)
        ctx.makeProcessor('Echo', EchoProcessor)
        ctx.connect('ADCFileReader','Echo')
        ctx.start()
