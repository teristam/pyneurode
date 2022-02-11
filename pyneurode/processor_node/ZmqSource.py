from .Processor import EchoProcessor, Source, Message
import time 
from pyneurode.zmq_client import *
import pickle 
from .ProcessorContext import ProcessorContext
from pyneurode.spike_sorter import simpleDownSample
import pyneurode.utils


class ZmqSource(Source):
    '''
    Read data from ZMQlink
    '''
    def __init__(self, adc_channel =20, Fs = 30000, time_bin = 0.1, data_dump_fn = None):
        super().__init__()
        self.data_dump_fn = data_dump_fn
        self.adc_channel = adc_channel
        self.data_leftover = []
        self.segment_size = int(Fs*time_bin)


    def startup(self):
        self.client = SpikeSortClient()
        if self.data_dump_fn is not None:
            self.data_dump = open(self.data_dump_fn,'wb')

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

    def process(self):
         # Data acquisition thread


        data_leftover = np.array([])
        global start_timestamp 
        while True:
            start = time.time()
            data_list = self.client.callback()

            msg_list = []

            for data in data_list:
               

                timestamp = None
                if 'data_timestamp' in data.keys():
                    timestamp = data['data_timestamp']

                #process spike data
                if 'spike' in data.keys():
                    # note: below python 3.10, perf_counter is not system-wide on Windows, thus
                    # the number returned by different process may not be aligned to each others
                    data['spike'].acq_timestamp = utils.perf_counter()
                    msg_list.append(Message('spike', data['spike'], timestamp))

                #process raw signal data
                if 'data' in data.keys():

                    data_ds = self.downsample_adc(data['data'])
                    if data_ds is not None:
                        msg_list.append(Message('adc_data', data_ds, timestamp)) 

            if self.data_dump_fn is not None:
                pickle.dump(msg_list,self.data_dump)

            return msg_list


if __name__ == '__main__':
    with ProcessorContext() as ctx:
        zmqSource = ZmqSource(adc_channel=68, data_dump_fn='data/64_channel_packetlist.pkl')
        echoProcessor = EchoProcessor()

        zmqSource.connect(echoProcessor)

        ctx.register_processors(zmqSource, echoProcessor)

        ctx.start()

