        
import time
from typing import List, Tuple
from pyneurode.processor_node.Message import Message, NumpyMessage
from pyneurode.processor_node.Processor import Message2, TimeSource
import numpy as np

class SineTimeSource(TimeSource):
    '''
    A TimeSource that generate multi-channel sine wave for debugging purpose
    '''
    def __init__(self, interval:float=10, frequency:float=10, channel_num:int=2, sampling_frequency:int=100,
                 noise_std:float=0, scale:float=1, offset:float=0, msg_type:float =None):
        super().__init__(interval)
        self.frequency = frequency #in Hz
        self.start_time = time.time()
        self.channel_num = int(channel_num)
        self.last_time = None #last time the data are plotted
        self.dt = 1/sampling_frequency
        self.noise_std = noise_std
        self.scale = scale
        self.offset = offset
        self.msg_type = msg_type

    def process(self):
        # Generate a sine waveform
        now = time.time() - self.start_time

        if self.last_time is None:
            t = np.arange(0, now, step=self.dt)
        else:
            # t = np.linspace(self.last_time, now, int((now-self.last_time)*self.Fs))
            t = np.arange(self.last_time, now, step=self.dt)

        if len(t)> 0:
            # print(t)
            self.last_time = t[-1]+self.dt
            y = np.sin(2*np.pi*self.frequency*t)*self.scale+self.offset
            y = np.tile(y, (self.channel_num,1)).T
            
            if self.noise_std > 0:
                # add in noise
                y += np.random.normal(scale=self.noise_std, size=y.shape)
                
            if self.msg_type is None:
                return NumpyMessage(y)
            elif self.msg_type =='message1':
                return Message2(y)
            else:
                return Message2(y)
            
            
    def get_IOspecs(self) -> Tuple[List[Message], List[Message]]:
        return [[], [NumpyMessage]]