import random
import time
from dataclasses import dataclass
from os import times
from sqlite3 import Timestamp
from typing import List, Optional, Tuple

import numpy as np
from pyneurode.processor_node.Processor import EchoSink, TimeSource
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.zmq_client import OpenEphysSpikeEvent
from pyneurode.processor_node.Message import Message
import logging

@dataclass
class SimulatedNeuron:
    """
    A class for holding a simulated neurons
    """
    firing_rate: int
    waveform: np.ndarray # in (num_channel, num_samples)
    firing_time: Optional[Tuple[int,int]]=None # a tuple indicating the firing time in sample unit, None will be firing all the time
    
    def __repr__(self):
        return f'SimulatedNeuron (firing_rate: {self.firing_rate}, waveform: {self.waveform.shape}, firing_time: {self.firing_rate}'

class SpikeGeneratorSource(TimeSource):
    """This source generate spikes for testing. The spike train follows a poison distribution
    """

    def __init__(self,neurons:List[List[SimulatedNeuron]], time_step:int = 100, Fs:int=30000, noise_level=2):
        super().__init__(1/time_step)
        self.num_electrode = len(neurons)
        self.neurons = neurons
        self.cells_per_electrode = [len(n) for n in neurons] #the number of cells in each electrode
        self.time_step = time_step # how many steps in 1s, that determine the maximum rate a neuron can fire
        self.Fs = Fs
        self.timestamp = 0
        self.noise_level = noise_level
        self.log(logging.INFO, f'I am creating the following neurons: {self.neurons}')
    
        
    def process(self):
        # loop through the neurons, and see if a spike needs to generated
        event2send = []
        
        for i in range(self.num_electrode):
            for neuron in self.neurons[i]:
                if random.uniform(0,1) < neuron.firing_rate/self.time_step: #TODO: it can't simulate firing rate less than 1 now
                    
                    #check for early break
                    if neuron.firing_time is not None:
                        if self.timestamp<self.firing_time[0] or self.timestamp > self.firing_time[1]:
                            # outside firing range
                            continue
                    
                    # generate spike event
                    # waveform in (num_channel, num_samples)
                    waveform = neuron.waveform + np.random.randn(*neuron.waveform.shape)*self.noise_level
                    spk_evt = OpenEphysSpikeEvent({'n_channels': self.num_electrode, 'n_samples': neuron.waveform.shape[1],
                                                   'electrode_id': i, 'sorted_id': 0, 
                                                   'timestamp': self.timestamp, 'channel': 0, 
                                                   'threshold': 6, 'source': 115}, 
                                                  waveform)
                    
                    # print(neuron.waveform.shape)
                    
                    event2send.append(Message('spike', spk_evt))
        
        self.timestamp += int(self.time_step/1000*self.Fs)
        return event2send

def make_neurons(neuron_nums: List[int], firing_rate: List[List[float]], templates: np.ndarray):
    neurons = []
    cell_idx = 0
    
    for i in range(len(neuron_nums)):
        neurons_in_chan = []
        for j in range(neuron_nums[i]):
            n=SimulatedNeuron(firing_rate[i][j], waveform=templates[cell_idx,:,:])
            neurons_in_chan.append(n)
            cell_idx += 2

        neurons.append(neurons_in_chan)
        
    
    return neurons
        
if __name__ == '__main__':
    # load the simulated waveform
    templates = np.load('src/pyneurode/data/spike_templates.npy') # cells x electrode x timepoints

    neurons = make_neurons([3,2,1],[[5,10,5],[10,20],[5]], templates=templates)
    
    with ProcessorContext() as ctx:
        
        spikegen = SpikeGeneratorSource(neurons)
        echo = EchoSink()
        spikegen.connect(echo)
        
        ctx.register_processors(spikegen, echo)
        
        ctx.start()

    
        