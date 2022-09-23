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

class SimulatedNeuron:
    """
    A class for holding a simulated neurons
    """
    def __init__(self, firing_rate:float, waveform: np.ndarray, firing_time: Optional[Tuple[int,int]]=None) -> None:  
        self.firing_rate = firing_rate
        self.waveform = waveform # in (num_channel, number_waveform,  num_samples)
        self.firing_time = firing_time # a tuple indicating the firing time in sample unit, None will be firing all the time
        
    def get_waveform(self):
        # randomly return a stored waveform
        amp_mode = np.random.normal(1.01,0.01) # add a small amplitude modulation
        return self.waveform[:, np.random.randint(0, self.waveform.shape[1]),:]*amp_mode
        
        
    
    def __repr__(self):
        return f'SimulatedNeuron (firing_rate: {self.firing_rate}, waveform: {self.waveform.shape}, firing_time: {self.firing_rate}'

class SpikeGeneratorSource(TimeSource):
    """This source generate spikes for testing. The spike train follows a poison distribution
    """

    def __init__(self,neurons:List[List[SimulatedNeuron]], time_step:int = 100, Fs:int=30000, amp_mod_level:float=0.5):
        super().__init__(1/time_step)
        self.num_electrode = len(neurons)
        self.neurons = neurons
        self.cells_per_electrode = [len(n) for n in neurons] #the number of cells in each electrode
        self.time_step = time_step # how many steps in 1s, that determine the maximum rate a neuron can fire
        self.Fs = Fs
        self.timestamp = 0
        self.amp_mod_level = amp_mod_level
        self.log(logging.INFO, f'I am creating the following neurons: {self.neurons}')
    
    def process(self):
        # loop through the neurons, and see if a spike needs to generated
        event2send = []
        
        for i in range(self.num_electrode):
            for neuron in self.neurons[i]:
                if random.uniform(0,1) < neuron.firing_rate/self.time_step: #TODO: it can't simulate firing rate less than 1Hz now
                    
                    #check if it is within the firing time of the neuron
                    if neuron.firing_time is not None:
                        if (neuron.firing_time[0] is not None and self.timestamp<neuron.firing_time[0]) or (neuron.firing_time[1] is not None and self.timestamp > neuron.firing_time[1]):
                            # outside firing range
                            continue
                    
                    # generate spike event
                    # waveform in (num_channel, num_samples)
                    waveform = neuron.get_waveform()
                    assert waveform.ndim == 2
                    spk_evt = OpenEphysSpikeEvent({'n_channels': self.num_electrode, 'n_samples': neuron.waveform.shape[1],
                                                   'electrode_id': i, 'sorted_id': 0, 
                                                   'timestamp': self.timestamp, 'channel': 0, 
                                                   'threshold': 6, 'source': 115}, 
                                                  waveform)
                    
                    # print(neuron.waveform.shape)
                    
                    event2send.append(Message('spike', spk_evt))
        
        self.timestamp += int(self.Fs/self.time_step)
        return event2send

def make_neurons(neuron_nums: List[int], firing_rate: List[List[float]], firing_time:List[List[Tuple[int,int]]], templates: np.ndarray):
    '''
    
    templates: (cell, electrode, no of waveform, time)
    '''
    neurons = []
    cell_idx = 0
    
    for i in range(len(neuron_nums)):
        neurons_in_chan = []
        for j in range(neuron_nums[i]):
            n=SimulatedNeuron(firing_rate[i][j], waveform=templates[cell_idx], firing_time=firing_time[i][j])
            neurons_in_chan.append(n)
            cell_idx += 1

        neurons.append(neurons_in_chan)
        
    
    return neurons
        
if __name__ == '__main__':
    # load the simulated waveform
    templates = np.load('src/pyneurode/data/spikes_waveforms.npy') # cells x electrode x timepoints
    Fs=30000
    # neurons = make_neurons([3,2,1],[[5,10,5],[10,20],[5]], templates=templates)
    neurons = make_neurons([2],firing_rate=[[5,3]],firing_time=[[(None),(Fs*5, None)]], templates=templates)
    
    with ProcessorContext() as ctx:
        
        spikegen = SpikeGeneratorSource(neurons)
        echo = EchoSink()
        spikegen.connect(echo)
        
        ctx.register_processors(spikegen, echo)
        
        ctx.start()

    
        