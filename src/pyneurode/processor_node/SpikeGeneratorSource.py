from typing import List
from pyneurode.processor_node.Processor import Source
from dataclasses import dataclass

@dataclass
class SimulatedNeuron:
    """
    A class for holding a simulated neurons
    """
    firing_rate: int
    electrode_id: int
    firing_time: float


def SpikeGeneratorSource(Source):
    """This source generate spikes for testing. The spike train follows a poison distribution
    """
    
    def __init__(self, spike_templates, neurons:List[List]):
        self.super().__init__()
        self.num_electrode = len(neurons)
        self.neurons = neurons
        self.cells_per_electrode = [len(n) for n in neurons] #the number of cells in each electrode
        
        
    
        
        