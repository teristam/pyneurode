'''
Simulate the firing field of a grid cell for testing purpose
'''

from pyneurode.processor_node.AnalogVisualizer import AnalogVisualizer
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.Processor import Source, TimeSource
import numpy as np
from scipy.stats import multivariate_normal

from pyneurode.processor_node.ProcessorContext import ProcessorContext


def guassmf2d(x, mean, sigma):
    # Guassian membership function to estimate the probability
    
    return np.exp(-((x-mean)**2) / sigma**2)

class SimGridCellSource(TimeSource):
    def __init__(self, interval, arena_size = 100, speed =2, grid_spacing=10, base_fr = 10, firing_field_sd=1):
        super().__init__(interval)
        self.arena_size = arena_size
        self.x = arena_size/2 # always starts at the middle of the arena
        self.y = arena_size/2
        self.grid_spacing = grid_spacing
        self.speed = speed # mean of a normal distribution of random movement
        self.base_fr = base_fr
        self.firing_field_sd = firing_field_sd
        


    def process(self):
        # At each run, perform a random walk around, and generate the firing rate with a gaussian function centred around the multiple of the grid spacing
        self.x += np.random.normal()*self.speed
        self.y += np.random.normal()*self.speed
        
        self.x = np.clip(self.x, 0, self.arena_size) # prevent the animal going outsidet the arena
        self.y = np.clip(self.y, 0, self.arena_size) 
        
        xshift = self.x - round(self.x/self.grid_spacing)*self.grid_spacing #the different from the closest grid center
        yshift = self.y - round(self.y/self.grid_spacing)*self.grid_spacing
        
        # use a 2d gaussian to calculate the firing rate
        if type(self.firing_field_sd) is list:
            fr_list = []
            for i in range(len(self.firing_field_sd)):
                fr = multivariate_normal.pdf((xshift, yshift),  mean=(0,0), cov=self.firing_field_sd[i]) 
                norm_factor = multivariate_normal.pdf((0, 0),  mean=(0,0), cov=self.firing_field_sd[i]) 
                fr = fr/norm_factor * self.base_fr # make sure the center is always at the base firing rate
                
                fr_list.append(fr)
            
            data = np.array([[self.x, self.y, *fr_list]])
        else:
            fr = multivariate_normal.pdf((xshift, yshift),  mean=(0,0), cov=self.firing_field_sd[i]) 
            norm_factor = multivariate_normal.pdf((0, 0),  mean=(0,0), cov=self.firing_field_sd[i]) 
            fr = fr/norm_factor * self.base_fr # make sure the center is always at the base firing rate
            data = np.array([[self.x, self.y, fr]])
            
        msg = Message('grid_cell', data)
        return msg
    
    def shutdown(self):
        super().shutdown()

if __name__ == '__main__':

    with ProcessorContext() as ctx:
        # sineWave = SineTimeSource(0.01,frequency = 12.3, channel_num=3, sampling_frequency=100, scale=10, offset=10,
        #                           noise_std=0)
        
        grid_cell = SimGridCellSource(0.1)

        analogVisualizer = AnalogVisualizer('Analog signal')
        
        gui = GUIProcessor(internal_buffer_size=5000)
        
        grid_cell.connect(gui)
        
        gui.register_visualizer(analogVisualizer, filters=['grid_cell'])