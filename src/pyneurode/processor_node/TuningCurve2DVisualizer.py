from abc import ABC
import logging
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.AnalogVisualizer import AnalogVisualizer
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message
import numpy as np
import shortuuid
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from typing import List, Union
from pyneurode.processor_node.Processor import SineTimeSource, Sink
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.SimGridCellSource import SimGridCellSource
from pyneurode.processor_node.Visualizer import Visualizer
from pyneurode.processor_node.MountainsortTemplateProcessor import RecomputeTemplateControlMessage
from scipy.ndimage import gaussian_filter

class TuningCurve2DVisualizer(Visualizer):
    '''
    Display a 2D analog signal
    '''
    def __init__(self, name:str, xy_idx, buffer_length:int = 500, scale = 6, xbin:int = 10, ybin:int=10, xmax=10, ymax=10, time_scale=None) -> None:
        super().__init__(name)
        self.plot_data = (np.arange(buffer_length), np.zeros((buffer_length,)))
        self.name = name #unique identying string for the visualizer
        self.data_count = 0
        self.plot_data_tag = self.name+'_plot_data'
        self.buffer = None
        self.buffer_length = buffer_length
        self.first_run = True
        self.scale = scale
        self.time_scale = time_scale # time period of each data point
        self.time_cursor = 'time_cursor'
        self.data_idx = None
        self.series_name = None
        self.control_panel = None
        self.xbin = xbin
        self.ybin = ybin
        self.xmax = xmax
        self.ymax = ymax
        self.xbinsize = self.xmax/xbin
        self.ybinsize = self.ymax/ybin
        self.tuning_curve = None
        self.xy_idx = xy_idx #index of the input data corresponding to the x and y coordinate of the 2D tuning curve, all others are treated as values
        self.smooth = True

    def init_gui(self):
        window_width = 800
        with dpg.window(label=self.name, width=window_width, height=500, tag=self.name):
            with dpg.group(horizontal=True):
                with dpg.child_window(width = -window_width*1.5/4):

                    # with dpg.drawlist(width=500, height=500) as self.canvas:
                    #         with dpg.draw_node(label = 'animal_position', user_data=0.0):
                    #             dpg.draw_circle([ 0, 0 ], 10, color=[0, 255,0], fill=[0,255,0])
                                
                                
                    values = np.zeros((self.xbin, self.ybin))
                    
                    with dpg.group(horizontal=True):
                        dpg.add_colormap_scale(min_scale=0, max_scale=self.scale, height=400, colormap=dpg.mvPlotColormap_Hot)
                        with dpg.plot(label="Heat Series", no_mouse_pos=True, height=-1, width=-1):
                            dpg.add_plot_axis(dpg.mvXAxis, label="x", lock_min=True, lock_max=True, no_gridlines=True, no_tick_marks=True)
                            with dpg.plot_axis(dpg.mvYAxis, label="y", no_gridlines=True, no_tick_marks=True, lock_min=True, lock_max=True):
                                self.heatseries = dpg.add_heat_series(values, self.xbin, self.ybin, scale_min=0, scale_max=self.scale, format='')
                                
      

            
                with dpg.child_window(width = window_width*1.5/4) as control_panel:
                    def scale_slider_update(sender, app_data,user_data):
                        self.scale = app_data
                        # dpg.fit_axis_data(self.y_axis)
                        
                    dpg.add_slider_float(label='Scale slider', width=window_width/4,
                                                max_value=100, default_value=20, callback=scale_slider_update)
                    

                    self.control_panel = control_panel

    def update(self, messages: List[Message]):
        # message data format time x channel
        if self.buffer is None:
            self.buffer = RingBuffer((self.buffer_length, messages[0].data.shape[1]))
        
        for m in messages:
            if m.data.shape[0] > self.buffer.length:
                #if the input data is too large, resize the data buffer
                self.buffer = RingBuffer((m.data.shape[0]*2, m.data.shape[1]))
            else:
                try:
                    self.buffer.write(m.data)
                except ValueError:
                    # data shape have changed, need to refresh all the plots
                    self.buffer = RingBuffer((self.buffer_length, m.data.shape[1]))
                    self.buffer.write(m.data)
                    self.first_run = True
                    if self.series_name is not None:
                        for s in self.series_name:
                            dpg.delete_item(s)
                    

        ####
        # update the plot

        # Re-arrange the data to be suitable for plotting
        data = self.buffer.readLatest(50)
        
        # bin the x and y position
        x = np.clip(np.floor(data[:,self.xy_idx[0]]/self.xbinsize), 0, self.xbin-1).astype(int)
        y = np.clip(np.floor(data[:,self.xy_idx[1]]/self.ybinsize), 0, self.ybin-1).astype(int)
        
        z_idx = np.setdiff1d(np.arange(data.shape[1]), self.xy_idx)
        z_values =  data[:, z_idx]
        sel_idx = 0
        z = z_values[:,0]
    
        
        if self.tuning_curve is None:
            self.tuning_curve = np.zeros((self.ybin, self.xbin))
            self.bin_n = np.zeros((self.ybin, self.xbin))
        
        for i in range(data.shape[0]):
            # Average is always sum(x)/x_n
            self.tuning_curve[x[i], y[i]] = (self.bin_n[x[i], y[i]]*self.tuning_curve[x[i], y[i]] + z[i])/(self.bin_n[x[i], y[i]]+1)
            self.bin_n[x[i], y[i]] += 1
            
        
        if not self.tuning_curve is None:
            if self.smooth:
                smoothed_curve = gaussian_filter(self.tuning_curve, sigma=1)
                dpg.set_value(self.heatseries, (smoothed_curve,))
                
            else:
                dpg.set_value(self.heatseries, (self.tuning_curve,))
            
        


class DummpyControlSink(Sink):
    def process(self, message: Message):
        if type(message) is RecomputeTemplateControlMessage:
            self.log(logging.DEBUG, 'Recompute message received')  
        else:
            self.log(logging.DEBUG, f'Received message {type(message)}')
                      
           
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with ProcessorContext() as ctx:

        grid_cell = SimGridCellSource(0.02, arena_size = 100, speed=2, firing_field_sd=5, grid_spacing=20)
        
        tuningCurve2DVisualizer = TuningCurve2DVisualizer('Synchronized signals',xy_idx = [0,1], scale=10, buffer_length=6000, 
                                                         xbin=100, ybin=100, xmax=100, ymax=100)
        
        analogVisualizer = AnalogVisualizer('Analog signal')
        control = DummpyControlSink()
        
        gui = GUIProcessor(internal_buffer_size=5000)
        
        grid_cell.connect(gui)
        
        gui.register_visualizer(tuningCurve2DVisualizer, filters=['grid_cell'], control_targets=[control])
        gui.register_visualizer(analogVisualizer, filters=['grid_cell'])

        
