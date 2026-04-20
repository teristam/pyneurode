from abc import ABC
import logging
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.AnalogVisualizer import AnalogVisualizer
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message
import numpy as np
import shortuuid
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from typing import Any, List, Optional, Union
from pyneurode.processor_node.NumpyFileSource import NumpyFileSource, NumpyMessage
from pyneurode.processor_node.Processor import Sink
from pyneurode.processor_node.SineTimeSource import SineTimeSource
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.SimGridCellSource import SimGridCellSource, SpacialDataMessage
from pyneurode.processor_node.Visualizer import Visualizer
from pyneurode.processor_node.MountainsortTemplateProcessor import RecomputeTemplateControlMessage
from scipy.ndimage import gaussian_filter



class TuningCurve2DVisualizer(Visualizer):
    '''
    Display a 2D analog signal
    It accepts data 
    '''
    def __init__(self, buffer_length:int = 500, filters=None, scale = 6, xbin:int = 10, ybin:int=10, xmax=10, ymax=10, time_scale=None) -> None:
        super().__init__(filters=filters)
        self.plot_data = (np.arange(buffer_length), np.zeros((buffer_length,)))
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
        self.smooth = True
        self.sel_cell_idx = 0
        self.color_map_scale = None
        self.cur_position = None
        self.log_display = True

    def init_gui(self):
        window_width = 800
        
        with dpg.window(label=self.name, width=window_width, height=500, tag=self.name):
            
            with dpg.theme(tag="plot_theme"):
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (0, 255, 0), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 8, category=dpg.mvThemeCat_Plots)

                        
            with dpg.group(horizontal=True):
                with dpg.child_window(width = -200):
                    values = np.zeros((self.xbin, self.ybin))
                    
                    with dpg.group(horizontal=True):
                        self.color_map_scale = dpg.add_colormap_scale(min_scale=0, max_scale=self.scale, height=400, colormap=dpg.mvPlotColormap_Jet)
                        with dpg.plot(label="Heat Series", no_mouse_pos=True, height=-1, width=-1) as plot:

                            dpg.add_plot_axis(dpg.mvXAxis, label="x", lock_min=True, lock_max=True, no_gridlines=True, no_tick_marks=True)
                            with dpg.plot_axis(dpg.mvYAxis, label="y", no_gridlines=True, no_tick_marks=True, lock_min=True, lock_max=True):
                                self.heatseries = dpg.add_heat_series(values, self.xbin, self.ybin, scale_min=0, scale_max=self.scale, format='')
                                dpg.bind_colormap(plot, dpg.mvPlotColormap_Jet)
                                
                                self.cur_position = dpg.add_scatter_series([], [])
                                dpg.bind_item_theme( self.cur_position, 'plot_theme')

            
                with dpg.child_window(width = 200) as control_panel:
                    with dpg.group():
                        def scale_slider_update(sender, value):
                            self.scale = value
                            dpg.configure_item(self.color_map_scale, max_scale=self.scale)
                            dpg.configure_item(self.heatseries, scale_max=self.scale)
                        
                        dpg.add_text('Colormap scale')
                        dpg.add_slider_float(label='Scale slider', width=window_width/4,
                                                    max_value=100, default_value=20, callback=scale_slider_update)

                        def listbox_callback(sender, selected_val):
                            self.sel_cell_idx = int(selected_val) # need to shift to account for the x and y coordinates
                            
                        dpg.add_text('Cell selection')
                        self.list_box=dpg.add_listbox(label='Cells',items=[], width=-1, num_items=10,
                                callback=listbox_callback)
                        
                        
                        dpg.add_text('Display mode')
                        def log_checkbox_callback(sender, log_distribution):
                            self.log_display = log_distribution
                            if log_distribution:
                                dpg.configure_item(self.color_map_scale, min_scale=-10)
                            else:
                                dpg.configure_item(self.color_map_scale, min_scale=0)
                            
                        dpg.add_checkbox(label='log distribution', default_value=True, callback=log_checkbox_callback)
                        
                        def smooth_toggle(sender, smooth_display):
                            self.smooth = smooth_display
                            
                        dpg.add_checkbox(label='Smooth', default_value = True, callback=smooth_toggle)
                        

                    self.control_panel = control_panel

    def update(self, messages: List[Message]):
        # message data format time x channel
        # it assume the data is a numpy array, with the first 2 channel corresponding
        # to x and y coordinate respectively. The other channel are neural data
        
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
        data = self.buffer.readLatest(50) #TODO: should use all data point instead
        
        # bin the x and y position
        # if data contains NaN, casting to int directly will lead to error
        x = np.clip(np.floor(data[:,0]/self.xbinsize), 0, self.xbin-1)
        y = np.clip(np.floor(data[:,1]/self.ybinsize), 0, self.ybin-1)
        z_idx = list(range(data.shape[1]-2))
        z =  data[:, 2:]
        
        #update the list box
        dpg.configure_item(self.list_box, items=z_idx, num_items=10)
        
        # plot the current position in the scatter plot
        dpg.set_value(self.cur_position, [[y[-1]/self.ybin], [1-x[-1]/self.xbin]]) # the heatmap axis goes from 0 to 1, and have different orientation

        if self.tuning_curve is None:
            self.tuning_curve = np.zeros((len(z_idx), self.ybin, self.xbin))
            self.bin_n = np.zeros((self.ybin, self.xbin))
        

        for i in range(data.shape[0]):
            # Average is always sum(x)/x_n
            if not np.isnan(x[i]) and not np.isnan(y[i]):
                x_pos, y_pos = int(x[i]), int(y[i])
                self.tuning_curve[:, x_pos, y_pos] = (self.bin_n[x_pos, y_pos]*self.tuning_curve[:, x_pos, y_pos] + z[i,:])/(self.bin_n[x_pos, y_pos]+1)
                self.bin_n[x_pos, y_pos] += 1
            
        
        if not self.tuning_curve is None and type(self.sel_cell_idx) is int:
            curve2display = self.tuning_curve[self.sel_cell_idx,:,:]
            if self.log_display:
                curve2display = np.log(curve2display+1e-10) # avoid inf
            if self.smooth:
                smoothed_curve = gaussian_filter(curve2display, sigma=1)
                dpg.set_value(self.heatseries, (smoothed_curve,))
            else:
                dpg.set_value(self.heatseries, (curve2display,))
            
        


class DummpyControlSink(Sink):
    def process(self, message: Message):
        if type(message) is RecomputeTemplateControlMessage:
            self.log(logging.DEBUG, 'Recompute message received')  
        else:
            self.log(logging.DEBUG, f'Received message {type(message)}')
                      
           
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with ProcessorContext() as ctx:

        # grid_cell = SimGridCellSource(0.02, arena_size = 100, speed=2, firing_field_sd=[10,20,40], grid_spacing=20)
        grid_cell = NumpyFileSource('data/openfield_cells.npy', interval=0.01,batch_size=100)
        
        tuningCurve2DVisualizer = TuningCurve2DVisualizer('Synchronized signals', scale=10, buffer_length=6000, 
                                                         xbin=100, ybin=100, xmax=60, ymax=60)
        
        analogVisualizer = AnalogVisualizer('Analog signal')
        
        gui = GUIProcessor(internal_buffer_size=5000)
        
        grid_cell.connect(gui)
        
        gui.register_visualizer(tuningCurve2DVisualizer, filters=[NumpyMessage.dtype])
        gui.register_visualizer(analogVisualizer, filters=[NumpyMessage.dtype])

        
