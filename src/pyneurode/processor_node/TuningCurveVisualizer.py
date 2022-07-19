import logging
import time
from typing import List

import numpy as np
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from pyneurode.processor_node.AnalogVisualizer import AnalogVisualizer, DummpyControlSink
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.Processor import MsgTimeSource, SineTimeSource
from pyneurode.processor_node.ProcessorContext import ProcessorContext
import dearpygui.dearpygui as dpg

class TuningCurveEstimator():
    def __init__(self, nbin, bin_size=1):
        self.tuning_curve = None
        self.bin_size = bin_size
        self.bin_n = np.zeros((nbin,))
        self.nbin = nbin
        
    def update(self, x, y):
        """_summary_

        Args:
            x (np.array): behavioral variable
            y (np.array): (time x channel)
        """
        x = x//self.bin_size
        
        # assert np.max(x) < self.nbin, 'The large data exceed the number of bin'
        assert len(x) == len(y), 'Length of two inputs must be the same'
        
        if self.tuning_curve is None:
            self.tuning_curve = np.zeros((y.shape[1], self.nbin))
            for i in range(self.nbin):
                ysub = y[i==x,:]
                if len(ysub)>0:
                    self.tuning_curve[:,i] = np.mean(ysub,axis=0)
                    self.bin_n[i] = np.sum(i==x)
        else:
            # update the current mean
            for i in range(self.nbin):
                m = np.sum(i==x)
                if m>0:
                    self.tuning_curve[:,i] = (self.bin_n[i]*self.tuning_curve[:,i] + m*np.mean(y[i==x,:],axis=0))/(m + self.bin_n[i])
                    self.bin_n[i] += m
    
    def get_tuning_curve(self):
        return (np.arange(self.nbin), self.tuning_curve)
    
    def get_num_cell(self):
        if self.tuning_curve is not None:
            return self.tuning_curve.shape[0]
        else:
            return 0


class TuningCurveVisualizer(AnalogVisualizer):
    """
    A visualizer that calculate and display the 1D tuning curve of cells in realtime
    """
    def __init__(self, name:str, x_variable_idx, nbin, bin_size = 1,
                 buffer_length:int = 500, scale = 1, time_scale=None, smoothing_alpha=0.1) -> None:
        super().__init__(name)
        self.plot_data = (np.arange(buffer_length), np.zeros((buffer_length,)))
        self.name = name #unique identying string for the visualizer
        self.data_count = 0
        self.plot_data_tag = self.name+'_plot_data'
        self.buffer = None
        self.buffer_length = buffer_length
        self.need_fit_axis = True
        self.scale = scale
        self.time_scale = time_scale # time period of each data point
        self.time_cursor = 'time_cursor'
        self.data_idx = None
        self.series_name = []
        self.control_list = []
        self.y_axes = []
        self.x_axes = []
        self.selected_cells = []
        self.subplots = None
        self.tuning_curve_estimator = TuningCurveEstimator(nbin = nbin, bin_size=bin_size)
        self.x_variable_idx = x_variable_idx
        
        
    def init_gui(self):
        window_width = 800
        with dpg.window(label=self.name, width=window_width, height=500, tag=self.name, autosize=True):
            with dpg.group(horizontal=True):
                self.plot_panel = dpg.add_child_window(width=-window_width*1/4) #negative means no. of pixel from the right of its container
                self.control_panel = dpg.add_child_window(width=window_width*1/4)
                dpg.add_spacer(height = 20, parent=self.control_panel)
                    
    def update_cell_select_list(self):
        # update the checkboxes for selecting which cell to trigger
        no_cell = self.tuning_curve_estimator.get_num_cell()
        self.selected_cells = [0 for _ in range(no_cell)]
        self.series_name = [f'{i}_{self.name}' for i in range(len(self.selected_cells))]
        self.y_axes = [None for i in range(len(self.selected_cells))]
        self.x_axes = [None for i in range(len(self.selected_cells))]
        
        # clear all the existing cells
        for gp in self.control_list:
            dpg.delete_item(gp) 
            
        self.control_list = []
        self.threshold_sliders = []
        self.trigger_direction = []
        
        def selected_cell_changed(sender, is_selected, cell_no):
            self.selected_cells[cell_no] = is_selected           
            self.update_subplots() # also update the supblot associated with the selected cell

        
        def slider_changed(sender, value, cell_no):
            #update the horizontal line of the threshold
            dpg.set_value(self.threshold_timeseries[cell_no], ([value],))
            
        for i in range(no_cell):
            with dpg.group(horizontal=True,parent=self.control_panel) as gp:
                dpg.add_checkbox(label = f'Channel {i}', callback=selected_cell_changed, user_data=i)
                self.control_list.append(gp)
                
                
    def update_subplots(self):
        
        if self.subplots is not None:
            dpg.delete_item(self.subplots)

    
        # Re-create the subplots and line series    
        
        self.log(logging.DEBUG, self.series_name)
        
        #Create subplot for each of the selected cells
        ncol = 3
        nrow = np.ceil(sum(self.selected_cells)/ncol)
        with dpg.subplots(nrow, ncol, parent=self.plot_panel, width=-1,height=-1) as subplots:
            self.subplots = subplots
            for i,is_selected in enumerate(self.selected_cells):
                if is_selected: # if the cell is selected
                    
                    with dpg.plot():
                        dpg.add_plot_legend()
                        xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="")
                        yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="")
                        self.y_axes[i] = yaxis
                        self.x_axes[i] = xaxis
                        
                        dpg.add_line_series([],[], label=self.series_name[i], parent=yaxis, tag=self.series_name[i])
                        
                        dpg.fit_axis_data(self.y_axes[i])   
                        dpg.fit_axis_data(self.x_axes[i])   
                    
                    
        self.need_fit_axis = True #refit the axis to the data

    def update(self, messages: List[Message]):
        
        for m in messages:
            self.tuning_curve_estimator.update(m.data[:, self.x_variable_idx ], m.data[:,:-1])
            
        
        if not len(self.control_list) == self.tuning_curve_estimator.get_num_cell():
            self.update_cell_select_list()
        
        
        # update the plot
        
        x, y = self.tuning_curve_estimator.get_tuning_curve()
        for i, is_selected in enumerate(self.selected_cells):
            if is_selected:
                dpg.set_value(self.series_name[i],(x, y[i,:])) #must be called in main thread
        
        
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # x = np.random.normal(5,scale=5,size=(200,1))
    x = np.random.randint(0,20,(200,1))
    y = np.zeros((len(x),3),dtype=float)
    for i in range(len(x)):
        y[i,:] = np.random.normal(x[i]%10, size=3)
        
    data = np.hstack([y,x])
    print(data.shape)
    
    msg = []
    for _ in range(200):
        idx = np.random.choice(np.arange(len(data)), size=100)
        msg += [Message('data', data[idx,:])]

    with ProcessorContext() as ctx:
        source = MsgTimeSource(0.1, msg)
        visualizer = TuningCurveVisualizer('Synchronized signals', x_variable_idx = -1, nbin=20, buffer_length=6000)
        control = DummpyControlSink()
        
        gui = GUIProcessor(internal_buffer_size=5000)
        
        source.connect(gui)
        gui.register_visualizer(visualizer,filters=['data'], control_targets=[control])
        
        ctx.start()