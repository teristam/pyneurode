import logging
from typing import List
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from pyneurode.processor_node.AnalogVisualizer import DummpyControlSink
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.Processor import SineTimeSource
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.Visualizer import Visualizer
import dearpygui.dearpygui as dpg
import numpy as np 

class FiringRateTriggerControl(Visualizer):
    
    '''
    Display a analog value
    '''
    def __init__(self, name:str,  buffer_length:int = 500, scale = 1, time_scale=None) -> None:
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
        
    def init_gui(self):
        window_width = 800
        with dpg.window(label=self.name, width=window_width, height=500, tag=self.name, autosize=True):
            with dpg.group(horizontal=True):
                self.plot_panel = dpg.add_child_window(width=window_width*3/4)
                    
                self.control_panel = dpg.add_child_window(autosize_x=True)
                    
    def update_cell_select_list(self):
        # update the checkboxes for selecting which cell to trigger
        no_cell = self.buffer.buffer.shape[1]
        
        # clear all the existing cells
        for gp in self.control_list:
            dpg.delete_item(gp) 
            
        self.control_list = []
        
        def selected_cell_changed(sender, is_selected, cell_no):
            if is_selected:
                self.selected_cells.append(cell_no)
            else:
                self.selected_cells.remove(cell_no)
                
            self.log(logging.DEBUG, f'Selected cells: {self.selected_cells}')            
            self.update_subplots() # also update the supblot associated with the selected cell

            
            
        for i in range(no_cell):
            with dpg.group(horizontal=True,parent=self.control_panel) as gp:
                dpg.add_checkbox(label = f'Cell {i}', callback=selected_cell_changed, user_data=i)
                dpg.add_slider_float()
                self.control_list.append(gp)
                
                
    def update_subplots(self):
        
        no_cell = len(self.selected_cells)

        if self.subplots is not None:
            dpg.delete_item(self.subplots)
        self.y_axes.clear()
        self.x_axes.clear()
    
        # Re-create the subplots and line series    
        self.series_name = [f'{i}_{self.name}' for i in self.selected_cells]
        
        self.log(logging.DEBUG, self.series_name)
        
        #Create subplot for each of the selected cells
        with dpg.subplots(no_cell, 1, parent=self.plot_panel, width=-1,height=-1) as subplots:
            self.subplots = subplots
            for i in range(no_cell):
                with dpg.plot():
                    xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="")
                    yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="")
                    self.y_axes.append(yaxis)
                    self.x_axes.append(xaxis)
                    
                    dpg.add_line_series([],[], label=self.series_name[i], parent=yaxis, tag=self.series_name[i])
                    
        self.need_fit_axis = True #refit the axis to the data


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
                    if len(self.control_list) == 0:
                        self.update_cell_select_list()
                except ValueError:
                    # data shape have changed, need to refresh all the plots
                    self.buffer = RingBuffer((self.buffer_length, m.data.shape[1]))
                    self.buffer.write(m.data)
                    self.first_run = True
                    if self.series_name is not None:
                        for s in self.series_name:
                            dpg.delete_item(s)
                    
                    # update the cell selection list
                    self.update_cell_select_list()

        ####
        # update the plot

        # Re-arrange the data to be suitable for plotting
        ydata = self.buffer.buffer.T.copy()
        # shift = np.arange(ydata.shape[0])[None,].T * self.scale
        # ydata += shift #shift the data for different channel

        ydata = ydata.tolist()
 
        xdata = np.arange(len(ydata[0]))
        if self.time_scale:
            xdata = xdata *self.time_scale

        xdata = xdata.tolist()

        for i in range(len(self.selected_cells)):
            if i<len(self.series_name):
                #sometimes there is a slight delay when new series_names are added
                #TODO: check if the series name depends on the order of the toggle of the checkbox
                dpg.set_value(self.series_name[i],(xdata,ydata[self.selected_cells[i]])) #must be called in main thread
            
        if self.need_fit_axis:
            self.need_fit_axis = False
            for i in range(len(self.selected_cells)):
                dpg.fit_axis_data(self.y_axes[i])   
                dpg.fit_axis_data(self.x_axes[i])    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with ProcessorContext() as ctx:
        sineWave = SineTimeSource(0.1,frequency = 12.3, channel_num=3, sampling_frequency=100)
        analog_visualizer = FiringRateTriggerControl('Synchronized signals')
        control = DummpyControlSink()
        
        gui = GUIProcessor(internal_buffer_size=5000)
        
        sineWave.connect(gui)
        gui.register_visualizer(analog_visualizer,filters=['sine'], control_targets=[control])
        
        ctx.start()