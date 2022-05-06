from abc import ABC
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.Processor import Message
import numpy as np
import shortuuid
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from typing import List, Union
from .Visualizer import Visualizer

class AnalogVisualizer(Visualizer):
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
        self.first_run = True
        self.scale = scale
        self.time_scale = time_scale # time period of each data point
        self.time_cursor = 'time_cursor'
        self.data_idx = None

    def init_gui(self):
        with dpg.window(label=self.name, width=800, height=500, tag=self.name):
            with dpg.plot(label='Plot', height=-1, width=-1) as plot_id:
                self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="")
                self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="")
                dpg.set_axis_limits_auto(self.x_axis)
                dpg.set_axis_limits_auto(self.y_axis)

    def update(self, messages: List[Message]):
        # message data format time x channel
        if self.buffer is None:
            self.buffer = RingBuffer((self.buffer_length, messages[0].data.shape[1]))
        
        for m in messages:
            self.buffer.write(m.data)

        ####
        # update the plot

        # Re-arrange the data to be suitable for plotting
        ydata = self.buffer.buffer.T.copy()
        shift = np.arange(ydata.shape[0])[None,].T * self.scale
        ydata += shift #shift the data for different channel

        ydata = ydata.tolist()
 
        xdata = np.arange(len(ydata[0]))
        if self.time_scale:
            xdata = xdata *self.time_scale

        xdata = xdata.tolist()

        series_name = [f'{i}_{self.name}' for i in range(len(ydata))]

        if self.first_run:
            self.first_run = False
            # print(series_name)
            for i in range(len(ydata)):
                dpg.add_line_series(xdata, ydata[i], label=series_name[i], parent=self.y_axis, tag=series_name[i])
                dpg.fit_axis_data(self.y_axis)
                dpg.fit_axis_data(self.x_axis)
            
            # dpg.add_vline_series([1.0], parent = self.y_axis, tag=self.time_cursor)
        else:
            for i in range(len(ydata)):
                dpg.set_value(series_name[i],(xdata,ydata[i])) #must be called in main thread
            
            



        # print(message)
        # self.plot_data[1][self.data_count%500]=message.data[0]
        # self.data_count += 1

        # dpg.set_value(self.plot_data_tag, self.plot_data)
