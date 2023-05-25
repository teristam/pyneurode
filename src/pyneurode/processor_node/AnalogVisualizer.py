from abc import ABC
import logging
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message, NumpyMessage
import numpy as np
import shortuuid
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from typing import List, Tuple, Union
from pyneurode.processor_node.Processor import Sink
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.Visualizer import Visualizer
from pyneurode.processor_node.MountainsortTemplateProcessor import RecomputeTemplateControlMessage
from pyneurode.processor_node.SineTimeSource import SineTimeSource


class AnalogVisualizer(Visualizer):
    '''
    Display a analog value
    '''
    def __init__(self, buffer_length:int = 500, scale = 1, time_scale=None, filters=None) -> None:
        super().__init__(filters=[NumpyMessage])
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

    def init_gui(self):
        window_width = 800
        with dpg.window(label=self.name, width=window_width, height=500, tag=self.name):
            with dpg.group(horizontal=True):
                with dpg.child_window(width = -200):
                    with dpg.plot(label='Plot', height=-1, width=-1) as plot_id:
                        self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="")
                        self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="")
                        dpg.set_axis_limits_auto(self.x_axis)
                        dpg.set_axis_limits_auto(self.y_axis)
                
                with dpg.child_window(width = 200) as control_panel:
                    def scale_slider_update(sender, app_data,user_data):
                        self.scale = app_data
                        dpg.fit_axis_data(self.y_axis)
                        
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
        ydata = self.buffer.buffer.T.copy()
        shift = np.arange(ydata.shape[0])[None,].T * self.scale
        ydata += shift #shift the data for different channel

        ydata = ydata.tolist()
 
        xdata = np.arange(len(ydata[0]))
        if self.time_scale:
            xdata = xdata *self.time_scale

        xdata = xdata.tolist()


        if self.first_run:
            self.first_run = False
            
            self.series_name = [f'{i}_{self.name}' for i in range(len(ydata))]

            for i in range(len(ydata)):
                dpg.add_line_series(xdata, ydata[i], label=self.series_name[i], parent=self.y_axis, tag=self.series_name[i])
                dpg.fit_axis_data(self.y_axis)
                dpg.fit_axis_data(self.x_axis)
            
            # dpg.add_vline_series([1.0], parent = self.y_axis, tag=self.time_cursor)
        else:
            for i in range(len(ydata)):
                dpg.set_value(self.series_name[i],(xdata,ydata[i])) #must be called in main thread
                
    def get_IOspecs(self) -> Tuple[List[Message], List[Message]]:
        return ([NumpyMessage],[])

class DummpyControlSink(Sink):
    def process(self, message: Message):
        if type(message) is RecomputeTemplateControlMessage:
            self.log(logging.DEBUG, 'Recompute message received')  
        else:
            self.log(logging.DEBUG, f'Received message {type(message)}')
                      
           
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with ProcessorContext() as ctx:
        sineWave = SineTimeSource(0.1,frequency = 12.3, channel_num=10, sampling_frequency=100)
        analog_visualizer = AnalogVisualizer('Synchronized signals',scale=20, buffer_length=6000)
        control = DummpyControlSink()
        
        gui = GUIProcessor(internal_buffer_size=5000)
        
        sineWave.connect(gui)
        gui.register_visualizer(analog_visualizer,filters=['sine'], control_targets=[control])
        ctx.register_processors(gui, sineWave,  )
        
        ctx.start()