from cgitb import enable
import logging
from re import S
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
import time

from pyneurode.processor_node.ZmqPublisherSink import ZmqMessage, ZmqPublisherSink
from pyneurode.processor_node.ZmqSubscriberSource import ZmqSubscriberSource 

class AnalogTriggerControl(Visualizer):
    
    '''
    Send a message when an analog signal exceed a certain threshold
    '''
    def __init__(self, name:str, message2send:Message, hysteresis=0.01, buffer_length:int = 500, scale = 1, time_scale=None) -> None:
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
        self.threshold_sliders = []
        self.threshold_timeseries=[]
        self.trigger_direction = []
        self.subplots = None
        self.message = message2send #message to send when the threshold is triggered
        self.hysteresis = hysteresis # minimum among of time that must pass before the next trigger is send
        self.last_trigger_time = 0
        
    def init_gui(self):
        window_width = 800
        with dpg.window(label=self.name, width=window_width, height=500, tag=self.name, autosize=True):
            with dpg.group(horizontal=True):
                self.plot_panel = dpg.add_child_window(width=-window_width*2/4) #negative means no. of pixel from the right of its container
                self.control_panel = dpg.add_child_window(width=window_width*2/4)
                    
    def update_cell_select_list(self):
        # update the checkboxes for selecting which cell to trigger
        no_cell = self.buffer.buffer.shape[1]
        self.selected_cells = [0 for _ in range(no_cell)]
        self.series_name = [f'{i}_{self.name}' for i in range(len(self.selected_cells))]
        self.threshold_timeseries = [None for i in range(len(self.selected_cells))]
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
                slider = dpg.add_slider_float(callback=slider_changed, user_data=i, max_value=2, min_value=-2, 
                                              default_value=1.1, width=100,  show=False)
                trigger_direction = dpg.add_combo(items=['Above','Below'], default_value='Above', width=100, show=False)
                
                self.threshold_sliders.append(slider)
                self.control_list.append(gp)
                self.trigger_direction.append(trigger_direction)
                
                
    def update_subplots(self):
        
        if self.subplots is not None:
            dpg.delete_item(self.subplots)

    
        # Re-create the subplots and line series    
        
        self.log(logging.DEBUG, self.series_name)
        
        #Create subplot for each of the selected cells
        with dpg.subplots(sum(self.selected_cells), 1, parent=self.plot_panel, width=-1,height=-1) as subplots:
            self.subplots = subplots
            for i,is_selected in enumerate(self.selected_cells):
                if is_selected: # if the cell is selected
                    dpg.configure_item(self.threshold_sliders[i], show=True)
                    dpg.configure_item(self.trigger_direction[i], show=True)
                    
                    with dpg.plot():
                        dpg.add_plot_legend()
                        xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="")
                        yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="")
                        self.y_axes[i] = yaxis
                        self.x_axes[i] = xaxis
                        
                        dpg.add_line_series([],[], label=self.series_name[i], parent=yaxis, tag=self.series_name[i])
                        
                        #add the line for the threshold
                        hs = dpg.add_hline_series([dpg.get_value(self.threshold_sliders[i])], parent=yaxis, label=f'threshold ch{i}')
                        self.threshold_timeseries[i] = hs
                else:
                    # disable the slider to avoid error for unchecked signals
                    dpg.configure_item(self.threshold_sliders[i], show=False)
                    dpg.configure_item(self.trigger_direction[i], show=False)
                    
                    
        self.need_fit_axis = True #refit the axis to the data
        
    def check_if_trigger(self, data):
        # check if the threshold has been exceeded
        for i, is_selected in enumerate(self.selected_cells):
                if is_selected:
                    if np.max(data[i]) > dpg.get_value(self.threshold_sliders[i]):
                        if (time.time() - self.last_trigger_time) > self.hysteresis:
                            #avoid sending too many messages
                            self.send_control_msg(self.message)
                            self.last_trigger_time = time.time()
                


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
                    self.check_if_trigger(m.data)
                    
                except ValueError:
                    # data shape have changed, need to refresh all the plots
                    self.buffer = RingBuffer((self.buffer_length, m.data.shape[1]))
                    self.buffer.write(m.data)
                    self.first_run = True
                    # update the cell selection list
                    self.update_cell_select_list()
                    self.check_if_trigger(m.data)

        ####
        # update the plot

        # Re-arrange the data to be suitable for plotting
        ydata = self.buffer.buffer.T.copy()

        ydata = ydata.tolist()
 
        xdata = np.arange(len(ydata[0]))
        if self.time_scale:
            xdata = xdata *self.time_scale

        xdata = xdata.tolist()

        for i, is_selected in enumerate(self.selected_cells):
            if is_selected:
                #sometimes there is a slight delay when new series_names are added
                #TODO: check if the series name depends on the order of the toggle of the checkbox
                dpg.set_value(self.series_name[i],(xdata,ydata[i])) #must be called in main thread
            
        if self.need_fit_axis:
            self.need_fit_axis = False
            for i, is_selected in enumerate(self.selected_cells):
                if is_selected:
                    dpg.fit_axis_data(self.y_axes[i])   
                    dpg.fit_axis_data(self.x_axes[i])    
                                        


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    msg = ZmqMessage({'rewarded':True})


    with ProcessorContext() as ctx:
        sineWave = SineTimeSource(0.1,frequency = 0.5, channel_num=3, sampling_frequency=100)
        analog_visualizer = AnalogTriggerControl('Analog Trigger Control', message2send=msg, hysteresis=0.5)
        zmqSink = ZmqPublisherSink()
        zmqSource = ZmqSubscriberSource()
        
        gui = GUIProcessor(internal_buffer_size=5000)
        
        sineWave.connect(gui)
        
        gui.register_visualizer(analog_visualizer,filters=['sine'], control_targets=[zmqSink])
        
        
        
        ctx.start()