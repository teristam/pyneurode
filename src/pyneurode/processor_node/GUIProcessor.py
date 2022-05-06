from __future__ import annotations # for postponed evaluation
from multiprocessing import Event
from pyneurode.processor_node.Processor import Processor, Message
from .BatchProcessor import BatchProcessor
import dearpygui.dearpygui as dpg
import logging
from queue import Empty
import numpy as np
from .Visualizer import Visualizer
from typing import List, Dict
import time 


class GUIProcessor(BatchProcessor):
    """    A GUIProcessor is a processor that receive messages for plotting and displaying the GUI.
    
    .. note:: its process() function will be called frequently. Avoid doing time consuming processing there.
    """

    def __init__(self, internal_buffer_size=1000):
        """

        Args:
            internal_buffer_size (int, optional): size of the internal buffer to hold messages before they are displayed. Defaults to 1000.
        """
        super().__init__(interval=1/60, internal_buffer_size=internal_buffer_size)
        self.visualizers:Dict(List(Visualizer)) = {}

    def startup(self):
        super().startup()

        self.log(logging.DEBUG, 'GUI starting up')
        self.plot_data = (np.arange(500), np.zeros((500,)))
        self.data_count = 0

        # build the GUI
        dpg.create_context()
        dpg.configure_app(docking=True, docking_space=True, init_file='dpg_layout.ini')
        dpg.create_viewport()
        dpg.setup_dearpygui()

        self.make_GUI()
        dpg.show_viewport()



    def register_visualizer(self, visualizer:Visualizer, filters:List[str]):
        """Register a Visualizer object with the GUIProcessor. Only message of types specified in the
        filter list will be passed to the visualizer.

        Args:
            visualizer (Visualizer): visualizer to be creat the GUI and call during frame update
            filters (List[str]): names of message type to be passed to the visualizer. 
        """
        # 
        # each visualizer is associate with one or more message type definied in the filters
        
        for f in filters:
            if not f in self.visualizers:
                self.visualizers[f] = []
                self.visualizers[f].append(visualizer)
            else:
                self.visualizers[f].append(visualizer)

    def main_loop(self):
        
        while not self.shutdown_event.is_set():
            while dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
                if self.end_time is None:
                    self.end_time  = time.monotonic() + self.interval
                
                # Wait till the future trigger point
                time2wait = self.end_time - time.monotonic()

                if time2wait>0:
                    time.sleep(time2wait)

                if time.monotonic() >= self.end_time:
                    self.end_time = time.monotonic() + self.interval #update future trigger time
                    data_list = self.ring_buffer.readAvailable()
                    data = self._process(data_list)
                    self.send(data)

            self.shutdown_event.set() #shutdown all the nodes
    
    def shutdown(self):
        self.log(logging.DEBUG, 'Destroying GUI context')
        dpg.destroy_context()
        
    def make_control_panel(self):
        # make the window for control panel
        with dpg.window(label='Control panel',width=-1, height=-1):
            dpg.add_button(label='Save layout', callback=lambda: dpg.save_init_file('dpg_layout.ini'))

    def make_GUI(self):
        # build the GUI
        self.make_control_panel()

        for msg_name, visualizer_list in self.visualizers.items():
            for v in visualizer_list:
                print('Initializing ', v.name)
                v.init_gui()

    def process(self, messages: List[Message] = None):
        # sort the messages according to their type
        #TODO use a separate thread for each visualizer
        msg_list = {}

        for msg in messages:
            if not msg.type in msg_list:
                msg_list[msg.type] = []
                msg_list[msg.type].append(msg)
            else:
                msg_list[msg.type].append(msg)

        # pass each type of messages to their respective visualizer        
        for msg_type, msgs in msg_list.items():
            if  msg_type in self.visualizers:
                for v in self.visualizers[msg_type]:
                        v.update(msgs)
