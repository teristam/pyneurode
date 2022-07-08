'''
A Visulizer to be used with GUIProcessor
It is used to visualize particular messages
'''

from abc import ABC
import functools
import logging
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.Message import Message
import numpy as np
import shortuuid
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from typing import List, Union
from pyneurode.processor_node.Processor import logger


class Visualizer:
    """The base Visualizer class. Intended to be extended by subclasses.
    The UI should be constructed using the `init_gui()` function and updated based on the messages passed to it in the `update()` function.
    
    
    Example::
    
        this is an exmaple showing how it works
    
    """
    def __init__(self, name:str):
        """

        Args:
            name (str): Name of the visualizer, for identification purpose
        """
        self.name = name
        self.log = functools.partial(logger, self.name)
        self.control_msgs = []


    def init_gui(self):
        """Build the GUI using dearpygui here


        Raises:
            NotImplementedError: Must be overrided by subclasses. 
        """
        
        raise NotImplementedError('init_gui must be implemented')
    
    def _refresh(self, messages:List[Message]):
        self.update(messages)
        
        # send control message out
        msg = self.control_msgs
        if len(msg) > 0:
            self.log(logging.INFO, msg )
        self.control_msgs = []
        
        return msg
    

    def update(self, messages:List[Message]):
        """ Update the interface.

        Args:
            messages (List[Message]): list of messages to be displayed

        Raises:
            NotImplementedError: Must be overrided by subclasses. 
        """
        # function called to update the figures
        raise NotImplementedError('update must be implemented')
    
    def send_control_msg(self, msg:Message):
        self.control_msgs.append(msg)


def shiftSignal4plot(x,shift_scale=10):
    # x should be in (channel x time)
    # shift the signal so they can be plotted in different rows
    shift = np.arange(x.shape[0])[None,].T *shift_scale
    return x+shift
