'''
A Visulizer to be used with GUIProcessor
It is used to visualize particular messages
'''

from abc import ABC
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.Processor import Message
import numpy as np
import shortuuid
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from typing import List, Union

class Visualizer:
    def __init__(self, name) -> None:
        self.name = name

    def init_gui(self):
        # build the GUI here
        raise NotImplementedError('init_gui must be implemented')

    def update(self, messages:List[Message]):
        # function called to update the figures
        raise NotImplementedError('update must be implemented')


def shiftSignal4plot(x,shift_scale=10):
    # x should be in (channel x time)
    # shift the signal so they can be plotted in different rows
    shift = np.arange(x.shape[0])[None,].T *shift_scale
    return x+shift
