'''
Provide function to receive signal via Open Sound protocol, mainly designed to 
accept data from Bonsai

'''

import logging
import threading
from typing import Any, Optional

import numpy as np
from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher

from pyneurode.processor_node.AnalogVisualizer import AnalogVisualizer
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.Processor import EchoSink, Source
from pyneurode.processor_node.ProcessorContext import ProcessorContext


class ADCMessage(Message):
    type = 'adc_message'
    def __init__(self, data: np.ndarray, timestamp: Optional[float] = None):
        '''
        data should be in the form of time x channel
        '''
        assert isinstance(data, np.ndarray), "Error, data must be a np.ndarray"
        super().__init__(self.type, data, timestamp)



def receive_message(addr, args, *osc_message):
    message_list = args[0]
    lock = threading.Lock()
    lock.acquire()
    print(osc_message)
    message_list.append(ADCMessage(np.array(osc_message).reshape(1,-1)))
    lock.release()

    

class OSCSource(Source):
    '''
    Read message from OSC and return them as analog message
    '''
    def __init__(self, ip_addr, port, channel):
        super().__init__()
        self.ip_addr = ip_addr
        self.port = port
        self.channel = channel
        self.message_list = []
        
    
    def startup(self):
        self.lock = threading.Lock()
        self.dispatcher = Dispatcher()
        self.dispatcher.map('/data', receive_message, self.message_list)

        self.server = osc_server.ThreadingOSCUDPServer(
            (self.ip_addr, self.port), self.dispatcher)   
        
        self.data_thread = threading.Thread(target = self.server.serve_forever)
        self.data_thread.start()
        self.log(logging.INFO, 'OSC server started')

        
    def process(self):
        if len(self.message_list)>0:
            # lock = threading.Lock()
            self.lock.acquire()
            messge2send = self.message_list.copy()
            self.message_list.clear()
            self.lock.release()
            
            return messge2send
        
    def shutdown(self):
        super().shutdown()
        self.server.shutdown()
        
        
if __name__ == '__main__':
    with ProcessorContext() as ctx:
        osc_source = OSCSource('127.0.0.1', 2323, '/data')
        
        gui = GUIProcessor()
        visualizer = AnalogVisualizer('OSC data')
        gui.register_visualizer(visualizer, [ADCMessage.type])
    
        echoSink = EchoSink()
        osc_source.connect(echoSink)
        osc_source.connect(gui)

            
            
        
        
    
