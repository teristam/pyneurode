from ctypes import alignment

import dearpygui.dearpygui as dpg

import pyneurode as dc
from pyneurode.node_editor.node_editor import NodeManager
from pyneurode.processor_node.AnalogVisualizer import *
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.MountainsortTemplateProcessor import \
    MountainsortTemplateProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.ScaleProcessor import ScaleProcessor
from pyneurode.processor_node.SineTimeSource import SineTimeSource

dpg.create_context()

'''
GUI design architecture:
have one single GUI node in the main thread where all the others can connect to 
- Different windows will be shown for different message type
- Provide different common visualization for common types
- Need to provide a mechanism to link the message to the visualization type
- provide a easy to use interface to build new visualization plugin

'''

if  __name__ == '__main__':
    start  = time.time()
    echofile = FileEchoSink.get_temp_filename()

    with ProcessorContext(auto_start=False) as ctx:

        sineWave = SineTimeSource(interval = 0.1,frequency = 12.3, channel_num=10, sampling_frequency=100)
        scaleProcessor = ScaleProcessor(4,5)
        analog_visualizer = AnalogVisualizer('v1',scale=4,time_scale=1/100)
        sineWave = SineTimeSource(interval = 0.1,frequency = 12.3, channel_num=10, sampling_frequency=100)
        scaleProcessor = ScaleProcessor(4,5)
        
        # gui = GUIProcessor()
        # gui.register_visualizer(analog_visualizer,filters=['sine'])

        sineWave.connect(scaleProcessor)
        # scaleProcessor.connect(gui,'sine')

        # ctx.start()
        
        
        NodeManager(ctx)
        
            

        
        