from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.SpikeSortProcessor import SpikeSortProcessor
from pyneurode.processor_node.SyncDataProcessor import SyncDataProcessor
import pyneurode as dc
import logging
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.AnalogVisualizer import *



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

    with ProcessorContext() as ctx:

        sineWave = SineTimeSource(0.1,frequency = 12.3, channel_num=10, sampling_frequency=100)
        scaleProcessor = ScaleProcessor(4,5)
        analog_visualizer = AnalogVisualizer('v1',scale=4,time_scale=1/100)
        
        gui = GUIProcessor()
        gui.register_visualizer(analog_visualizer,filters=['sine'])

        sineWave.connect(scaleProcessor)
        scaleProcessor.connect(gui,'sine')

        ctx.register_processors(sineWave, scaleProcessor, gui)
        
        ctx.start()
        
        
        
        