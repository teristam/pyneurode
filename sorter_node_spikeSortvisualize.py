from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.SpikeSortProcessor import SpikeSortProcessor
from pyneurode.processor_node.SyncDataProcessor import SyncDataProcessor
import pyneurode as dc
import logging
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.SpikeClusterVisualizer import *



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

        # reading too fast may overflow the pipe
        fileReader = FileEchoSource(0.1, 'data/sort_data.pkl', filetype='message')
        # sineWave = SineTimeSource(0.1,frequency = 12.3, channel_num=10, sampling_frequency=100)
        cluster_vis = SpikeClusterVisualizer('cluster_vis')

        gui = GUIProcessor()
        gui.register_visualizer(cluster_vis,filters=['df_sort'])

        fileReader.connect(gui,'df_sort')

        ctx.register_processors(fileReader, gui)
        
        ctx.start()