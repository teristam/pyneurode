import logging
import time
from datetime import datetime

import dearpygui.dearpygui as dpg

import pyneurode as dc
from pyneurode.processor_node.AnalogVisualizer import *
from pyneurode.processor_node.ArduinoSink import ArduinoSink
from pyneurode.processor_node.ArduinoTriggerSink import ArduinoTriggerSink
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.LatencyVisualizer import LatencyVisualizer
from pyneurode.processor_node.MountainsortTemplateProcessor import \
    MountainsortTemplateProcessor
from pyneurode.processor_node.TemplateMatchProcessor import TemplateMatchProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.Spike2ArduinoTriggerProcessor import \
    Spike2ArduinoTriggerProcessor
from pyneurode.processor_node.SpikeClusterVisualizer import \
    SpikeClusterVisualizer
from pyneurode.processor_node.SpikeSortProcessor import SpikeSortProcessor
from pyneurode.processor_node.SyncDataProcessor import SyncDataProcessor
from pyneurode.processor_node.ZmqSource import ZmqSource

'''
GUI design architecture:
have one single GUI node in the main thread where all the others can connect to 
- Different windows will be shown for different message type
- Provide different common visualization for common types
- Need to provide a mechanism to link the message to the visualization type
- provide a easy to use interface to build new visualization plugin

'''

logging.basicConfig(level=logging.INFO)

if  __name__ == '__main__':
    start  = time.time()

    with ProcessorContext() as ctx:
        #TODO: need some way to query the output type of processor easily

        # reading too fast may overflow the pipe
        zmqSource = FileEchoSource(interval=0.01, filename='E:/decoder_test_data/M10_2022-06-23_12-43-50_test4/M10_test4_20220623_124341_packets.pkl', 
                                filetype='message',batch_size=3)
        
        # zmqSource = ZmqSource(adc_channel=20,time_bin = 0.01)
        templateTrainProcessor = MountainsortTemplateProcessor(interval=0.01,min_num_spikes=500, training_period=20)
        templateMatchProcessor = TemplateMatchProcessor(interval=0.01,time_bin=0.01)
        syncDataProcessor = SyncDataProcessor(interval=0.02)
        gui = GUIProcessor(internal_buffer_size=5000)
        spike2arduino = Spike2ArduinoTriggerProcessor([3], [13])
        arduinoSink = ArduinoTriggerSink("COM5")

        zmqSource.connect(templateTrainProcessor, filters='spike')
        zmqSource.connect(templateMatchProcessor, filters='spike')
        templateTrainProcessor.connect(templateMatchProcessor)
        templateMatchProcessor.connect(syncDataProcessor)
        templateMatchProcessor.connect(spike2arduino)
        spike2arduino.connect(arduinoSink)
        zmqSource.connect(syncDataProcessor, 'adc_data')

        zmqSource.connect(gui, 'adc_data')
        syncDataProcessor.connect(gui)
        templateMatchProcessor.connect(gui, ['df_sort','metrics'])


        analog_visualizer = AnalogVisualizer('Synchronized signals',scale=20, buffer_length=6000)
        pos_visualizer = AnalogVisualizer('pos', buffer_length=6000)
        cluster_vis = SpikeClusterVisualizer('cluster_vis')
        latency_vis = LatencyVisualizer('latency')


        gui.register_visualizer(analog_visualizer,filters=['synced_data'])
        gui.register_visualizer(pos_visualizer, filters=['adc_data'])
        gui.register_visualizer(cluster_vis, filters=['df_sort'])
        gui.register_visualizer(latency_vis, filters=['metrics'])

        ctx.register_processors(zmqSource, templateMatchProcessor, templateTrainProcessor,
                                syncDataProcessor, gui, spike2arduino)
        
        ctx.start()
