from pyneurode.processor_node.ArduinoSink import ArduinoSink
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.Spike2ArduinoProcessor import Spike2ArduinoProcessor
from pyneurode.processor_node.SpikeSortProcessor import SpikeSortProcessor
from pyneurode.processor_node.SyncDataProcessor import SyncDataProcessor
import pyneurode as dc
import logging
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.AnalogVisualizer import *
from pyneurode.processor_node.SpikeClusterVisualizer import SpikeClusterVisualizer
from pyneurode.processor_node.LatencyVisualizer import LatencyVisualizer
from pyneurode.processor_node.ZmqSource import ZmqSource
import time 
from datetime import datetime

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
        # fileReader = FileReaderSource('../pyneurode_study/data/data_packets_M2_D23.pkl',interval=0.05)
        zmqSource = ZmqSource(adc_channel=20,time_bin = 0.01)
        spikeSortProcessor = SpikeSortProcessor(interval=0.002, min_num_spikes=2000,time_bin=0.01)
        syncDataProcessor = SyncDataProcessor(interval=0.02)
        gui = GUIProcessor()
        animal_name = input('Please enter the designation of the animal: ')
        now = datetime.now()
        filesave = FileEchoSink(f'data/{animal_name}_{now.strftime("%Y%m%d_%H%M%S")}_df_sort_.pkl')
        packet_save = FileEchoSink(f'data/{animal_name}_{now.strftime("%Y%m%d_%H%M%S")}_packets.pkl')
        spike2arduino = Spike2ArduinoProcessor([0,1,2,3,4,5], [13,12,11,10,9,8], 0.002)
        arduinoSink = ArduinoSink("COM4")

        zmqSource.connect(spikeSortProcessor, filters='spike')
        zmqSource.connect(packet_save)
        spikeSortProcessor.connect(syncDataProcessor)
        spikeSortProcessor.connect(spike2arduino)
        spike2arduino.connect(arduinoSink)
        spikeSortProcessor.connect(filesave, ['df_sort'])
        zmqSource.connect(syncDataProcessor, 'adc_data')

        zmqSource.connect(gui, 'adc_data')
        syncDataProcessor.connect(gui)
        spikeSortProcessor.connect(gui, ['df_sort','metrics'])


        analog_visualizer = AnalogVisualizer('Synchronized signals',scale=20, buffer_length=6000)
        pos_visualizer = AnalogVisualizer('pos', buffer_length=6000)
        cluster_vis = SpikeClusterVisualizer('cluster_vis', max_spikes=200)
        latency_vis = LatencyVisualizer('latency')


        gui.register_visualizer(analog_visualizer,filters=['synced_data'])
        gui.register_visualizer(pos_visualizer, filters=['adc_data'])
        gui.register_visualizer(cluster_vis, filters=['df_sort'])
        gui.register_visualizer(latency_vis, filters=['metrics'])

        ctx.register_processors(zmqSource, spikeSortProcessor, syncDataProcessor, gui, filesave, arduinoSink, spike2arduino, packet_save)
        
        ctx.start()