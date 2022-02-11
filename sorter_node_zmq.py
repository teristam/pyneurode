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
from pyneurode.processor_node.SpikeClusterVisualizer import SpikeClusterVisualizer
from pyneurode.processor_node.LatencyVisualizer import LatencyVisualizer
from pyneurode.processor_node.ZmqSource import ZmqSource

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
        #TODO: need some way to query the output type of processor easily

        # reading too fast may overflow the pipe
        # zmqSource = FileReaderSource('data/ch256_dump.pkl', adc_channel = 260)
        zmqSource = FileEchoSource(interval=0.0002, filename='data/ch256_dump.pkl', batch_size=35, filetype='message')
        #interval=0.0002,  batch_size=35 is similar to realtime rate
        zmqSource = ZmqSource(adc_channel=20)
        # zmqSource = ZmqSource(adc_channel=68)
        repeat = 16
        # zmqSource = ZmqSource(adc_channel=16*repeat+4)


        # spikeSortProcessor = SpikeSortProcessor(interval=0.001, min_num_spikes=2000)
        # spikeSortProcessor = SpikeSortProcessor(interval=0.001, min_num_spikes=10000)
        spikeSortProcessor = SpikeSortProcessor(interval=0.001, min_num_spikes=2000*repeat, do_pca=False)


        syncDataProcessor = SyncDataProcessor(interval=0.02)
        gui = GUIProcessor()
        # filesave = FileEchoSink('data/df_sort_ch16.pkl')
        # filesave = FileEchoSink('data/df_sort_ch64.pkl')
        filesave = FileEchoSink(f'data/df_sort_ch{16*repeat}.pkl')


        zmqSource.connect(spikeSortProcessor, filters='spike')
        spikeSortProcessor.connect(syncDataProcessor)
        spikeSortProcessor.connect(filesave, ['df_sort'])
        zmqSource.connect(syncDataProcessor, 'adc_data')

        zmqSource.connect(gui, 'adc_data')
        syncDataProcessor.connect(gui)
        spikeSortProcessor.connect(gui, ['df_sort','metrics'])


        analog_visualizer = AnalogVisualizer('Synchronized signals',scale=20, buffer_length=1000)
        pos_visualizer = AnalogVisualizer('pos', buffer_length=1000)
        cluster_vis = SpikeClusterVisualizer('cluster_vis')
        latency_vis = LatencyVisualizer('latency')


        gui.register_visualizer(analog_visualizer,filters=['synced_data'])
        gui.register_visualizer(pos_visualizer, filters=['adc_data'])
        gui.register_visualizer(cluster_vis, filters=['df_sort'])
        gui.register_visualizer(latency_vis, filters=['metrics'])

        ctx.register_processors(zmqSource, spikeSortProcessor, syncDataProcessor, filesave,gui)
        
        ctx.start()