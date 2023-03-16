import logging
import time

import git

from pyneurode.processor_node.AnalogSignalMerger import AnalogSignalMerger
from pyneurode.processor_node.AnalogVisualizer import *
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.LatencyVisualizer import LatencyVisualizer
from pyneurode.processor_node.MountainsortTemplateProcessor import \
    MountainsortTemplateProcessor
from pyneurode.processor_node.OSCSource import ADCMessage, OSCSource
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.SpikeClusterVisualizer import \
    SpikeClusterVisualizer
from pyneurode.processor_node.TemplateMatchProcessor import \
    SpikeTrainMessage, TemplateMatchProcessor, TemplateMatchProcessor
from pyneurode.processor_node.TuningCurve2DVisualizer import TuningCurve2DVisualizer
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
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:6] #also record down the SHA for later reference.

    start  = time.time()


    with ProcessorContext() as ctx:
        #TODO: need some way to query the output type of processor easily

        # reading too fast may overflow the pipe
        zmqSource = FileEchoSource(interval=0.01, filename='data/openfield_packets.pkl', 
                                filetype='message',batch_size=3)
        
        # zmqSource = ZmqSource(time_bin = 0.01)
        oscSource  =  OSCSource('127.0.0.1', 2323, '/data')

        # zmqSource = FileEchoSource(interval=0.02, filename='E:\decoder_test_data\JM8_2022-09-03_16-57-23_test1\JM8_20220903_165714_9b0440_packets.pkl', 
        #                         filetype='message',batch_size=3)
        
        templateTrainProcessor = MountainsortTemplateProcessor(interval=0.01,
                min_num_spikes=1000, training_period=None)
        templateMatchProcessor = TemplateMatchProcessor(interval=0.01,time_bin=0.01)
        signalMerger =  AnalogSignalMerger({ADCMessage:(2,1), SpikeTrainMessage:(1,1)}, interval=0.1)

        gui = GUIProcessor(internal_buffer_size=5000)
        # packet_save = FileEchoSink(f'data/openfield_packets.pkl')
        # zmqSource.connect(packet_save)
        
        zmqSource.connect(templateTrainProcessor, filters='spike')
        zmqSource.connect(templateMatchProcessor, filters='spike')

        templateTrainProcessor.connect(templateMatchProcessor)
        templateMatchProcessor.connect(signalMerger)
        
        oscSource.connect(signalMerger)
        templateMatchProcessor.connect(signalMerger, filters=[SpikeTrainMessage.dtype])

        oscSource.connect(gui, [ADCMessage.dtype])
        signalMerger.connect(gui)
        templateMatchProcessor.connect(gui, ['df_sort','metrics'])


        analog_visualizer = AnalogVisualizer('Synchronized signals',scale=20, buffer_length=6000)
        pos_visualizer = AnalogVisualizer('pos', buffer_length=6000)
        cluster_vis = SpikeClusterVisualizer('cluster_vis')
        latency_vis = LatencyVisualizer('latency')
        tuningCurve2DVisualizer = TuningCurve2DVisualizer('Rate map', scale=10, buffer_length=6000, 
                                                         xbin=100, ybin=100, xmax=500, ymax=500)


        gui.register_visualizer(analog_visualizer, filters=['synced_data'])
        gui.register_visualizer(pos_visualizer, filters=[ADCMessage.dtype])
        gui.register_visualizer(cluster_vis, filters=['df_sort'])
        gui.register_visualizer(latency_vis, filters=['metrics'])
        gui.register_visualizer(tuningCurve2DVisualizer, filters=['synced_data'])
