from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.SpikeSortProcessor import SpikeSortProcessor
from pyneurode.processor_node.SyncDataProcessor import SyncDataProcessor
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

# logging.basicConfig(level=logging.ERROR)

if  __name__ == '__main__':

    # Create a context object that manages some global states
    with ProcessorContext() as ctx:

        # Initialize the processors we need
        zmqSource = ZmqSource(adc_channel=20)
        spikeSortProcessor = SpikeSortProcessor(interval=0.001, 
            min_num_spikes=2000)
        syncDataProcessor = SyncDataProcessor(interval=0.02)
        gui = GUIProcessor()

        # Connect the processors to each others. The filters 
        # specify what type of messages are sent through that connection
        zmqSource.connect(spikeSortProcessor, filters='spike')
        spikeSortProcessor.connect(syncDataProcessor)
        zmqSource.connect(syncDataProcessor, 'adc_data')

        # Connect to the GUI processors so that messages can
        # be visualized
        zmqSource.connect(gui, 'adc_data')
        syncDataProcessor.connect(gui)
        spikeSortProcessor.connect(gui, ['df_sort','metrics'])

        # Initialize the set of visualizers
        analog_vis = AnalogVisualizer('Synchronized signals',
                scale=20, buffer_length=1000)
        pos_vis = AnalogVisualizer('pos', buffer_length=1000)
        cluster_vis = SpikeClusterVisualizer('cluster_vis')
        latency_vis = LatencyVisualizer('latency')

        # Register the visualizers to the GUI processor
        gui.register_visualizer(analog_vis,filters=['synced_data'])
        gui.register_visualizer(pos_vis, filters=['adc_data'])
        gui.register_visualizer(cluster_vis, filters=['df_sort'])
        gui.register_visualizer(latency_vis, filters=['metrics'])

        # Register all the processors to the context
        ctx.register_processors(zmqSource, spikeSortProcessor, 
            syncDataProcessor,gui)
        
        # Start all processors
        ctx.start()