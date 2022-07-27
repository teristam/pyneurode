from pyneurode.processor_node.AnalogTriggerControl import AnalogTriggerControl
from pyneurode.processor_node.ArduinoSink import ArduinoSink
from pyneurode.processor_node.ArduinoTriggerSink import ArduinoTriggerSink
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.MountainsortTemplateProcessor import MountainsortTemplateProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.Spike2ArduinoTriggerProcessor import Spike2ArduinoTriggerProcessor
from pyneurode.processor_node.SpikeSortProcessor import SpikeSortProcessor
from pyneurode.processor_node.SyncDataProcessor import SyncDataProcessor
import pyneurode as dc
import logging
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.AnalogVisualizer import *
from pyneurode.processor_node.SpikeClusterVisualizer import SpikeClusterVisualizer
from pyneurode.processor_node.LatencyVisualizer import LatencyVisualizer
from pyneurode.processor_node.TemplateMatchProcessor import TemplateMatchProcessor
from pyneurode.processor_node.TuningCurveVisualizer import TuningCurveVisualizer
from pyneurode.processor_node.ZmqPublisherSink import ZmqMessage, ZmqPublisherSink
from pyneurode.processor_node.ZmqSource import ZmqSource
import time 
from datetime import datetime
import git

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
    msg = ZmqMessage({'rewarded':True})


    with ProcessorContext() as ctx:
        #TODO: need some way to query the output type of processor easily

        # reading too fast may overflow the pipe
        # zmqSource = FileEchoSource(interval=0.01, filename='data/M7_test1_20220725_154822_b68d06_packets.pkl', 
        #                         filetype='message',batch_size=3)
        
        zmqSource = ZmqSource(adc_channel=20,time_bin = 0.01)
        templateTrainProcessor = MountainsortTemplateProcessor(interval=0.01,
                min_num_spikes=20000, calibration_time = 60*2, training_period=None)
        templateMatchProcessor = TemplateMatchProcessor(interval=0.01,time_bin=0.01)
        syncDataProcessor = SyncDataProcessor(interval=0.02)
        gui = GUIProcessor(internal_buffer_size=5000)
        analogControl = AnalogTriggerControl("Trigger control", message2send=msg, hysteresis=4)
        visualizer = TuningCurveVisualizer('Tuning curves', x_variable_idx = -1, nbin=100, bin_size=2.7/100, buffer_length=6000)


        spike2arduino = Spike2ArduinoTriggerProcessor([0,1,2,3,4,5], [13,12,11,10,9,8])
        arduinoSink = ArduinoTriggerSink("COM4")
        zmqPublisherSink = ZmqPublisherSink(verbose=True)

        #potential bug: when it takes too long to acquire the spikes, the syncdata buffer may overflow

        animal_name = input('Please enter the designation of the animal: ')
        now = datetime.now()
        filesave = FileEchoSink(f'data/{animal_name}_{now.strftime("%Y%m%d_%H%M%S")}_{sha}_df_sort.pkl')
        packet_save = FileEchoSink(f'data/{animal_name}_{now.strftime("%Y%m%d_%H%M%S")}_{sha}_packets.pkl')

        zmqSource.connect(templateTrainProcessor, filters='spike')
        zmqSource.connect(templateMatchProcessor, filters='spike')
        zmqSource.connect(packet_save)

        templateTrainProcessor.connect(templateMatchProcessor)
        templateMatchProcessor.connect(spike2arduino,'spike_train')
        templateMatchProcessor.connect(syncDataProcessor)

        spike2arduino.connect(arduinoSink)
        templateMatchProcessor.connect(filesave, ['df_sort'])
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
        gui.register_visualizer(analogControl, filters=['synced_data'], control_targets=[zmqPublisherSink, filesave])
        gui.register_visualizer(visualizer=visualizer, filters=['synced_data'])

        ctx.start()