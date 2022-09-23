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
from pyneurode.processor_node.SpikeGeneratorSource import make_neurons, SpikeGeneratorSource

import pyneurode.node_editor.node_editor as node_editor

logging.basicConfig(level=logging.INFO)

if  __name__ == '__main__':
    templates = np.load('src/pyneurode/data/spikes_waveforms.npy') # cells x electrode x timepoints
    Fs=30000
    # Restrict the size of the template to avoid the waveform alignment to go wrong
    neurons = make_neurons([2,2],firing_rate=[[30,10],[20,30]],firing_time=[[(None),(Fs*30, None)],[(Fs*30,None),(Fs*30,None)]], templates=templates[:,:,:,:130])
    

    with ProcessorContext(auto_start=False) as ctx:
        #TODO: need some way to query the output type of processor easily

        # reading too fast may overflow the pipe
        source = SpikeGeneratorSource(neurons)
        templateTrainProcessor = MountainsortTemplateProcessor(interval=0.01,min_num_spikes=500,training_period=None)
        templateMatchProcessor = TemplateMatchProcessor(interval=0.01,time_bin=0.01)
        syncDataProcessor = SyncDataProcessor(interval=0.02, ignore_adc=True)
        gui = GUIProcessor(internal_buffer_size=5000)
        spike2arduino = Spike2ArduinoTriggerProcessor([3], [13])
        arduinoSink = ArduinoTriggerSink("COM5")

        source.connect(templateTrainProcessor, filters='spike')
        source.connect(templateMatchProcessor, filters='spike')
        templateTrainProcessor.connect(templateMatchProcessor)
        templateMatchProcessor.connect(syncDataProcessor)
        templateMatchProcessor.connect(spike2arduino)
        spike2arduino.connect(arduinoSink)
        source.connect(syncDataProcessor, 'adc_data')

        source.connect(gui, 'adc_data')
        syncDataProcessor.connect(gui)
        templateMatchProcessor.connect(gui, ['df_sort','metrics'])
        templateTrainProcessor.connect(gui, ['metrics'])


        analog_visualizer = AnalogVisualizer('Synchronized signals',scale=20, buffer_length=6000)
        pos_visualizer = AnalogVisualizer('pos', buffer_length=6000)
        cluster_vis = SpikeClusterVisualizer('cluster_vis', max_spikes=200)
        latency_vis = LatencyVisualizer('latency')


        gui.register_visualizer(analog_visualizer,filters=['synced_data'])
        gui.register_visualizer(pos_visualizer, filters=['adc_data'])
        gui.register_visualizer(cluster_vis, filters=['df_sort'], control_targets=templateTrainProcessor)
        gui.register_visualizer(latency_vis, filters=['metrics'])

        
        # ctx.start()
        
        node_editor.init_node_editor(ctx)
