import logging
from datetime import datetime

import dearpygui.dearpygui as dpg

import pyneurode as dc
from pyneurode.node_editor.node_editor import NodeManager
from pyneurode.processor_node.AnalogTriggerControl import AnalogTriggerControl
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
from pyneurode.processor_node.TuningCurveVisualizer import TuningCurveVisualizer
from pyneurode.processor_node.ZmqPublisherSink import ZmqMessage, ZmqPublisherSink
from pyneurode.processor_node.ZmqSource import ZmqSource

logging.basicConfig(level=logging.INFO)

dpg.create_context()

if __name__ == '__main__':
    msg = ZmqMessage({'rewarded': True})

    with ProcessorContext(auto_start=False) as ctx:
        # reading too fast may overflow the pipe
        zmqSource = FileEchoSource(interval=0.01, filename='data/M7_2022-07-16_17-17-33_test1/packets_new.pkl',
                                   filetype='message', batch_size=3)

        # zmqSource = ZmqSource(adc_channel=20,time_bin = 0.01)
        templateTrainProcessor = MountainsortTemplateProcessor(interval=0.01, min_num_spikes=500, training_period=None)
        templateMatchProcessor = TemplateMatchProcessor(interval=0.01, time_bin=0.01)
        syncDataProcessor = SyncDataProcessor(interval=0.02)
        analogControl = AnalogTriggerControl("Trigger control", message2send=msg, hysteresis=1)
        gui = GUIProcessor(internal_buffer_size=5000)
        spike2arduino = Spike2ArduinoTriggerProcessor([3], [13])
        zmqPublisherSink = ZmqPublisherSink(verbose=True)
        visualizer = TuningCurveVisualizer(x_variable_idx=-1, nbin=100, bin_size=2.7/100, buffer_length=6000)

        zmqSource.connect(templateTrainProcessor, filters='spike')
        zmqSource.connect(templateMatchProcessor, filters='spike')
        templateTrainProcessor.connect(templateMatchProcessor)
        templateMatchProcessor.connect(syncDataProcessor)
        templateMatchProcessor.connect(spike2arduino)
        zmqSource.connect(syncDataProcessor, 'adc_data')

        analog_visualizer = AnalogVisualizer(scale=20, buffer_length=6000)
        pos_visualizer = AnalogVisualizer(buffer_length=6000)
        cluster_vis = SpikeClusterVisualizer()
        latency_vis = LatencyVisualizer()

        gui.register_visualizer(syncDataProcessor, analog_visualizer)
        gui.register_visualizer(zmqSource, pos_visualizer)
        gui.register_visualizer(templateMatchProcessor, cluster_vis, control_targets=templateTrainProcessor)
        gui.register_visualizer(templateTrainProcessor, latency_vis)
        gui.register_visualizer(syncDataProcessor, analogControl, control_targets=zmqPublisherSink)
        gui.register_visualizer(syncDataProcessor, visualizer)

        NodeManager(ctx)
