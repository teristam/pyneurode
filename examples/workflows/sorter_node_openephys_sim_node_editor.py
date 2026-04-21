import logging

import dearpygui.dearpygui as dpg

import pyneurode as dc
from pyneurode.node_editor.node_editor import NodeManager
from pyneurode.processor_node.AnalogVisualizer import *
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.LatencyVisualizer import LatencyVisualizer
from pyneurode.processor_node.MountainsortTemplateProcessor import (
    MountainsortTemplateProcessor,
)
from pyneurode.processor_node.TemplateMatchProcessor import TemplateMatchProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.SpikeClusterVisualizer import SpikeClusterVisualizer
from pyneurode.processor_node.SyncDataProcessor import SyncDataProcessor
from pyneurode.processor_node.SpikeGeneratorSource import (
    make_neurons,
    SpikeGeneratorSource,
)

logging.basicConfig(level=logging.INFO)

dpg.create_context()

if __name__ == "__main__":
    templates = np.load(
        "examples/spikes/spikes_waveforms.npy"
    )  # cells x tetrode x channel x time

    Fs = 30000
    # Restrict the size of the template to avoid the waveform alignment to go wrong
    neurons = make_neurons(
        [2, 2],
        firing_rate=[[30, 10], [20, 30]],
        firing_time=[[(None), (Fs * 30, None)], [(Fs * 30, None), (Fs * 30, None)]],
        templates=templates[:, :, :, :130],
    )

    with ProcessorContext(auto_start=False) as ctx:
        source = SpikeGeneratorSource(neurons)
        templateTrainProcessor = MountainsortTemplateProcessor(
            interval=0.01, min_num_spikes=2000, training_period=None
        )
        templateMatchProcessor = TemplateMatchProcessor(interval=0.01, time_bin=0.01)
        syncDataProcessor = SyncDataProcessor(interval=0.02, ignore_adc=True)
        gui = GUIProcessor(internal_buffer_size=5000)

        source.connect(templateTrainProcessor, filters="spike")
        source.connect(templateMatchProcessor, filters="spike")
        templateTrainProcessor.connect(templateMatchProcessor)

        analog_visualizer = AnalogVisualizer(scale=20, buffer_length=6000)
        pos_visualizer = AnalogVisualizer(buffer_length=6000)
        cluster_vis = SpikeClusterVisualizer(max_spikes=200)
        latency_vis = LatencyVisualizer()

        gui.register_visualizer(syncDataProcessor, analog_visualizer)
        gui.register_visualizer(source, pos_visualizer)
        gui.register_visualizer(templateMatchProcessor, cluster_vis, control_targets=templateTrainProcessor)
        gui.register_visualizer(templateTrainProcessor, latency_vis)

        NodeManager(ctx)
