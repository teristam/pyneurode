from ctypes import alignment

import dearpygui.dearpygui as dpg

import pyneurode as dc
from pyneurode.node_editor.node_editor import NodeManager
from pyneurode.processor_node.AnalogVisualizer import *
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.MountainsortTemplateProcessor import \
    MountainsortTemplateProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.ScaleProcessor import ScaleProcessor
from pyneurode.processor_node.SineTimeSource import SineTimeSource

dpg.create_context()


if  __name__ == '__main__':
    start  = time.time()
    echofile = FileEchoSink.get_temp_filename()

    with ProcessorContext(auto_start=False) as ctx:

        sineWave = SineTimeSource(interval = 0.1,frequency = 12.3, channel_num=10, sampling_frequency=100)
        scaleProcessor = ScaleProcessor(4,5)
        analog_visualizer1 = AnalogVisualizer(scale=4,time_scale=1/100)
        analog_visualizer2 = AnalogVisualizer(scale=4,time_scale=1/100)

        # sineWave = SineTimeSource(interval = 0.1,frequency = 12.3, channel_num=10, sampling_frequency=100)
        # scaleProcessor = ScaleProcessor(4,5)
        
        #TODO now the same type of message, although coming from different processor, will got mixed together when received at the GUIProcessor
        gui = GUIProcessor()
        gui.register_visualizer(sineWave, analog_visualizer1)
        gui.register_visualizer(scaleProcessor, analog_visualizer2)


        sineWave.connect(scaleProcessor)
        # sineWave.connect(gui)
        # scaleProcessor.connect(gui)
        # scaleProcessor.connect(gui,'sine')

            
        NodeManager(ctx)
        
            

        
        