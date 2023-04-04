import logging
import time
from datetime import datetime

import dearpygui.dearpygui as dpg
import git

import pyneurode as dc
from pyneurode.processor_node.AnalogTriggerControl import AnalogTriggerControl
from pyneurode.processor_node.AnalogVisualizer import *
from pyneurode.processor_node.ArduinoSink import ArduinoSink
from pyneurode.processor_node.ArduinoTriggerSink import ArduinoTriggerSink
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.LatencyVisualizer import LatencyVisualizer
from pyneurode.processor_node.MountainsortTemplateProcessor import \
    MountainsortTemplateProcessor
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.Spike2ArduinoTriggerProcessor import \
    Spike2ArduinoTriggerProcessor
from pyneurode.processor_node.SpikeClusterVisualizer import \
    SpikeClusterVisualizer
from pyneurode.processor_node.SpikeSortProcessor import SpikeSortProcessor
from pyneurode.processor_node.SyncDataProcessor import SyncDataProcessor
from pyneurode.processor_node.TemplateMatchProcessor import \
    TemplateMatchProcessor
from pyneurode.processor_node.TuningCurveVisualizer import \
    TuningCurveVisualizer
from pyneurode.processor_node.ZmqPublisherSink import (ZmqMessage,
                                                       ZmqPublisherSink)
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
    msg = ZmqMessage({'rewarded':True})


    with ProcessorContext() as ctx:
        #TODO: need some way to query the output type of processor easily
        
        zmqSource = ZmqSource(adc_channel=20,time_bin = 0.01)

        
        templateTrainProcessor = MountainsortTemplateProcessor(interval=0.01,
                min_num_spikes=4000)
        templateMatchProcessor = TemplateMatchProcessor(interval=0.005,time_bin=0.01,
                                                        do_pca=False, internal_buffer_size=10000)

        # animal_name = input('Please enter the designation of the animal: ')
        animal_name = 'latency_ch32'
        now = datetime.now()
        filesave = FileEchoSink(f'data/{animal_name}_{now.strftime("%Y%m%d_%H%M%S")}_{sha}_df_sort.pkl')

        zmqSource.connect(templateTrainProcessor, filters='spike')
        zmqSource.connect(templateMatchProcessor, filters='spike')

        templateTrainProcessor.connect(templateMatchProcessor)

        templateMatchProcessor.connect(filesave, ['df_sort'])
