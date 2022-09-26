from ctypes import alignment
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
import inspect
import dearpygui.dearpygui as dpg
from pyneurode.node_editor import node_editor

dpg.create_context()

'''
GUI design architecture:
have one single GUI node in the main thread where all the others can connect to 
- Different windows will be shown for different message type
- Provide different common visualization for common types
- Need to provide a mechanism to link the message to the visualization type
- provide a easy to use interface to build new visualization plugin

'''

def make_node(processor:Processor) -> Tuple[Dict, Dict]:
    '''
    Parse the object information and make it into a node
    '''
    with dpg.node(label=processor.proc_name):
        sig = inspect.signature(processor.__init__)
        
        input_cls, output_cls = processor.get_IOspecs()
        
        
        # Add input and output nodes
        inputs = {}
        for input in input_cls:
            with dpg.node_attribute(label=processor.proc_name, attribute_type=dpg.mvNode_Attr_Input) as node_input:
                dpg.add_text(input.__name__)
                inputs[input.__name__] = node_input
                
        
        outputs = {}
        for output in output_cls:
            with dpg.node_attribute(label=processor.proc_name, attribute_type=dpg.mvNode_Attr_Output) as node_output:
                dpg.add_text(output.__name__, indent=200)
                outputs[output.__name__] = node_output
                
                
        
        for param in sig.parameters.keys():
            with dpg.node_attribute(label=processor.proc_name, attribute_type=dpg.mvNode_Attr_Static): #static attribute
                if sig.parameters[param].annotation == 'float':
                    dpg.add_input_float(label=param, width = 150)
                elif sig.parameters[param].annotation == 'int':
                    dpg.add_input_int(label=param, width = 150)
                    
        return inputs, outputs
                    
                    
def link_callback(sender,app_data):
    print(f'Connecting {app_data[0]} and {app_data[1]}')
    dpg.add_node_link(app_data[0], app_data[1], parent=sender)
                    

if  __name__ == '__main__':
    start  = time.time()
    echofile = FileEchoSink.get_temp_filename()

    with ProcessorContext(auto_start=False) as ctx:

        sineWave = SineTimeSource(0.1,frequency = 12.3, channel_num=10, sampling_frequency=100)
        scaleProcessor = ScaleProcessor(4,5)
        analog_visualizer = AnalogVisualizer('v1',scale=4,time_scale=1/100)
        
        # gui = GUIProcessor()
        # gui.register_visualizer(analog_visualizer,filters=['sine'])

        sineWave.connect(scaleProcessor)
        # scaleProcessor.connect(gui,'sine')

        # ctx.start()
        
        
        node_editor.init_node_editor(ctx)
        
            

        
        