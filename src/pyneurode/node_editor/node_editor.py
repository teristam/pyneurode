import dearpygui.dearpygui as dpg
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
import inspect 

def make_node(processor:Processor) -> Tuple[Dict, Dict]:
    '''
    Parse the object information and make it into a node
    '''
    with dpg.node(label=processor.proc_name):
        sig = inspect.signature(processor.__init__)
        
        input_cls, output_cls = processor.get_IOspecs()
        
        
        # Add input and output nodes
        inputs = {}
        
        if not isinstance(processor, Source):
            for input in input_cls:
                with dpg.node_attribute(label=processor.proc_name, attribute_type=dpg.mvNode_Attr_Input) as node_input:
                    dpg.add_text(input.__name__)
                    inputs[input.__name__] = node_input
                    
            
        outputs = {}
        
        if not isinstance(processor, Sink):
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
        

def init_node_editor(ctx:ProcessorContext):
    dpg.create_context()
    print('Building nodes')
    with dpg.window(label="Node editor", width=1000, height=400):
        with dpg.node_editor(width=-1, callback=link_callback):
            nodes = {}
            # Find all processors in context and build node for them
            for k,p in ctx.processors.items():
                nodes[p.proc_name] = make_node(p)
            
            # add in the connection
            for k, p in ctx.processors.items():
                node_outputs = list(nodes[p.proc_name][1].values())

                for k in p.out_queues.keys():
                    node_inputs = list(nodes[k][0].values())
                    
                    if node_outputs and node_inputs:
                        dpg.add_node_link(node_outputs[0], node_inputs[0])
                    

    
    dpg.create_viewport(title='Custom Title', width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    