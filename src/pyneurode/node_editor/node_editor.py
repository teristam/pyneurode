import dearpygui.dearpygui as dpg
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
import inspect
from pyneurode.processor_node.GUIProcessor import GUIProcessor
import igraph as ig
 
import logging

def make_node(processor:Processor) -> Tuple[Dict, Dict]:
    '''
    Parse the object information and make it into a node
    '''
    with dpg.node(label=processor.proc_name) as node:
        sig = inspect.signature(processor.__init__)
        # logging.debug(f'{processor.proc_name}: {sig}')

        input_cls, output_cls = processor.get_IOspecs()
        
        # logging.debug(f'{processor.proc_name}: {input_cls} {output_cls}')

        
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
            print(param,sig.parameters[param].annotation )
            with dpg.node_attribute(label=processor.proc_name, attribute_type=dpg.mvNode_Attr_Static): #static attribute
                annotation = sig.parameters[param].annotation
                if annotation is float:
                    dpg.add_input_float(label=param, width = 150)
                elif annotation is int:
                    dpg.add_input_int(label=param, width = 150)
                elif annotation is bool:
                    dpg.add_checkbox(label=param)
                elif annotation is str:
                    dpg.add_input_text(label=param, width=120)
                    
        return inputs, outputs,  node
                    
                    
def link_callback(sender,app_data):
    print(f'Connecting {app_data[0]} and {app_data[1]}')
    dpg.add_node_link(app_data[0], app_data[1], parent=sender)        
        

def init_node_editor(ctx:ProcessorContext):
    dpg.create_context()
    print('Building nodes')
    with dpg.window(label="Node editor", width=1000, height=400):
        with dpg.node_editor(width=-1, callback=link_callback):
            nodes = {}
            nodes_idx = {}
            idx = 0
            # Find all processors in context and build node for them
            for k,p in ctx.processors.items():
                if not isinstance(p,GUIProcessor): #avoid creating too many connections for GUI
                    nodes[p.proc_name] = make_node(p)
                    nodes_idx[p.proc_name] = idx
                    idx += 1
                    
            edges = [] # buld the node graph for layout later

            # add in the connection
            for k, p in ctx.processors.items(): #current processor
                try:
                    node_outputs = list(nodes[p.proc_name][1].values())

                    for k in p.out_queues.keys(): #name of the target processor
                        node_inputs = list(nodes[k][0].values())
                        
                        if node_outputs and node_inputs:
                            dpg.add_node_link(node_outputs[0], node_inputs[0])
                            edges.append([nodes_idx[k], nodes_idx[p.proc_name]])
                            
                except KeyError:
                    pass
                
            
            # Create the graph and generate the best laytout
            g = ig.Graph(edges, directed=True)
            layout = g.layout(layout='grid')
            #shift the pos of nodes so that they always start at 0,0
            layout = np.array(list(layout))
            layout -= layout.min(axis=0)
            layout += 0.2 # padding
            
            print(layout)
            # update the node position with the layout
            scale = 350
            for k, v in nodes.items():
                pos = layout[nodes_idx[k]]
                dpg.set_item_pos(v[2], pos*scale) #node object
                
                    

    
    dpg.create_viewport(title='Custom Title', width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    