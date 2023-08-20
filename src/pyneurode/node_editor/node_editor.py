from pathlib import Path
import warnings
import dearpygui.dearpygui as dpg
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
import inspect
from pyneurode.processor_node.GUIProcessor import GUIProcessor
import igraph as ig
import inspect
import logging
import importlib

from pyneurode.processor_node.Visualizer import Visualizer

class NodeManager():
    def __init__(self, context_manager, create_gui_processor=True) -> None:
        self.context:ProcessorContext = context_manager
        self.nodes = {} # a dictionary containing the tuple of (input, output, node), the key is the processor name 
        self.nodes_idx = {}  #index of the node for dearpygui
        self.node_import_path = {} # used to keep track of the fall import path of node
        self.node_editor = None
        self.gui_processor = None
        self.visualizers = {}
        
        if create_gui_processor:
            # Allow for visualizer support
            gui = GUIProcessor()
            self.gui_processor = gui
            self.context.register_processors(gui)
        
        self.init_node_editor(self.context)

        
        
    def make_node(self, processor:Union[Processor, Visualizer], node_editor) -> Tuple[Dict, Dict]:
        #Parse the object information and make it into a node
        
        if isinstance(processor, Processor):
            node_name = processor.proc_name
        else:
            node_name = processor.name
        
        
        with dpg.node(label=node_name, parent=node_editor) as node:
            sig = inspect.signature(processor.__init__)
            # logging.debug(f'{processor.proc_name}: {sig}')

            input_cls, output_cls = processor.get_IOspecs()
            
            # logging.debug(f'{processor.proc_name}: {input_cls} {output_cls}')

            
            # Add input and output nodes
            inputs = {}
            
            if not isinstance(processor, Source):
                for input in input_cls:
                    with dpg.node_attribute(label=node_name, attribute_type=dpg.mvNode_Attr_Input) as node_input:
                        dpg.add_text(input.__name__)
                        inputs[input.__name__] = node_input
                        
                
            outputs = {}
            
            if not isinstance(processor, Sink) and not isinstance(processor, Visualizer):
                for output in output_cls:
                    with dpg.node_attribute(label=node_name, attribute_type=dpg.mvNode_Attr_Output) as node_output:
                        dpg.add_text(output.__name__, indent=200)
                        outputs[output.__name__] = node_output
                    
                    
            
            for param in sig.parameters.keys():
                print(param,sig.parameters[param].annotation )
                with dpg.node_attribute(label=node_name, attribute_type=dpg.mvNode_Attr_Static): #static attribute
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
        
    
    def connect_visualizer(self, input_name, visualizer_name):
        # register the visualizer to the GUI processor
        if self.gui_processor is not None:
            # Connect the processor to the GUI
            source_proc = self.context.get_processor(input_name)
            # input.connect(self.gui_processor)
            
            # register the visualizer to the GUI
            visualizer = self.visualizers[visualizer_name]
            self.gui_processor.register_visualizer(source_proc, visualizer)
        
    
                        
    def link_callback(self, sender,app_data):
        attr1, attr2 = app_data
        print(f'Connecting {attr1} and {attr2}')
        dpg.add_node_link(attr1, attr2, parent=sender)    
        
        #Find the corresponding proccessor and connect them together
        proc1_name, proc1_type = self.find_processor_from_attr(attr1)
        proc2_name, proc2_type = self.find_processor_from_attr(attr2)


        # connect the processors together, taking care of the connection direction
        if proc1_type =='input':
            if 'Visualizer' in proc1_name:
                self.connect_visualizer(proc2_name, proc1_name)
            else:
                
                input = self.context.get_processor(proc1_name)
                output = self.context.get_processor(proc2_name)
                input.connect(output)
        else:
            if 'Visualizer' in proc2_name:
                self.connect_visualizer(proc1_name, proc2_name)
            else:
                output = self.context.get_processor(proc1_name)
                input = self.context.get_processor(proc2_name)
                input.connect(output)

    def delink_callback(self, sender, link):
        link_info = dpg.get_item_configuration(link)
        attr_1 = link_info['attr_1']
        attr_2 = link_info['attr_2']
        
        proc1_name, proc1_type = self.find_processor_from_attr(attr_1)
        proc2_name, proc2_type = self.find_processor_from_attr(attr_2)
        
        if proc1_type =='input':
            input = self.context.get_processor(proc1_name)
            output = self.context.get_processor(proc2_name)
            input.disconnect(output)
        else:
            output = self.context.get_processor(proc1_name)
            input = self.context.get_processor(proc2_name)
            input.disconnect(output)
            
        dpg.delete_item(link)
        
    def on_delete_keypress(self):
        for node in dpg.get_selected_nodes(self.node_editor):
            node_name = dpg.get_item_label(node)
            dpg.delete_item(node)
            self.context.remove_processors(node_name)
            
    
    def find_processor_from_attr(self, attr):
        # find the processor that has  that attribute
        for processor_name in self.nodes.keys():
            input, output, _ = self.nodes[processor_name]
            
            for k,v in input.items():
                if v==attr:
                    return processor_name, 'input'
            
            for k,v in output.items():
                if v ==attr:
                    return processor_name, 'output'

        
        raise ValueError(f'Cannot find the provided attributes {attr}')

    def play(self):
        self.context.start()
        
    def stop(self):
        self.context.stop()
       
        #TODO the following doesn't work, because it is going to destry to local window first 
        #close visualizer windows
        # for k, proc in self.context.processors.items():
        #     print(proc)
        #     if isinstance(proc, GUIProcessor):
        #         proc.shutdown()

    
    def get_available_processors(self):
        files = [f for f in os.listdir('src/pyneurode/processor_node') if f.endswith('.py')]
        class_names = set()
        for f in files:
            module_name = Path(f).stem
            module = importlib.import_module('pyneurode.processor_node.'+module_name)
            for obj_name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Processor):
                    class_names.add(obj_name)
                    self.node_import_path[obj_name] = f'pyneurode.processor_node.{module_name}'
        
        return class_names
    
    def get_available_visualizer(self):
        files = [f for f in os.listdir('src/pyneurode/processor_node') if f.endswith('.py')]
        class_names = set()
        for f in files:
            module_name = Path(f).stem
            module = importlib.import_module('pyneurode.processor_node.'+module_name)
            for obj_name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Visualizer):
                    class_names.add(obj_name)
                    self.node_import_path[obj_name] = f'pyneurode.processor_node.{module_name}'
        
        return class_names
    
    def add_node(self, sender, app_data, node_name):
        # create a processor and create its node interface
        node_module = importlib.import_module(self.node_import_path[node_name])
        NodeClass = getattr(node_module, node_name)
        node = NodeClass()
        self.context.register_processors(node)
        self.nodes[node.proc_name] = self.make_node(node, self.node_editor)
        
    def add_visualizer_node(self, sender, app_data, node_name):
        # create a visualizer and create its node interface
        node_module = importlib.import_module(self.node_import_path[node_name])
        NodeClass = getattr(node_module, node_name)
        node = NodeClass()
        self.nodes[node.name] = self.make_node(node, self.node_editor)
        self.visualizers[node.name] = node

     
    def build_nodes_tree(self):
        # build the tree nodes showing all available nodes
        nodes_name = self.get_available_processors()
        nodes2remove = set()
        with dpg.tree_node(label = 'Source', default_open=True):
            for name in nodes_name:
                if name.endswith('Source'):
                    with dpg.group():
                        dpg.add_button(label = name, callback=self.add_node, user_data=name)
                        nodes2remove.add(name)
                        
        nodes_name = nodes_name - nodes2remove
        nodes2remove.clear()
        
        with dpg.tree_node(label = 'Sink', default_open=True):
            for name in nodes_name:
                if name.endswith('Sink'):
                    with dpg.group():
                        dpg.add_button(label = name, callback=self.add_node,  user_data=name)
                        nodes2remove.add(name)
                        
        nodes_name = nodes_name - nodes2remove
        nodes2remove.clear()

        with dpg.tree_node(label = 'Transformer', default_open=True):
            for name in nodes_name:
                with dpg.group():
                    dpg.add_button(label = name, callback=self.add_node,  user_data=name)
                    
        with dpg.tree_node(label = 'Visualizer', default_open=True):
            nodes_name = self.get_available_visualizer()
            for name in nodes_name:
                with dpg.group():
                    dpg.add_button(label = name, callback=self.add_visualizer_node,  user_data=name)
                    
        
    def init_node_editor(self, ctx:ProcessorContext):

        dpg.create_context()
        dpg.configure_app(docking=True, docking_space=True, init_file='node_editor.ini')
        print('Building nodes')
        
        with dpg.window(label='Nodes', width=400, height=-1):
            self.build_nodes_tree()
                    
        with dpg.window(label="Node editor", width=1200, height=1200):
            with dpg.group(horizontal=True):
                dpg.add_button(label='Play', callback=self.play)
                dpg.add_button(label='Stop', callback=self.stop)
                
            with dpg.node_editor(width=-1, height=-1, callback=self.link_callback, delink_callback=self.delink_callback) as self.node_editor:

                idx = 0
                # Find all processors in context and build node for them
                for k,p in ctx.processors.items():
                    if not isinstance(p,GUIProcessor): #avoid creating too many connections for GUI
                        self.nodes[p.proc_name] = self.make_node(p, self.node_editor)
                        self.nodes_idx[p.proc_name] = idx
                        idx += 1
                    else:
                        # if it is a GUI processor, build nodes for its visualizer instead
                        for src_proc, viss in p.source_visualizer_map.items():
                            for v in viss:
                                self.nodes[v.name] = self.make_node(v, self.node_editor)
                                self.nodes_idx[v.name] = idx
                                idx += 1

                        
                edges = [] # buld the node graph for layout later

                # add in the connection
                for k, p in ctx.processors.items(): #current processor
                    print('processor: ', p)
                    if not isinstance(p, GUIProcessor):
                        # Connect output of current proc to the input of the next proc
                        node_outputs = list(self.nodes[p.proc_name][1].values()) 

                        for k in p.out_queues.keys(): #name of the target processor
                            if not 'GUIProcessor' in k:
                                # Normal nodes
                                node_inputs = list(self.nodes[k][0].values())
                                
                                if node_outputs and node_inputs:
                                    dpg.add_node_link(node_outputs[0], node_inputs[0])
                                    edges.append([self.nodes_idx[k], self.nodes_idx[p.proc_name]])      
                            else:
                                # Output is GUI processor , need special handling
                                gui_proc = ctx.get_processor(k)
                                viss = gui_proc.source_visualizer_map[p.proc_name]
                                for v in viss:
                                    node_inputs = list(self.nodes[v.name][0].values())
                                    if node_outputs and node_inputs:
                                        dpg.add_node_link(node_outputs[0], node_inputs[0])
                                        edges.append([self.nodes_idx[p.proc_name], self.nodes_idx[v.name]]) 
                                        
                                                          
            
                    
                
                # Create the graph and generate the best layout
                if len(edges)>0:
                    g = ig.Graph(edges, directed=True)
                    layout = g.layout(layout='grid')
                    #shift the pos of nodes so that they always start at 0,0
                    layout = np.array(list(layout))
                    layout -= layout.min(axis=0)
                    layout += 0.2 # padding
                    
                    # print(layout)
                    # update the node position with the layout
                    scale = 350
                    for k, v in self.nodes.items():
                        pos = layout[self.nodes_idx[k]]
                        dpg.set_item_pos(v[2], pos*scale) #node object
                else:
                    warnings.warn('Warning: no node edge can be found')
                    
                    
        # register global event
        with dpg.handler_registry():
            dpg.add_key_press_handler(key=dpg.mvKey_Delete, callback=self.on_delete_keypress)
        
        dpg.create_viewport(title='Custom Title', width=800, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
        