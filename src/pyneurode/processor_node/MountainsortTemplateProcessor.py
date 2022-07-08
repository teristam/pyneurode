import logging
import pickle
import random
import threading
import time
from multiprocessing import Event

import numpy as np
from pyneurode.processor_node.BatchProcessor import *
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from pyneurode.spike_sorter import *
from pyneurode.spike_sorter_cy import template_match_all_electrodes_cy
from pyneurode.processor_node.Message import Message, MetricsMessage

import pyneurode.utils
import warnings

def collect_spikes(spike_queue:Queue,spike_data):
    print('Collect spikes starting...')
    # collect the spikes from message passing and store it locally
    while True:
        try:
            spike_data.append(spike_queue.get(block=True,timeout=0.001))
        except Empty:
            continue

class SpikeTemplateMessage(Message):
    type = 'spike_template_message'
    def __init__(self, templates, template_cluster_id, template_electrode_id,pca_transformers=None, standard_scalers=None):
        self.data = {
            'templates':templates,
            'template_cluster_id': template_cluster_id,
            'template_electrode_id': template_electrode_id,
            'pca_transformer': pca_transformers,
            'standard_scalers': standard_scalers
        }
        
class RecomputeTemplateControlMessage(Message):
    type = 'recompute_template'
    def __init__(self):
        super().__init__(self.type, None, None)

class MountainsortTemplateProcessor(BatchProcessor):
    """Create spikes template using isosplit from mountainsort

    """

    def startup(self):
        super().startup()

        # Spawn another thread for receiving the spikes
        self.spike_data = []
        self.spike_print_counter = 0
        self.spike_len_prev = 0
        self.template = None
        self.spike_print_counter = 0
        self.pca_component = 3
        self.last_sort_time = None # time since last template computation

        # self.collect_spike_thread = threading.Thread(target=collect_spikes, args=(self.in_queue,self.spike_data))
        # self.collect_spike_thread.start()

    # @profile_cProfile()
    def run(self):
        return super().run()

    def __init__(self, interval=None, internal_buffer_size=1000, min_num_spikes=2000, do_pca=True, training_period=None):
        super().__init__(interval=interval, internal_buffer_size=internal_buffer_size)
        self.MIN_NUM_SPIKE = min_num_spikes
        self.do_pca = do_pca
        self.training_period = training_period # the period at which the spike template will be recompute, in second
        self.prev_metrics_time = time.time()
 
    def process(self, msgs):
        # each time, it will load multiple message
        # each message may  contain multiple spikesEvent
        
        data = []
        need_sort = False

        
        for m in msgs:
            if type(m) is RecomputeTemplateControlMessage:
                self.log(logging.DEBUG, f'Received re-sort command.')
                need_sort = True
            elif m.type == 'spike':
                data.append(m.data)
        
        self.spike_data = self.spike_data + data # TODO: possible performance regresssion here, as the list is growing every loop


        if self.spike_print_counter != len(self.spike_data)//500:
            self.spike_print_counter = len(self.spike_data)//500
            self.log(logging.DEBUG, f'spike length {len(self.spike_data)}')

        # Cluster to find templates
        if self.last_sort_time is None:
            if (len(self.spike_data)>(self.spike_len_prev+self.MIN_NUM_SPIKE)):
                need_sort = True
        else:
            if (self.training_period is not None) and (time.time()-self.last_sort_time)>self.training_period:
                need_sort = True
        
        if need_sort:
            start_time = time.time()

            self.log(logging.DEBUG, 'Computing spike template')
            spike2sort = self.spike_data.copy() 

            df = makeSpikeDataframe(spike2sort)


            df, pca_transformers,standard_scalers = sort_all_electrodes(df, pca_component=3) #sort spikes

            (templates, template_cluster_id, template_electrode_id) = generate_spike_templates(df)
            print(template_cluster_id)

            # Some book keeping
            self.spike_len_prev = len(self.spike_data) # update previous spike data len
            print(f'Update template takes {time.time()-start_time:.3f}s')
            
            self.last_sort_time = time.time()
            
            ## TODO: meausre the clustering metrics
            
            msg = SpikeTemplateMessage(
                templates,
                template_cluster_id,
                template_electrode_id,
                pca_transformers,
                standard_scalers
            )
            
            return msg
        else:
            if (time.time()-self.prev_metrics_time) >=1:
                # send update every second
                metrics_msg = MetricsMessage(self.proc_name, 
                                             {
                                            "spike_size": len(self.spike_data)
                                            } )
                
                self.prev_metrics_time = time.time()
                
                return metrics_msg
                

            