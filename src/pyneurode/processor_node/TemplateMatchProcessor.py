import logging
import pickle
import random
import threading
import time
from multiprocessing import Event

import numpy as np
from pyneurode.processor_node.BatchProcessor import *
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.MountainsortTemplateProcessor import MountainsortTemplateProcessor, SpikeTemplateMessage
from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from pyneurode.spike_sorter import *
from pyneurode.spike_sorter_cy import template_match_all_electrodes_cy
from pyneurode.processor_node.Message import Message, MetricsMessage

import pyneurode.utils
import warnings


class TemplateMatchProcessor(BatchProcessor):

    def startup(self):
        super().startup()

        # Spawn another thread for receiving the spikes
        self.spike_data = []
        self.spike_print_counter = 0
        self.spike_len_prev = 0
        self.pca_transformer = None
        self.spike_templates = None
        self.templates = None
        self.template_cluster_id = None
        self.template = None
        self.prev_sort_time = None
        self.df_sort_residue = None
        self.time_bin_start = 0
        self.Fs = 30000
        self.start_timestamp  = 0 # TODO: need to record this somewhere
        self.df_sort_list = []
        self.buffer_is_valid = False
        self.spike_train_buffer = None
        self.spiketrain_buffer_length = 10000
        self.spike_print_counter = 0
        self.sorted_idx_prev = 0
        self.template_electrode_id = None
        self.df_sort = None
        self.pca_component = 3

        # self.collect_spike_thread = threading.Thread(target=collect_spikes, args=(self.in_queue,self.spike_data))
        # self.collect_spike_thread.start()

    # @profile_cProfile()
    def run(self):
        return super().run()

    def __init__(self, interval=None, internal_buffer_size=1000, min_num_spikes=2000, do_pca=True, time_bin=0.1):
        super().__init__(interval=interval, internal_buffer_size=internal_buffer_size)
        self.do_pca = do_pca
        self.time_bin = time_bin  #TODO: need to have a setting file to synchronize this time bin between spike and ADC data
 
    def process(self, msgs):
        # each time, it will load multiple message
        # each message may  contain multiple spikesEvent
        
        data = []
        # check if there is a new template coming in
        for m in msgs:
            if type(m) is SpikeTemplateMessage:
                self.log(logging.INFO, "Updating templates")
                self.templates = m.data['templates']
                self.template_cluster_id = m.data['template_cluster_id']
                self.template_electrode_id = m.data['template_electrode_id']
                self.pca_transformers = m.data['pca_transformer']
                self.standard_scalers = m.data['standard_scalers']
                self.buffer_is_valid = False # refresh the buffer
            elif m.type == 'spike':
                data.append(m.data)

        self.spike_data = self.spike_data + data
            
        # Template matching
        if self.templates is not None and len(self.spike_data)>0:
            
            start_time = utils.perf_counter()
            # do template matching
            self.df_sort = makeSpikeDataframe(self.spike_data)
            self.spike_data.clear()


            #TODO: selectively do PCA on some of the channels only
            if self.do_pca:
                self.df_sort = template_match_all_electrodes_fast(self.df_sort, self.templates, 
                        self.template_electrode_id, self.template_cluster_id, self.pca_transformers, self.standard_scalers)
            else:
                self.df_sort = template_match_all_electrodes_cy(self.df_sort, self.templates, 
                        self.template_electrode_id, self.template_cluster_id)
                


            self.sorted_idx_prev += len(self.df_sort)

            # Convert the spike time to time-binned array
            self.df_sort['spike_time'] = (self.df_sort.timestamps.values-self.start_timestamp)/self.Fs #shift time to be relative to start

            #carry over the previous residue
            if self.df_sort_residue is not None and len(self.df_sort_residue)>0:
                self.df_sort = pd.concat([self.df_sort_residue, self.df_sort])
            
            bins = np.arange(self.time_bin_start, self.df_sort.spike_time.max(), self.time_bin )

            # IMPORTANT: leave the spike beyound the bin edge end point to the next cycle
            # otherwise those spike won't be counted
            self.df_sort_residue = self.df_sort[self.df_sort.spike_time>=bins[-1]] # spike is beyond the bin edge, left till next cycle
            self.df_sort = self.df_sort[self.df_sort.spike_time<bins[-1]]
                        
            spk_train, skp_time_event=sort2spiketrain(self.template_cluster_id, self.df_sort.cluster_id, self.df_sort.spike_time, bins)
            total_spikes = np.sum(spk_train[:])
            # print(df_sort)
            if not total_spikes == len(self.df_sort):
                warnings.warn(f'Some spikes are omitted spike_train {total_spikes} df_sort {len(self.df_sort)}') #make sure all spikes are included

            self.time_bin_start = bins[-1] #update the start of the bin

            self.df_sort['sort_timestamps'] = utils.perf_counter() #add in the sorting finish time

            lapsed_time = utils.perf_counter()- start_time

            # Save the sorted spikes to buffer
            if self.buffer_is_valid:
                # TODO: use a event process specifically designed for that
                # self.log(logging.DEBUG, 'writing to buffer')
                self.spike_train_buffer.write(spk_train.T)
            else:
                # initialize the spike train buffer
                self.spike_train_buffer = RingBuffer((self.spiketrain_buffer_length, len(self.template_cluster_id)), dt=self.time_bin)
                self.spike_train_buffer.is_valid = True
                self.spike_train_buffer.write(spk_train.T)
                self.buffer_is_valid = True
                self.neuron_num = len(self.template_cluster_id)


            # self.log(logging.DEBUG, f'Sending {spk_train.shape}')

            # also calculate the time to sort each spikes in ms
            if len(self.df_sort)>0:

                metrics_msg = MetricsMessage(self.proc_name, 
                                             {
                                            "time_per_spike": lapsed_time*1000/len(self.df_sort), #in ms
                                            'in_queue_size': self.in_queue.qsize(),
                                            'df_sort_size': len(self.df_sort)
                                            } )
                
                return (SpikeTrainMessage(spk_train), SortedSpikeMessage(self.df_sort), metrics_msg)
            else:
                (SpikeTrainMessage(spk_train), SortedSpikeMessage(self.df_sort))


class SpikeTrainMessage(Message):
    """Message that contains the binned spike train
    spiketrain is in the format (neuron x time)
    """
    type = 'spike_train'

    def __init__(self, spk_train:np.ndarray):
        
        if isinstance(spk_train, np.ndarray):
            self.data = spk_train
        else:
            raise TypeError("The input should be a numpy array")
        
        self.timestamp = time.time()

class SortedSpikeMessage(Message):
    type = 'df_sort'
    
    def __init__(self, sorted_dataframe:pd.DataFrame):
        
        if isinstance(sorted_dataframe, pd.DataFrame):
            self.data = sorted_dataframe
        else:
            raise TypeError('The input should be a pandas dataframe')
        
        self.timestamp = time.time()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    with ProcessorContext() as ctx:
        zmqSource = FileEchoSource(interval=0.01, filename='E:/decoder_test_data/M10_2022-06-23_12-43-50_test4/M10_test4_20220623_124341_packets.pkl', 
                                filetype='message',batch_size=3)
        templateTrainProcessor = MountainsortTemplateProcessor(interval=0.01,min_num_spikes=500,time_bin=0.01)
        templateMatchProcessor = TemplateMatchProcessor(interval=0.01)
        echoProcessor = EchoProcessor()

        zmqSource.connect(templateTrainProcessor, filters='spike')
        zmqSource.connect(templateMatchProcessor, filters='spike')
        templateTrainProcessor.connect(templateMatchProcessor)
        # templateTrainProcessor.connect(echoProcessor)

        ctx.register_processors(zmqSource,templateTrainProcessor, templateMatchProcessor,echoProcessor)

        ctx.start()


