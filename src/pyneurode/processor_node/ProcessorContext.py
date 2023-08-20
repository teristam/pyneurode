from __future__ import annotations
from decimal import getcontext # for postponed evaluation
import multiprocessing as mp
from pyneurode.processor_node.Processor import *

import logging
import tempfile
import os
import time
import functools
import threading

from pyneurode.processor_node.Visualizer import Visualizer
from pyneurode.processor_node.Context import Context

logging.basicConfig(level=logging.DEBUG)

def process_wrapper(name, proc_class, shutdown_event, *args, **kwargs):
    proc = proc_class(name, shutdown_event, *args, **kwargs)
    proc.run()

def process_run_wrapper(proc):
    proc.run()
    

class ProcessorContext(Context):
    # a class for managing all the subprocesses
    # it should also manage the various queues of the subprocess to connect them to each other
        
    def __init__(self, auto_start=True):
        self.log = functools.partial(logger, 'ProcessorContext')
        self.shutdown_event = mp.Event()
        self.processors:dict = {} #dictionary of the processors managed by the context
        self.links = []
        self.subprocs = []
        self.auto_start = True

    def register_processors(self, *args:Processor):
        # register and initialize the processor
        
        for p in args:
            p.init_context(self)
            self.processors[p.proc_name] = p
            
    def remove_processors(self, *args:Processor):
        for p in args:
            self.processors.pop(p)
            self.log(logging.DEBUG, f'Removed {p} from context')

    def start(self):
        self.shutdown_event.clear()

        # spwan the subprocess and start the processing
        for k, proc in self.processors.items():

            self.log(logging.DEBUG, f'Spawning process for {k}')
            p = mp.Process(target = process_run_wrapper, args=(proc,), name = proc.proc_name )
            p.start()
            self.subprocs.append(p)
            
    def stop(self):
        self.shutdown_event.set()

            
    def get_processor(self, name) -> Processor:
        #return a processor
        return self.processors[name]
    
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        super().__exit__(exc_type, exc_value, exc_tb)
        if self.auto_start:
            self.start()




if __name__ == '__main__':

    start  = time.time()
    echoFile =  FileEchoSink.get_temp_filename()

    print(echoFile)
    # tempfile.TemporarayFile cannot be pickled and sent over to other processes

    with ProcessorContext() as ctx:
        print(Context.get_contexts())
        source = DummyTimeSource(0.1)
        add = AddProcessor()
        echo1 = FileEchoSink(filename=echoFile, verbose=True)
        echo2 = EchoSink()

        source.connect(add)
        add.connect(echo1)
        add.connect(echo2)

        # ctx.register_processors(source, add, echo1, echo2)

        ctx.start()

        time.sleep(10)
        logging.debug('Sending the stop signal now')
        
        # send stop signal after 5s
        ctx.shutdown_event.set()

    time.sleep(2)    
    print(FileEchoSink.open_data_dump(echoFile))


# %%
