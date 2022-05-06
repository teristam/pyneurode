from __future__ import annotations # for postponed evaluation
import multiprocessing as mp
from pyneurode.processor_node.Processor import *

import logging
import tempfile
import os
import time
import functools


logging.basicConfig(level=logging.DEBUG)

def process_wrapper(name, proc_class, shutdown_event, *args, **kwargs):
    proc = proc_class(name, shutdown_event, *args, **kwargs)
    proc.run()

def process_run_wrapper(proc):
    proc.run()

class ProcessorContext:
    # a class for managing all the subprocesses
    # it should also manage the various queues of the subprocess to connect them to each other
    
    def __init__(self):
        self.log = functools.partial(logger, 'ProcessorContext')
        self.shutdown_event = mp.Event()
        self.processors:list[Processor] = {}
        self.links = []
        self.subprocs = []

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, exc_tb):
        # clean up
        pass

    def register_processors(self, *args:Processor):
        # register and initialize the processor
        
        for p in args:
            p.init_context(self)
            self.processors[p.proc_name] = p

    def start(self):
        # spwan the subprocess and start the processing
        for k, proc in self.processors.items():

            self.log(logging.DEBUG, f'Spawning process for {k}')
            p = mp.Process(target = process_run_wrapper, args=(proc,), name = proc.proc_name )
            p.start()
            self.subprocs.append(p)




if __name__ == '__main__':

    start  = time.time()
    echoFile =  FileEchoSink.get_temp_filename()

    print(echoFile)
    # tempfile.TemporarayFile cannot be pickled and sent over to other processes

    with ProcessorContext() as ctx:
        
        source = DummyTimeSource(0.1)
        add = AddProcessor()
        echo1 = FileEchoSink(filename=echoFile)
        echo2 = EchoProcessor()

        source.connect(add)
        add.connect(echo1)
        add.connect(echo2)

        ctx.register_processors(source, add, echo1, echo2)

        ctx.start()

        time.sleep(3)
        logging.debug('Sending the stop signal now')
        # send stop signal after 5s
        ctx.shutdown_event.set()

    time.sleep(1)    
    print(FileEchoSink.open_data_dump(echoFile))


# %%
