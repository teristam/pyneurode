from __future__ import annotations  # for postponed evaluation

import functools
import logging
import threading

from pyneurode.processor_node.Message import Message

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import cProfile
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from functools import wraps
from multiprocessing import *
from queue import Empty
from typing import *

import numpy as np
import pyneurode.utils as utils
import shortuuid
from pyneurode.processor_node.Context import Context

# import zmq

start_time = time.monotonic()

def logger(name, level, msg, exc_info=None):
    elapsed = time.monotonic() - start_time
    minute = int(elapsed//60)
    seconds = (elapsed - minute*60)

    myLogger = logging.getLogger('zmq_decode')

    myLogger.log(level, f'\t{minute}:{seconds:.2f} {name:15} \t\t {msg}', exc_info=exc_info)




class Channel:
    '''
    A class that manage the connection between two Processors
    It will only allows  message types that match with the filter to pass through
    '''
    def __init__(self, queue:Queue, filters:Union[List[str],List[Message]]=None):
        self.queue = queue
        self.filters = filters
        self.filters = filters

    def send(self, message:Message):
        '''
        Send the data. Only data type matching the filter will be put into the message queue
        '''

        if self.filters is not None:
            if message.type in self.filters:
                # the timeout is necessary to avoid the other process shutting down first
                # and block the current process
                self.queue.put(message)
        else:
            self.queue.put(message)

    def recv(self, timeout = 0.1):
        '''
        Get data from the queue
        '''
        try:
            return self.queue.get(block=True, timeout=timeout)
        except TimeoutError:
            return None

class Processor(Context):
    """The base class that handle real-time processing

    """
    QUEUE_PULL_TIMEOUT=0.01
    # base class for all processor
    # when start, create its own process, open the port for sending and receiving data
    # should hide the underlying protocol used, so it is possible to use zmq and multiprocessing.Queue
    
    def __init__(self):
        self.out_queues: Dict[str, Channel] = {}
        self.proc_name = self.__class__.__name__ +'_'+ shortuuid.ShortUUID().random(5) #unique identying string for the processor
        self.log = functools.partial(logger, self.proc_name)
        self.shutdown_event = None
        self.in_queue = Queue()
        self.log(logging.DEBUG, 'Initializing')
        self.latency = None #processing latency
        self.run_count = 0
        
        # regsiter itself to the current context
        try:
            context = type(self).get_context() # have to class it from class
            self.log(logging.DEBUG, f'Context found. I am registering myself to {context}')
            self.get_context().register_processors(self)
        except TypeError:
            self.log(logging.DEBUG, 'No context found. Have to register manually.')

    def init_context(self, ctx):
        self.shutdown_event = ctx.shutdown_event

    def startup(self):
        """ Functions to be first called before processing run. Can be overrided for customized startup behaviour
        """
        self.log(logging.DEBUG, 'Starting up')

    def shutdown(self):
        """ Close the queue and flush all the messages
        """
        self.log(logging.DEBUG, 'Shut down')
        self.in_queue.close()
        self.in_queue.join_thread() #flush the all message to the pipe immediately so that it can be shut down properly

    def run(self):
        assert self.shutdown_event is not None, 'The processor context has not been initiated'
        # the actual data processing goes here
        # polling, TODO: can be for socket or from queue
        try:
            self.startup()
            self.main_loop()
            self.log(logging.INFO, "Shutting down normally")
            self.shutdown()
        except Exception as e:
            self.log(logging.ERROR, f'Exception caught {e}', exc_info = True)
        
    def connect(self, to_processor:Processor, filters=None):
        self.log(logging.DEBUG, f'Connecting {self.proc_name} and {to_processor.proc_name}')
        self.out_queues[to_processor.proc_name] = Channel(to_processor.in_queue, filters)

    def send(self, data:Union[Message, List[Message]], output_name:str=None):
        # output the data to downstream queue specified by the output_name (if specified)
        # if no output_name is specified, then output to all
        # self.log(logging.DEBUG, f'Sending {data}')

        if data is not None and not isinstance(data,Message) and not type(data) in [list, tuple]:
            raise TypeError(f'Data to be sent must be wrapped as a Message object: {data}')

        if data is not None:
            if not type(data) in [list,tuple]:
                data = [data]

            for d in data:
                if output_name is not None:
                    self.out_queues[output_name].send(d)
                    # self.log(logging.DEBUG, f'{d} send to {output_name}')
                else:
                    for q in self.out_queues.keys():
                        self.out_queues[q].send(d)

    def send_metrics(self):
        # also send out the latency every 100 run
        if self.run_count % 10 == 0:
            msg = Message(
                    "metrics",
                    {
                        "processor": self.proc_name,
                        "measures": {
                            "latency": self.latency*1000, #in ms
                        },
                    },
                )
            self.send(msg)
        
    def main_loop(self):
        while not self.shutdown_event.is_set():
            try:
                data = self.in_queue.get(block=True, timeout=self.QUEUE_PULL_TIMEOUT)
                if self.in_queue.qsize() > 1000:
                    self.log(logging.WARNING,'in-queue size is larger than 1000. Error may occur')
                processed_data = self._process(data)
                if isinstance(processed_data,list):
                    for d in processed_data:
                        self.send(d)
                else:
                    self.send(processed_data)
            except Empty: #if timeout
                continue
    
    def get_IOspecs(self) -> Tuple[List[Message], List[Message]]:
        """return a tuple ([input message], [output message]) with the Message type that this processor is expected to
        receive and produce respectively
        """
        
        return (
            [Message],
            [Message]
        )

    def _process(self, message=None):
        # also monitor the processing time
        start = time.perf_counter()

        msg =  self.process(message)

        lapsed = time.perf_counter() - start
        if self.latency is None:
            self.latency = lapsed
        else:
            self.latency = (self.latency + lapsed)/2

        self.run_count += 1
        self.send_metrics()

        return msg

    def process(self, message:Message) -> Message:
        """The function to be override by subclass to perform realtime processing.
        The overriden function should work on the passed-in message and return the results as another Message

        Args:
            message (Message): Data to be processed.

        Raises:
            NotImplementedError: This function must be implemented in subclasses

        Returns:
            Message: processed messages
        """
        # override this function in sub-class to implement specific functions
        raise NotImplementedError(f'{self.__class__.__name__}.process is not implemented')



class Sink(Processor):
    '''
    Sink: a processor that doesn't send any output
    '''
    def main_loop(self):
        while not self.shutdown_event.is_set():
            try:
                data = self.in_queue.get(block=True, timeout=self.QUEUE_PULL_TIMEOUT)
                self._process(data)
            except Empty: #if timeout
                continue    

    def process(self,message:Message):
        raise NotImplementedError(f'{self.__class__.__name__}.process is not implemented')


class Source(Processor):
    '''
    Source: a processor that doesn't depend on input
    '''
    def main_loop(self):
        while not self.shutdown_event.is_set():
            try:
                processed_data = self._process()
                if isinstance(processed_data, list):
                    for d in processed_data:
                        self.send(d)
                else:
                    self.send(processed_data)
            except Empty: #if timeout
                continue

    def _process(self):
        return self.process()

    def process(self):
        raise NotImplementedError(f'{self.__class__.__name__}.process is not implemented')


class TimeSource(Source):
    '''
    TimeSource: a Source that trigger an event in a given interval
    '''
    def __init__(self, interval:float):
        super().__init__()
        self.interval = interval
        self.end_time = None

    def _process(self):

        if self.interval>0:
            if self.end_time is None:
                self.end_time  = time.monotonic() + self.interval
        
            # Wait till the future trigger point
            time2wait = self.end_time - time.monotonic()

            if time2wait>0:
                time.sleep(time2wait)
            else:
                self.log(logging.DEBUG, 'Not able to keep with the specified speed.')

            # note: the sleep() function may not be entirely accurate, because it may skip some ms
            # but in general it should be good enough, so for simplicity we skip checking the time again
            self.end_time = time.monotonic() + self.interval
            return self.process()
            
        else:
            return self.process()

    def process(self):
        raise NotImplementedError(f'{self.__class__.__name__}.process is not implemented')

class SineTimeSource(TimeSource):
    '''
    A TimeSource that generate multi-channel sine wave for debugging purpose
    '''
    def __init__(self, interval:float, frequency:float, channel_num:int, sampling_frequency:int=100, noise_std:float=0):
        super().__init__(interval)
        self.frequency = frequency #in Hz
        self.start_time = time.time()
        self.channel_num = int(channel_num)
        self.last_time = None #last time the data are plotted
        self.dt = 1/sampling_frequency
        self.noise_std = noise_std

    def process(self):
        # Generate a sine waveform
        now = time.time() - self.start_time

        if self.last_time is None:
            t = np.arange(0, now, step=self.dt)
        else:
            # t = np.linspace(self.last_time, now, int((now-self.last_time)*self.Fs))
            t = np.arange(self.last_time, now, step=self.dt)

        if len(t)> 0:
            # print(t)
            self.last_time = t[-1]+self.dt
            y = np.sin(2*np.pi*self.frequency*t)
            y = np.tile(y, (self.channel_num,1)).T
            
            if self.noise_std > 0:
                # add in noise
                y += np.random.normal(scale=self.noise_std, size=y.shape)
            # print(y.shape)
            return Message('sine', y)

class DummyTimeSource(TimeSource):
    def startup(self):
        self.msg = 0 

    def process(self):
        self.msg += 1
        self.log(logging.DEBUG, f'Producing msg {self.msg}')
        return Message('dummy',self.msg)

class MsgTimeSource(TimeSource):
    # Reply the supplied message in sequence
    def __init__(self, interval:float, messages:List[Message]):
        """Construct a Time source that send message sequentially at fixed interval

        Args:
            interval (float): interval in second
            messages (List[Message]): list of message to send
        """
        super().__init__(interval)
        self.msg_list = messages
        self.msg_counter = 0

    def process(self):
        msg =  self.msg_list[self.msg_counter%len(self.msg_list)]
        # self.log(logging.DEBUG, f'Producing msg {msg}')
        self.msg_counter += 1
        return msg
    
    
class AddProcessor(Processor):
    def process(self, msg):
        return Message('dummy',msg.data + 100)
    
class ScaleProcessor(Processor):
    def __init__(self, a:float, b:float):
        super().__init__()
        self.a = a
        self.b = b
        
    def process(self, message: Message) -> Message:
        x = message.data*self.a+self.b
        return Message(message.type, x)

class EchoSink(Sink):
    """ Simply print the message it receives
    """
    def process(self,data):
        self.log(logging.INFO, f'Printing {data}')
        
        
class DummpyNumpyTimeSource(TimeSource):
    '''
    Generate random numpy array of a particular shape
    '''
    def __init__(self, proc_name: str, shutdown_event: Event, interval:float, shape:Tuple[int,int]):
        super().__init__(proc_name, shutdown_event, interval)
        self.shape = shape
    
    def process(self):
        msg = Message('data', np.random.rand(self.shape[0], self.shape[1]))
        self.log(logging.DEBUG, f'Processing msg {msg}')
        return msg



class FileEchoSource(TimeSource):
    '''
    Read message back saved by FileEchoSink. It assume each pick.read() will return a message object
    '''

    def __init__(self, interval:float, filename, filetype:str, verbose=False, batch_size = 10):
        '''
        filetype can ether be 'message','list', or 'simple'
        '''
        super().__init__(interval)
        self.filename = filename
        self.filetype = filetype
        self.data = None
        self.data_counter = 0
        self.verbose = verbose
        self.last_msg = None
        self.batch_size = batch_size

    def startup(self):
        super().startup()
        self.datafile = open(self.filename, 'rb')
        self.log(logging.DEBUG, 'Loading data')

        if self.filetype == 'list':
            self.data = pickle.load(self.datafile)
        elif self.filetype == 'message':
            self.data = []
            while True:
                try:
                    msg = pickle.load(self.datafile)
                    if type(msg) is list:
                        if len(msg)>0:
                            self.data = self.data + msg
                    elif type(msg) is Message:
                        self.data.append(msg)
                    else:
                        raise TypeError(f'Datatype {type(msg)} is not supported')
                except EOFError:
                    break
        else:
            raise ValueError("The filetype is not supported")
                
        self.log(logging.DEBUG, f'Finishing loading data. Totaly message count: {len(self.data)} ')

    def process(self):

        if self.data_counter>0 and self.data_counter % 10000 == 0:
            self.log(logging.DEBUG, f'Finish sending {self.data_counter} messages')

        if self.filetype == 'message':
            # return a batch of messages at once for performance
            if self.data_counter + self.batch_size > len(self.data):
                self.data_counter = 0 # restart

            msg = self.data[self.data_counter:self.data_counter + self.batch_size]
            self.data_counter += self.batch_size
            
            # modify the acquire_time so that we can measure the latency
            for m in msg:
                if m.type == 'spike':
                    m.data.acq_timestamp = utils.perf_counter()

            if self.verbose:
                self.log(logging.DEBUG, msg)
            return msg
        elif self.filetype == 'list':
            assert self.data is not None, 'Not able to load data'
            msg = Message('data', self.data[self.data_counter])
            if self.data_counter < len(self.data):
                self.data_counter += 1
            return msg
        else:
            msg = pickle.load(self.datafile)
            return Message('data', msg)        

class FileEchoSink(Sink):
    '''
    Dump the queue to a file and also echo any messages recevied
    '''
    def __init__(self, filename:str, verbose = False):
        super().__init__()
        self.filename = filename
        self.verbose = verbose

    def startup(self):
        super().startup()
        self.file = open(self.filename,'wb')

    def shutdown(self):
        super().shutdown()
        self.file.close()

    def process(self, message: Message):
        if self.verbose:
            self.log(logging.INFO, f'Dumping {message}')
        pickle.dump(message,self.file)


    @staticmethod
    def open_data_dump(filename):
        data = []
        with open(filename,'rb') as f:
            while True:
                try:
                    msg = pickle.load(f)
                    data.append(msg)
                except EOFError:
                    break
        
        return data

    @staticmethod
    def get_temp_filename():
        return os.path.join(tempfile.gettempdir(), os.urandom(10).hex()) #generate a random file


def profile_pyinstrument(func):
    def wrapper(*args, **kwargs):
        profiler = Profiler(interval=0.0001)
        profiler.start()
        func(*args, **kwargs)
        profiler.stop()
        profiler.open_in_browser()

    return wrapper

def profile_cProfile(output_file=None):
    def decorator_wrapper(func):
        @wraps(func) # preserve information about original function
        def wrapper(*args, **kwargs):
            _output_file = output_file or f'{func.__name__ }.prof'
            print(f'profiler output file {_output_file}')
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                func(*args, **kwargs)
            except:
                pass
            finally:
                profiler.disable()
                profiler.dump_stats(_output_file)
                print(f'Written profiler data to {_output_file}')

        
        return wrapper

    return decorator_wrapper
