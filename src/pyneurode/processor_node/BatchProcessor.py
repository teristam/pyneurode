from __future__ import annotations # for postponed evaluation
import threading
from collections import deque
from pyneurode.processor_node.Processor import *
from pyneurode import RingBuffer
from pyneurode.RingBuffer.RingBufferG import RingBufferG
from pyneurode.processor_node.ProcessorContext import ProcessorContext
import time
from pyneurode.processor_node.Message import Message
from typing import List

def acquire_data(queue:Queue, buffer:RingBufferG, shutdown_event:Event, pull_time_out, need_pause:threading.Event, queue_max):
    while not shutdown_event.is_set():
        try:
            data = queue.get(block=True, timeout=pull_time_out)
            if data is None:
                print('acquire data : None data received')
            buffer.write(data)

            # Make the processing thread wait if if there are too many item queued up
            # otherwise the processing thread will make the DAQ thread very slow and lead to build up of items
            if queue.qsize()> queue_max:
                need_pause.set()
            
            if queue.qsize() < queue_max/2:
                need_pause.clear()

        except Empty:
            continue


class BatchProcessor(Processor):
    """A BatchProcessor starts an background thread to keep pulling the data, the process() function
    is called every specified interval. During each specified interval, all the data acquired since last call to 
    process() will be send to the current process

    """
 

    def __init__(self, interval=0.001, internal_buffer_size = 10000, verbose=True):
        """
        Args:
            interval (float, optional): Time interval at which batches of messages will be processed at once. Defaults to 0.001.
            internal_buffer_size (int, optional): internal butter size at which the messages will be held untill they are processed. Defaults to 10000.
            verbose (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.ring_buffer = RingBufferG(internal_buffer_size)
        self.end_time = None
        self.interval = interval
        self.verbose = verbose
        self.max_queue_size = 100

    def startup(self):
        # start a background thread to keep pulling the data
        self.need_pause = threading.Event() #whether to pause the processing thread to allow queue to clear
        self.acq_thread = threading.Thread(target=acquire_data, 
            args=(self.in_queue, self.ring_buffer, self.shutdown_event, self.QUEUE_PULL_TIMEOUT, self.need_pause, self.max_queue_size))
        self.acq_thread.daemon = True #make it a daemon thread, so it won't block the main program from exiting, and will close automatically
        self.acq_thread.start()

    def main_loop(self):

        while not self.shutdown_event.is_set():

            if self.interval >0:
                if self.end_time is None:
                    self.end_time  = time.monotonic() + self.interval
                
                # Wait till the future trigger point
                time2wait = self.end_time - time.monotonic()

                if time2wait>0:
                    time.sleep(time2wait)
                # else:
                #     if self.verbose:
                #         self.log(logging.DEBUG, f'Not able to keep with the specified speed. time2wait: {time2wait} ')

                if time.monotonic() >= self.end_time:
                    self.end_time = time.monotonic() + self.interval #update future trigger time
                    if not self.need_pause.is_set():
                        # pause processing so that the read queue can keep up
                        data_list = self.ring_buffer.readAvailable()
                        data = self._process(data_list)
                        self.send(data)
                    # else:
                        # self.log(logging.DEBUG, 'Waiting for acquire thread')
            else:
                # do not wait
                if not self.need_pause.is_set():
                    data_list = self.ring_buffer.readAvailable()
                    data = self._process(data_list)
                    self.send(data)

    def process(self, messages:List[Message]=None):
        """Process the input messages. A batch of messages will be passed at a batch every time interval

        Args:
            messages (List[Message], optional): messages to be processed. Defaults to None.

        Raises:
            NotImplementedError: This function must be implemented in subclasses
        """
        raise NotImplementedError(f'{self.__class__.__name__}.process is not implemented')


class DummyBatchProcessor(BatchProcessor):
    def process(self, data):
        self.log(logging.DEBUG, f'data read: {data}')
        return data


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    start  = time.time()
    with ProcessorContext() as ctx:

        ctx.makeProcessor('Source', DummyTimeSource, 1)
        ctx.makeProcessor('BatchProcessor', DummyBatchProcessor, interval=2)
        ctx.makeProcessor('Echo', EchoProcessor)

        ctx.connect('Source', 'BatchProcessor')
        ctx.connect('BatchProcessor', 'Echo')

        ctx.start()

        # time.sleep(10)
        # logging.debug('Sending the stop signal now')
        # #send stop signal after 5s
        # ctx.shutdown_event.set()


