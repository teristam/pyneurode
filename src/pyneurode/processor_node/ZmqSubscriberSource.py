
import logging
from pyneurode.processor_node.Processor import Source, TimeSource
import zmq
import json

class ZmqSubscriberSource(TimeSource):
    
    def __init__(self, interval=0, addr="tcp://localhost:5554"):
        super().__init__(interval=interval)
        self.addr = addr
        
    def startup(self):
        self.context = zmq.Context()
        self.poller = zmq.Poller()

        self.socket = self.context.socket(zmq.SUB) 
        self.socket.connect(self.addr)

        self.socket.setsockopt(zmq.SUBSCRIBE, b'')


        self.poller.register(self.socket, zmq.POLLIN)
        
    def process(self):
        socks = dict(self.poller.poll(1))
        # check if there is any message in queues
        
        if self.socket in socks:
            try:
                message = self.socket.recv(zmq.NOBLOCK)
            except zmq.ZMQError as err:
                self.log(logging.ERROR, f"Got error {err}")
                
            if message:
                result = json.loads(message.decode('utf-8'))
                self.log(logging.DEBUG, f'Recevied {result}')