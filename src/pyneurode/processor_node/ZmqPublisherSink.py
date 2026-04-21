import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import zmq
from matplotlib.font_manager import json_dump
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.Processor import MsgTimeSource, Sink
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.ZmqSubscriberSource import ZmqSubscriberSource


class ZmqMessage(Message):
    type = 'zmq_message'
    def __init__(self, data:Dict, timestamp: Optional[float] = None):
        super().__init__(self.type, data, timestamp)

class ZmqPublisherSink(Sink):
    
    def __init__(self, addr="tcp://*:5554", verbose=False):
        super().__init__()
        self.addr = addr
        self.verbose = verbose
        
    def startup(self):
        super().startup()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.addr)
        self.log(logging.INFO,self.addr)

    def process(self, message: Message):
        if type(message) is ZmqMessage:
            if self.verbose:
                self.log(logging.INFO, message)
            self.socket.send_string(json.dumps(message.data))
            


if __name__ == '__main__':
    
    msg = [ZmqMessage({
        'number': 1.34,
        'text': 'this is a text'
    })]
    
    with ProcessorContext() as ctx:
        
        
        source = MsgTimeSource(1, msg)
        zmqSink = ZmqPublisherSink()
        zmqSource = ZmqSubscriberSource(interval=0)

        source.connect(zmqSink)

        ctx.start()
