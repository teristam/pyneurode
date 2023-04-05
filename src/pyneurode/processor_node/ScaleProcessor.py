    
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.Processor import Processor


class ScaleProcessor(Processor):
    def __init__(self, a:float, b:float):
        super().__init__()
        self.a = a
        self.b = b
        
    def process(self, message: Message) -> Message:
        x = message.data*self.a+self.b
        return Message(message.dtype, x)