    
from typing import List, Tuple
from pyneurode.processor_node.Message import Message, NumpyMessage
from pyneurode.processor_node.Processor import Processor


class ScaleProcessor(Processor):
    def __init__(self, a:float, b:float):
        super().__init__()
        self.a = a
        self.b = b
        
    def process(self, message: NumpyMessage) -> Message:
        x = message.data*self.a+self.b
        return NumpyMessage(x)
    
    def get_IOspecs(self) -> Tuple[List[Message], List[Message]]:
        return ([NumpyMessage], [NumpyMessage])