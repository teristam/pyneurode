from dataclasses import dataclass
from typing import *
import numpy as np

class Message:
    '''
    Simple class that transfer data between Processors
    '''
    
    def __init__(self, type:str, data:Any, timestamp:Optional[float]=None):
        self.type = type
        self.data = data
        self.timestamp = timestamp

    def __repr__(self) -> str:

        if isinstance(self.data, np.ndarray):
            return f'Message (type: {self.type}, data shape: {self.data.shape}, timestamp: {self.timestamp})'
        else:
            return f'Message (type: {self.type}, data: {self.data}, timestamp: {self.timestamp})'
        

class MetricsMessage(Message):
    '''
    Message that contains metrics information
    '''
    type = 'metrics'
    
    def __init__(self, processor:str, measures:dict):
        self.data = {
            'processor': processor,
            'measures':measures
        }
    