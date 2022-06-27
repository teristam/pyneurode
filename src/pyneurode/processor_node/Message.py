from dataclasses import dataclass
from typing import *
import numpy as np
from py import process

@dataclass
class Message:
    '''
    Simple dataclass that transfer data between Processors
    '''
    type: str
    data: Any
    timestamp: float = None

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
    