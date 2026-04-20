from dataclasses import dataclass
from typing import *
from typing import Any, Optional
import numpy as np

class Message:
    '''
    Simple class that transfer data between Processors
    '''
    
    def __init__(self, dtype:str, data:Any, timestamp:Optional[float]=None, source:Optional[str]=''):
        self.dtype = dtype
        self.data = data
        self.timestamp = timestamp
        self.source = source

    def __repr__(self) -> str:

        if isinstance(self.data, np.ndarray):
            return f'Message (dtype: {self.dtype}, data shape: {self.data.shape}, timestamp: {self.timestamp})'
        else:
            return f'Message (dtype: {self.dtype}, data: {self.data}, timestamp: {self.timestamp})'
        

class MetricsMessage(Message):
    '''
    Message that contains metrics information
    '''
    dtype = 'metrics'
    
    def __init__(self, processor:str, measures:dict):
        data = {
            'processor': processor,
            'measures':measures
        }
        super().__init__(self.dtype, data)
    

class NumpyMessage(Message):
    dtype = 'numpy'
    def __init__(self, data:np.ndarray, timestamp: Optional[float] = None):
        super().__init__(self.dtype, data, timestamp)