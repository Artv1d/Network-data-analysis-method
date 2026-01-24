from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class DataHandler(ABC):
    def __init__(self):
        self._next_handler = None
    
    def set_next(self, handler):
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, data, context):
        pass
    
    def _pass_to_next(self, data, context):
        if self._next_handler:
            return self._next_handler.handle(data, context)
        return data
