import pandas as pd
from .handlers import (
    LoadDataHandler,
    CleanDataHandler,
    FeatureEngineeringHandler,
    SplitDataHandler,
    SaveDataHandler
)


class DataProcessor:
    def __init__(self, target_column, output_dir='.'):
        self.target_column = target_column
        self.output_dir = output_dir
        self.context = {}
        self._build_chain()
    
    def _build_chain(self):
        self.first_handler = LoadDataHandler()
        
        (self.first_handler
         .set_next(CleanDataHandler(missing_threshold=0.5))
         .set_next(FeatureEngineeringHandler(target_column=self.target_column))
         .set_next(SplitDataHandler(target_column=self.target_column))
         .set_next(SaveDataHandler(output_dir=self.output_dir)))
    
    def process(self, file_path):
        print("\nStarting data processing pipeline...")
        
        self.context = {
            'file_path': file_path,
            'target_column': self.target_column
        }
        
        result = self.first_handler.handle(pd.DataFrame(), self.context)
        print("\nProcessing completed successfully")
        
        return self.context
