import pandas as pd
import numpy as np
import os
from .base_handler import DataHandler


class LoadDataHandler(DataHandler):
    def handle(self, data, context):
        print(f"Loading data from {context['file_path']}")
        
        encodings = ['utf-8', 'cp1251', 'windows-1251', 'latin1']
        loaded_data = None
        
        for encoding in encodings:
            try:
                loaded_data = pd.read_csv(context['file_path'], encoding=encoding)
                break
            except:
                continue
        
        if loaded_data is None:
            loaded_data = pd.read_csv(context['file_path'])
        
        print(f"Loaded: {len(loaded_data)} rows, {len(loaded_data.columns)} columns")
        context['original_shape'] = loaded_data.shape
        context['original_columns'] = list(loaded_data.columns)
        
        return self._pass_to_next(loaded_data, context)


class CleanDataHandler(DataHandler):
    def __init__(self, missing_threshold=0.5):
        super().__init__()
        self.missing_threshold = missing_threshold
    
    def handle(self, data, context):
        print("Cleaning data...")
        
        missing_ratio = data.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > self.missing_threshold].index
        
        if len(cols_to_drop) > 0:
            data = data.drop(columns=cols_to_drop)
            context['dropped_columns'] = list(cols_to_drop)
        
        for column in data.columns:
            if data[column].isnull().any():
                if pd.api.types.is_numeric_dtype(data[column]):
                    data[column] = data[column].fillna(data[column].median())
                else:
                    if not data[column].mode().empty:
                        data[column] = data[column].fillna(data[column].mode()[0])
        
        data = data.drop_duplicates()
        context['cleaned_shape'] = data.shape
        
        return self._pass_to_next(data, context)


class FeatureEngineeringHandler(DataHandler):
    def __init__(self, target_column=None):
        super().__init__()
        self.target_column = target_column
    
    def handle(self, data, context):
        print("Engineering features...")
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != self.target_column:
                if data[col].nunique() == 2:
                    data[col] = pd.factorize(data[col])[0]
                elif data[col].str.replace('.', '', 1).str.isdigit().all():
                    try:
                        data[col] = pd.to_numeric(data[col])
                    except:
                        pass
        
        return self._pass_to_next(data, context)


class SplitDataHandler(DataHandler):
    def __init__(self, target_column):
        super().__init__()
        self.target_column = target_column
    
    def handle(self, data, context):
        print("Splitting data into X and y...")
        
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        context['X'] = data.drop(columns=[self.target_column])
        context['y'] = data[self.target_column]
        
        print(f"X shape: {context['X'].shape}")
        print(f"y shape: {context['y'].shape}")
        
        return self._pass_to_next(data, context)


class SaveDataHandler(DataHandler):
    def __init__(self, output_dir='.'):
        super().__init__()
        self.output_dir = output_dir
    
    def handle(self, data, context):
        print("Saving processed data...")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        x_path = os.path.join(self.output_dir, 'x_data.npy')
        y_path = os.path.join(self.output_dir, 'y_data.npy')
        feature_names_path = os.path.join(self.output_dir, 'feature_names.txt')
        
        np.save(x_path, context['X'].values)
        np.save(y_path, context['y'].values)
        
        with open(feature_names_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(context['X'].columns))
        
        print(f"Saved: {x_path}, {y_path}")
        
        return self._pass_to_next(data, context)
