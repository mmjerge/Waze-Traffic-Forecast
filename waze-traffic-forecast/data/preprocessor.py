import os
import pandas as pd
import numpy as np
from datetime import datetime

class WazePreprocessor:
    def __init__(self, data_dir):
        """Initialize the preprocessor with directory containing Waze parquet files."""
        self.data_dir = data_dir
        self.file_paths = self._get_file_paths()
        
    def _get_file_paths(self):
        """Get paths to all relevant parquet files."""
        file_paths = {
            'jam_segments': os.path.join(self.data_dir, 'waze-jam-segments000.parquet'),
            'jam_line': os.path.join(self.data_dir, 'waze-jam-line000.parquet'),
            'jams': [
                os.path.join(self.data_dir, f'waze-jams00{i}.parquet') 
                for i in range(4) if os.path.exists(os.path.join(self.data_dir, f'waze-jams00{i}.parquet'))
            ],
            'alerts': os.path.join(self.data_dir, 'waze-alerts000.parquet')
        }
        return file_paths
    
    def load_data(self, sample_size=None):
        data = {}
        
        if os.path.exists(self.file_paths['jam_segments']):
            data['segments_df'] = pd.read_parquet(self.file_paths['jam_segments'])
            if sample_size:
                data['segments_df'] = data['segments_df'].sample(min(sample_size, len(data['segments_df'])))
        
        if os.path.exists(self.file_paths['jam_line']):
            data['jam_line_df'] = pd.read_parquet(self.file_paths['jam_line'])
            if sample_size:
                data['jam_line_df'] = data['jam_line_df'].sample(min(sample_size, len(data['jam_line_df'])))
        
        data['jams_df'] = pd.DataFrame()
        for jams_file in self.file_paths['jams']:
            if os.path.exists(jams_file):
                jams = pd.read_parquet(jams_file)
                if sample_size:
                    jams = jams.sample(min(sample_size, len(jams)))
                data['jams_df'] = pd.concat([data['jams_df'], jams])
        
        return data
    
    def clean_data(self, data):
        cleaned_data = {}
        
        if 'segments_df' in data:
            segments_df = data['segments_df'].copy()
            
            if 'scrapedatetime' in segments_df.columns:
                if segments_df['scrapedatetime'].dtype != 'datetime64[ns]':
                    try:
                        segments_df['scrapedatetime'] = pd.to_datetime(
                            segments_df['scrapedatetime'], unit='s'
                        )
                    except:
                        pass
            
            cleaned_data['segments_df'] = segments_df
        
        if 'jams_df' in data and not data['jams_df'].empty:
            jams_df = data['jams_df'].copy()
            
            for col in ['severity', 'speed', 'length', 'delay']:
                if col in jams_df.columns:
                    jams_df[col] = jams_df[col].fillna(0)
            
            cleaned_data['jams_df'] = jams_df
            
        return cleaned_data
