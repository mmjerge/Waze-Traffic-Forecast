import os
import pandas as pd
import numpy as np
from datetime import datetime

class WazePreprocessor:
    def __init__(self, data_dir, use_csv=False):
        """
        Initialize the preprocessor with directory containing Waze data files.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing Waze data files
        use_csv : bool, default=False
            If True, force using CSV files instead of parquet
        """
        self.data_dir = data_dir
        self.use_csv = use_csv
        self.file_paths = self._get_file_paths()
    
    def _get_file_paths(self):
        """Get paths to all relevant data files (parquet or csv)."""
        # Define file name templates without extensions
        base_names = {
            'jam_segments': 'waze-jam-segments000',
            'jam_line': 'waze-jam-line000',
            'jams': ['waze-jams00{}'.format(i) for i in range(4)],
            'alerts': 'waze-alerts000'
        }
        
        file_paths = {}
        
        # Handle single files (jam_segments, jam_line, alerts)
        for key in ['jam_segments', 'jam_line', 'alerts']:
            base_name = base_names[key]
            # Check if parquet exists and we're not forcing CSV
            parquet_path = os.path.join(self.data_dir, f'{base_name}.parquet')
            csv_path = os.path.join(self.data_dir, f'{base_name}.csv')
            
            if os.path.exists(parquet_path) and not self.use_csv:
                file_paths[key] = parquet_path
            elif os.path.exists(csv_path):
                file_paths[key] = csv_path
            else:
                file_paths[key] = None  # No valid file found
        
        # Handle multiple jams files
        file_paths['jams'] = []
        for base_name in base_names['jams']:
            parquet_path = os.path.join(self.data_dir, f'{base_name}.parquet')
            csv_path = os.path.join(self.data_dir, f'{base_name}.csv')
            
            if os.path.exists(parquet_path) and not self.use_csv:
                file_paths['jams'].append(parquet_path)
            elif os.path.exists(csv_path):
                file_paths['jams'].append(csv_path)
        
        return file_paths
    
    def load_data(self, sample_size=None):
        """
        Load data from parquet or csv files.
        
        Parameters:
        -----------
        sample_size : int, optional
            If provided, sample this many rows from each dataframe
            
        Returns:
        --------
        dict of DataFrames
            Dictionary containing loaded dataframes
        """
        data = {}
        
        # Load jam_segments
        if self.file_paths['jam_segments']:
            file_path = self.file_paths['jam_segments']
            if file_path.endswith('.parquet'):
                data['segments_df'] = pd.read_parquet(file_path)
            else:
                data['segments_df'] = pd.read_csv(file_path)
                
            if sample_size:
                data['segments_df'] = data['segments_df'].sample(min(sample_size, len(data['segments_df'])))
        
        # Load jam_line
        if self.file_paths['jam_line']:
            file_path = self.file_paths['jam_line']
            if file_path.endswith('.parquet'):
                data['jam_line_df'] = pd.read_parquet(file_path)
            else:
                data['jam_line_df'] = pd.read_csv(file_path)
                
            if sample_size:
                data['jam_line_df'] = data['jam_line_df'].sample(min(sample_size, len(data['jam_line_df'])))
        
        # Load and concatenate jams files
        data['jams_df'] = pd.DataFrame()
        for jams_file in self.file_paths['jams']:
            if jams_file.endswith('.parquet'):
                jams = pd.read_parquet(jams_file)
            else:
                jams = pd.read_csv(jams_file)
                
            if sample_size:
                jams = jams.sample(min(sample_size, len(jams)))
            
            data['jams_df'] = pd.concat([data['jams_df'], jams])
            
        # Load alerts
        if self.file_paths['alerts']:
            file_path = self.file_paths['alerts']
            if file_path.endswith('.parquet'):
                data['alerts_df'] = pd.read_parquet(file_path)
            else:
                data['alerts_df'] = pd.read_csv(file_path)
                
            if sample_size:
                data['alerts_df'] = data['alerts_df'].sample(min(sample_size, len(data['alerts_df'])))
        
        return data
    
    def clean_data(self, data):
        """Clean loaded data."""
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
            
        if 'alerts_df' in data:
            alerts_df = data['alerts_df'].copy()
            cleaned_data['alerts_df'] = alerts_df
            
        return cleaned_data
