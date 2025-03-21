"""
Dataset class for Waze traffic data that handles loading, processing,
and preparing data for graph transformer models.
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .data.preprocessor import WazePreprocessor
from .data.graph_builder import WazeGraphBuilder

class WazeGraphDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        sample_size=None, 
        interval_minutes=15, 
        max_snapshots=100,
        feature_cols=None,
        prediction_horizon=1,
        sequence_length=12
    ):
        """
        Initialize the Waze graph dataset.
        
        Args:
            data_dir: Directory containing Waze parquet files
            sample_size: Number of rows to sample (for testing)
            interval_minutes: Time interval between snapshots
            max_snapshots: Maximum number of snapshots to create
            feature_cols: List of feature columns to use
            prediction_horizon: Number of future time steps to predict
            sequence_length: Number of historical time steps to use as input
        """
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.interval_minutes = interval_minutes
        self.max_snapshots = max_snapshots
        self.feature_cols = feature_cols
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        
        self.X, self.A, self.timestamps = self._process_data()
        
    def _process_data(self):
        """Process the data and create tensors."""
        graph_builder = WazeGraphBuilder()
        
        preprocessor = WazePreprocessor(self.data_dir) if hasattr(self, 'preprocessor') else None
        
        try:
            if preprocessor:
                raw_data = preprocessor.load_data(self.sample_size)
                cleaned_data = preprocessor.clean_data(raw_data)
                nodes_df, edges_df = graph_builder.build_graph_structure(
                    cleaned_data['segments_df'], 
                    cleaned_data.get('jams_df')
                )
            else:
                try:
                    segments_file = os.path.join(self.data_dir, 'waze-jam-segments000.parquet')
                    jams_file = os.path.join(self.data_dir, 'waze-jams000.parquet')
                    
                    if os.path.exists(segments_file) and os.path.exists(jams_file):
                        segments_df = pd.read_parquet(segments_file)
                        jams_df = pd.read_parquet(jams_file)
                        
                        if self.sample_size:
                            segments_df = segments_df.sample(min(self.sample_size, len(segments_df)))
                            jams_df = jams_df.sample(min(self.sample_size, len(jams_df)))
                        
                        nodes_df, edges_df = graph_builder.build_graph_structure(segments_df, jams_df)
                    else:
                        raise FileNotFoundError("Waze data files not found")
                except Exception as e:
                    print(f"Error loading parquet files directly: {str(e)}")
                    print("Falling back to dummy data for testing...")
                    
                    num_nodes = 50
                    num_edges = 200
                    
                    nodes_df = pd.DataFrame({'node_id': np.arange(num_nodes)})
                    
                    sources = np.random.randint(0, num_nodes, num_edges)
                    targets = np.random.randint(0, num_nodes, num_edges)
                    timestamps = pd.date_range(start='2023-01-01', periods=24, freq='1H')
                    
                    edges_df = pd.DataFrame({
                        'source': sources,
                        'target': targets,
                        'scrapedatetime': np.random.choice(timestamps, num_edges),
                        'speed': np.random.uniform(10, 100, num_edges),
                        'severity': np.random.randint(0, 5, num_edges),
                        'delay': np.random.uniform(0, 30, num_edges),
                        'length': np.random.uniform(100, 1000, num_edges)
                    })
            
            snapshots = graph_builder.create_temporal_snapshots(
                edges_df, 
                self.interval_minutes, 
                self.max_snapshots
            )
            
            if self.feature_cols is None:
                self.feature_cols = []
                for col in ['speed', 'severity', 'delay', 'length']:
                    if col in edges_df.columns:
                        self.feature_cols.append(col)
            
            X, A = graph_builder.prepare_tensor_data(
                snapshots, 
                nodes_df, 
                self.feature_cols
            )
            
            timestamps = [s['timestamp'] for s in snapshots]
            
            return X, A, timestamps
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            
            X = torch.zeros((6, 4, 4))  # [time_steps, nodes, features]
            A = torch.zeros((6, 4, 4))  # [time_steps, nodes, nodes]
            timestamps = list(range(6))
            
            return X, A, timestamps
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.X is None:
            return 0
        total_timesteps = self.X.shape[0]
        
        if total_timesteps < (self.sequence_length + self.prediction_horizon):
            print(f"Warning: Not enough timesteps ({total_timesteps}) for sequence_length ({self.sequence_length}) + prediction_horizon ({self.prediction_horizon})")
            return 0
            
        return max(1, total_timesteps - (self.sequence_length + self.prediction_horizon) + 1)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if len(self) <= 0:
            x_seq = self.X[:self.sequence_length] if self.X.shape[0] >= self.sequence_length else self.X
            a_seq = self.A[:self.sequence_length] if self.A.shape[0] >= self.sequence_length else self.A
            
            y_seq = x_seq
            
            timestamp = self.timestamps[0] if self.timestamps else None
            
            return {
                'x_seq': x_seq,
                'a_seq': a_seq,
                'y_seq': y_seq,
                'timestamp': timestamp
            }
            
        start_idx = idx
        end_idx = idx + self.sequence_length
        x_seq = self.X[start_idx:end_idx]
        a_seq = self.A[start_idx:end_idx]
        
        target_start = end_idx
        target_end = end_idx + self.prediction_horizon
        target_end = min(target_end, self.X.shape[0])
        y_seq = self.X[target_start:target_end]
        
        if target_end - target_start < self.prediction_horizon:
            last_timestep = self.X[target_end-1:target_end]
            padding = last_timestep.repeat(self.prediction_horizon - (target_end - target_start), 1, 1)
            y_seq = torch.cat([y_seq, padding], dim=0)
        
        return {
            'x_seq': x_seq,
            'a_seq': a_seq,
            'y_seq': y_seq,
            'timestamp': self.timestamps[start_idx] if self.timestamps else None
        }