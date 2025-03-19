import os
import torch
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
        preprocessor = WazePreprocessor(self.data_dir)
        raw_data = preprocessor.load_data(self.sample_size)
        cleaned_data = preprocessor.clean_data(raw_data)
        
        graph_builder = WazeGraphBuilder()
        nodes_df, edges_df = graph_builder.build_graph_structure(
            cleaned_data['segments_df'], 
            cleaned_data.get('jams_df')
        )
        
        snapshots = graph_builder.create_temporal_snapshots(
            edges_df, 
            self.interval_minutes, 
            self.max_snapshots
        )
        
        X, A = graph_builder.prepare_tensor_data(
            snapshots, 
            nodes_df, 
            self.feature_cols
        )
        
        timestamps = [s['timestamp'] for s in snapshots]
        
        return X, A, timestamps
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.X is None:
            return 0
        return max(0, self.X.shape[0] - self.sequence_length - self.prediction_horizon + 1)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Input sequence: [idx:idx+sequence_length, :, :]
        x_seq = self.X[idx:idx+self.sequence_length]
        a_seq = self.A[idx:idx+self.sequence_length]
        
        # Target sequence: [idx+sequence_length:idx+sequence_length+prediction_horizon, :, :]
        y_seq = self.X[idx+self.sequence_length:idx+self.sequence_length+self.prediction_horizon]
        
        return {
            'x_seq': x_seq,
            'a_seq': a_seq,
            'y_seq': y_seq,
            'timestamp': self.timestamps[idx] if self.timestamps else None
        }