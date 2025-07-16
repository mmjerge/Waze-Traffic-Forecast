#!/usr/bin/env python3
"""
Test script to verify data loading works with the local dataset.
"""
import os
import sys
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from waze_traffic_forecast._config import load_config
from waze_traffic_forecast.dataset import WazeGraphDataset

def test_data_loading():
    """Test that data can be loaded from the local files."""
    print("Testing data loading...")
    
    # Load configuration
    config = load_config('config.yaml')
    print(f"Data directory: {config['data']['directory']}")
    
    # Check if files exist
    data_dir = config['data']['directory']
    segments_file = os.path.join(data_dir, 'waze-jam-segments000-subset000.csv')
    jams_file = os.path.join(data_dir, 'waze-jams000-subset000.csv')
    
    print(f"Checking for segments file: {segments_file}")
    print(f"Segments file exists: {os.path.exists(segments_file)}")
    
    print(f"Checking for jams file: {jams_file}")
    print(f"Jams file exists: {os.path.exists(jams_file)}")
    
    if not os.path.exists(segments_file) or not os.path.exists(jams_file):
        print("ERROR: Required data files not found!")
        return False
    
    # Try to load a small sample
    print("\nTrying to load data...")
    try:
        # Test direct pandas loading
        print("Testing direct pandas loading...")
        segments_df = pd.read_csv(segments_file, nrows=100)
        jams_df = pd.read_csv(jams_file, nrows=100)
        
        print(f"Segments shape: {segments_df.shape}")
        print(f"Segments columns: {list(segments_df.columns)}")
        print(f"Jams shape: {jams_df.shape}")
        print(f"Jams columns: {list(jams_df.columns)}")
        
        # Test dataset loading
        print("\nTesting dataset loading...")
        dataset = WazeGraphDataset(
            data_dir=data_dir,
            full_graph=False,
            subgraph_nodes=100,  # Very small for testing
            sample_size=1000,    # Small sample
            max_snapshots=10,    # Few snapshots
            use_csv=True
        )
        
        print(f"Dataset initialized successfully!")
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            print("Testing sample retrieval...")
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            if 'x_seq' in sample:
                print(f"x_seq shape: {sample['x_seq'].shape}")
            
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✅ Data loading test passed!")
    else:
        print("\n❌ Data loading test failed!")