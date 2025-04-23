"""
Script to build a graph dataset from raw Waze traffic data and save processed tensors.
"""
import os
import argparse
import torch
from waze_traffic_forecast.dataset import WazeGraphDataset

def main():
    """Build the Waze traffic graph dataset and save feature and adjacency tensors."""
    parser = argparse.ArgumentParser(description='Build graph from Waze data')
    # Parse command-line arguments
    parser.add_argument('--data_dir', type=str, default='/scratch/mj6ux/data/waze',
                        help='Directory containing Waze parquet files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save processed data')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of rows to sample from each file (for testing)')
    parser.add_argument('--interval', type=int, default=15,
                        help='Time interval between snapshots in minutes')
    parser.add_argument('--max_snapshots', type=int, default=100,
                        help='Maximum number of snapshots to create')
    
    args = parser.parse_args()
    # Create output directory if it does not exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Building graph dataset from Waze data in {args.data_dir}")
    # Initialize dataset with graph construction parameters
    dataset = WazeGraphDataset(
        data_dir=args.data_dir,
        sample_size=args.sample,
        interval_minutes=args.interval,
        max_snapshots=args.max_snapshots
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Save tensors if creation was successful
    if dataset.X is not None and dataset.A is not None:
        print(f"Feature tensor shape: {dataset.X.shape}")
        print(f"Adjacency tensor shape: {dataset.A.shape}")
        
        if args.output_dir:
            output_file = os.path.join(args.output_dir, 'waze_graph_data.pt')
            # Write feature tensor and adjacency list to disk
            torch.save({
                'X': dataset.X,
                'A': dataset.A,
                'timestamps': dataset.timestamps
            }, output_file)
            print(f"Saved graph data to {output_file}")
    else:
        print("Failed to create tensors")

if __name__ == "__main__":
    main()