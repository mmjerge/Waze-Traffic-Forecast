"""
Dataset class for Waze traffic data that supports both sparse subgraph and full graph training.
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
        full_graph=False,
        subgraph_nodes=5000,
        batch_size=1000,
        num_hops=2,
        sample_size=None, 
        interval_minutes=15, 
        max_snapshots=100,
        feature_cols=None,
        prediction_horizon=1,
        sequence_length=12,
        use_csv=True  # Add parameter to force using CSV files
    ):
        """
        Initialize the Waze graph dataset.
        
        Args:
            data_dir: Directory containing Waze data files
            full_graph: Whether to use full graph training with mini-batches
            subgraph_nodes: Maximum nodes for subgraph when not using full graph
            batch_size: Number of seed nodes per batch for mini-batch training
            num_hops: Number of hops for neighborhood sampling
            sample_size: Number of rows to sample (for testing)
            interval_minutes: Time interval between snapshots
            max_snapshots: Maximum number of snapshots to create
            feature_cols: List of feature columns to use
            prediction_horizon: Number of future time steps to predict
            sequence_length: Number of historical time steps to use as input
            use_csv: Whether to use CSV files instead of parquet files
        """
        self.data_dir = data_dir
        self.full_graph = full_graph
        self.subgraph_nodes = subgraph_nodes
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.sample_size = sample_size
        self.interval_minutes = interval_minutes
        self.max_snapshots = max_snapshots
        self.feature_cols = feature_cols
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        self.use_csv = use_csv
        
        # Process data
        self.X, self.A, self.edge_index, self.nodes_df, self.timestamps = self._process_data()
        
        # For mini-batch training with full graph
        self.current_epoch = 0
        self.num_nodes = self.X.shape[1] if self.X is not None else 0
        
    def _process_data(self):
        """Process the data and create tensors."""
        graph_builder = WazeGraphBuilder()
        edge_index = None
        nodes_df = None
        
        try:
            # Check for both parquet and CSV files
            segments_parquet = os.path.join(self.data_dir, 'waze-jam-segments000.parquet')
            segments_csv = os.path.join(self.data_dir, 'waze-jam-segments000.csv')
            jams_parquet = os.path.join(self.data_dir, 'waze-jams000.parquet')
            jams_csv = os.path.join(self.data_dir, 'waze-jams000.csv')
            
            # Determine which files to use based on availability and preference
            use_csv_segments = self.use_csv or (not os.path.exists(segments_parquet) and os.path.exists(segments_csv))
            use_csv_jams = self.use_csv or (not os.path.exists(jams_parquet) and os.path.exists(jams_csv))
            
            segments_file = segments_csv if use_csv_segments else segments_parquet
            jams_file = jams_csv if use_csv_jams else jams_parquet
            
            if os.path.exists(segments_file) and os.path.exists(jams_file):
                # Load data from appropriate file format
                if segments_file.endswith('.csv'):
                    print(f"Reading segments from CSV: {segments_file}")
                    segments_df = pd.read_csv(segments_file)
                else:
                    print(f"Reading segments from parquet: {segments_file}")
                    segments_df = pd.read_parquet(segments_file)
                
                if jams_file.endswith('.csv'):
                    print(f"Reading jams from CSV: {jams_file}")
                    jams_df = pd.read_csv(jams_file)
                else:
                    print(f"Reading jams from parquet: {jams_file}")
                    jams_df = pd.read_parquet(jams_file)
                
                if self.sample_size:
                    segments_df = segments_df.sample(min(self.sample_size, len(segments_df)))
                    jams_df = jams_df.sample(min(self.sample_size, len(jams_df)))
                
                # Build graph structure with or without node sampling
                if self.full_graph:
                    print("Using full graph training with mini-batches")
                    max_nodes = None  # No limit on nodes
                else:
                    print(f"Using sparse subgraph with maximum {self.subgraph_nodes} nodes")
                    max_nodes = self.subgraph_nodes
                
                nodes_df, edges_df = graph_builder.build_graph_structure(
                    segments_df, jams_df, max_nodes=max_nodes
                )
                
                # Create temporal snapshots
                snapshots = graph_builder.create_temporal_snapshots(
                    edges_df, 
                    self.interval_minutes, 
                    self.max_snapshots,
                    max_edges_per_snapshot=None if self.full_graph else 100000
                )
                
                # Prepare feature columns
                if self.feature_cols is None:
                    self.feature_cols = []
                    for col in ['speed', 'severity', 'delay', 'length']:
                        if col in edges_df.columns:
                            self.feature_cols.append(col)
                
                # Prepare tensor data
                X, A = graph_builder.prepare_tensor_data(
                    snapshots, 
                    nodes_df, 
                    self.feature_cols
                )
                
                # Extract timestamps
                timestamps = [s['timestamp'] for s in snapshots]
                
                # Create edge index for mini-batch training
                if self.full_graph:
                    # Store edge index for neighborhood sampling
                    edge_index = self._create_edge_index(edges_df, nodes_df)
                
                return X, A, edge_index, nodes_df, timestamps
                
            else:
                raise FileNotFoundError(f"Waze data files not found. Looked for {segments_file} and {jams_file}")
                
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            # Return minimal test data
            X = torch.zeros((6, 4, 4))
            A = [torch.sparse.FloatTensor(
                indices=torch.LongTensor([[0, 1], [1, 2]]), 
                values=torch.FloatTensor([1.0, 1.0]), 
                size=(4, 4)
            ) for _ in range(6)]
            edge_index = torch.LongTensor([[0, 1], [1, 2]])
            nodes_df = pd.DataFrame({'node_id': range(4)})
            timestamps = list(range(6))
            
            return X, A, edge_index, nodes_df, timestamps
            
    def _create_edge_index(self, edges_df, nodes_df):
        """
        Create edge index tensor from edges DataFrame.
        
        Args:
            edges_df: DataFrame with edge data
            nodes_df: DataFrame with node IDs
            
        Returns:
            edge_index: Edge index tensor [2, E]
        """
        # Create node ID mapping
        node_to_idx = {node_id: i for i, node_id in enumerate(nodes_df['node_id'])}
        
        # Extract source and target node indices
        sources = []
        targets = []
        
        for _, edge in edges_df.iterrows():
            source = edge['source']
            target = edge['target']
            
            if source in node_to_idx and target in node_to_idx:
                sources.append(node_to_idx[source])
                targets.append(node_to_idx[target])
        
        # Create edge index tensor
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        
        return edge_index
    
    def init_epoch(self, epoch):
        """Initialize for a new epoch (for mini-batch mode)."""
        self.current_epoch = epoch
        np.random.seed(42 + epoch)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.X is None:
            return 0
        
        if self.full_graph:
            # For mini-batch training, return number of batches
            return int(np.ceil(self.num_nodes / self.batch_size))
        else:
            # For subgraph training, use sliding window approach
            total_timesteps = self.X.shape[0]
            if total_timesteps < (self.sequence_length + self.prediction_horizon):
                print(f"Warning: Not enough timesteps ({total_timesteps}) for sequence_length ({self.sequence_length}) + prediction_horizon ({self.prediction_horizon})")
                return 0
                
            return max(1, total_timesteps - (self.sequence_length + self.prediction_horizon) + 1)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.full_graph:
            return self._get_minibatch_item(idx)
        else:
            return self._get_subgraph_item(idx)
    
    def _get_subgraph_item(self, idx):
        """Get a sample using the sliding window approach for subgraph training."""

        if len(self) <= 0:
            x_seq = self.X[:self.sequence_length] if self.X.shape[0] >= self.sequence_length else self.X
            
            if isinstance(self.A, list):
                a_seq = self.A[:self.sequence_length] if len(self.A) >= self.sequence_length else self.A
            else:
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
        
        if isinstance(self.A, list):
            a_seq = self.A[start_idx:end_idx]
        else:
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
    
    def _get_minibatch_item(self, idx):
        """Get a mini-batch of nodes and their neighborhood for full graph training."""

        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, self.num_nodes)
        
        rng = np.random.RandomState(42 + self.current_epoch + idx)
        seed_nodes = rng.permutation(self.num_nodes)[batch_start:batch_end].tolist()
        
        sampled_nodes, sampled_edge_index = self._sample_neighborhood(seed_nodes)
        

        x_sampled = torch.stack([
            self.X[t, sampled_nodes] for t in range(self.X.shape[0])
        ])
        

        total_steps_needed = self.sequence_length + self.prediction_horizon
        if x_sampled.shape[0] < total_steps_needed:
            padding = torch.zeros(
                (total_steps_needed - x_sampled.shape[0], x_sampled.shape[1], x_sampled.shape[2]),
                device=x_sampled.device
            )
            x_sampled = torch.cat([x_sampled, padding], dim=0)
        
        x_seq = x_sampled[:self.sequence_length]
        
        y_seq = x_sampled[self.sequence_length:self.sequence_length + self.prediction_horizon]
        

        a_seq = [sampled_edge_index] * self.sequence_length
        
        return {
            'x_seq': x_seq,
            'a_seq': a_seq,
            'y_seq': y_seq,
            'sampled_nodes': sampled_nodes,
            'global_indices': seed_nodes, 
            'batch_idx': idx
        }
    
    def _sample_neighborhood(self, seed_nodes):
        """
        Sample multi-hop neighborhood for seed nodes.
        
        Args:
            seed_nodes: List of seed node indices
            
        Returns:
            all_nodes: List of all sampled nodes
            sampled_edge_index: Edge index for the sampled subgraph
        """
        if self.edge_index is None:
            return seed_nodes, torch.zeros((2, 0), dtype=torch.long)
        
        adjacency_list = {}
        for i in range(self.edge_index.shape[1]):
            source, target = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            if source not in adjacency_list:
                adjacency_list[source] = []
            adjacency_list[source].append(target)
        
        visited = set(seed_nodes)
        frontier = set(seed_nodes)
        all_nodes = list(seed_nodes)
        
        sampled_edges = []
        
        for hop in range(self.num_hops):
            new_frontier = set()
            
            for node in frontier:
                if node not in adjacency_list:
                    continue
                    
                neighbors = adjacency_list[node]
                
                for neighbor in neighbors:
                    # Add edges between current node and its neighbors
                    sampled_edges.append((node, neighbor))
                    
                    # Add unvisited neighbors to the new frontier
                    if neighbor not in visited:
                        visited.add(neighbor)
                        all_nodes.append(neighbor)
                        new_frontier.add(neighbor)
            
            # Update frontier for next hop
            frontier = new_frontier
            if len(frontier) == 0:
                break
        
        # Create mapping from original to new node indices
        node_idx_map = {node: i for i, node in enumerate(all_nodes)}
        
        # Create new edge index with remapped node indices
        sampled_edge_index = []
        for source, target in sampled_edges:
            sampled_edge_index.append([node_idx_map[source], node_idx_map[target]])
        
        if sampled_edge_index:
            sampled_edge_index = torch.tensor(sampled_edge_index, dtype=torch.long).t()
        else:
            # Empty edge index
            sampled_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return all_nodes, sampled_edge_index