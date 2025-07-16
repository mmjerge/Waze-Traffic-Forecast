"""
Dataset class for Waze traffic data that supports both sparse subgraph and full graph training.
"""
import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm  # Import tqdm for progress bars
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
        
        # Initialize with None values
        self.X = None
        self.A = None
        self.edge_index = None
        self.nodes_df = None
        self.timestamps = None
        
        # Process data with better error handling
        try:
            print("Processing Waze traffic data...")
            self.X, self.A, self.edge_index, self.nodes_df, self.timestamps = self._process_data()
            
            # For mini-batch training with full graph
            self.current_epoch = 0
            self.num_nodes = self.X.shape[1] if self.X is not None else 0
            print(f"Initialized dataset with {self.num_nodes} nodes and {self.X.shape[0] if self.X is not None else 0} time steps")
        except Exception as e:
            print(f"Error initializing dataset: {str(e)}")
            print("Falling back to minimal test data")
            # Initialize minimal test data if loading the real data fails
            self._initialize_minimal_test_data()
            self.current_epoch = 0
            self.num_nodes = 4  # For the minimal test data
        
    def _initialize_minimal_test_data(self):
        """Initialize minimal test data if loading the real data fails."""
        print("Initializing minimal test data")
        self.X = torch.zeros((6, 4, 4))
        # FIX: Use torch.sparse_coo_tensor instead of sparse.FloatTensor
        self.A = [torch.sparse_coo_tensor(
            indices=torch.LongTensor([[0, 1], [1, 2]]), 
            values=torch.FloatTensor([1.0, 1.0]), 
            size=(4, 4)
        ) for _ in range(6)]
        self.edge_index = torch.LongTensor([[0, 1], [1, 2]])
        self.nodes_df = pd.DataFrame({'node_id': range(4)})
        self.timestamps = list(range(6))
        
    def _process_data(self):
        """Process the data and create tensors with better error handling."""
        graph_builder = WazeGraphBuilder()
        edge_index = None
        nodes_df = None
        
        try:
            print(f"Looking for data files in {self.data_dir}")
            # Check for both parquet and CSV files
            segments_parquet = os.path.join(self.data_dir, 'waze-jam-segments000-subset000.parquet')
            segments_csv = os.path.join(self.data_dir, 'waze-jam-segments000-subset000.csv')
            jams_parquet = os.path.join(self.data_dir, 'waze-jams000-subset000.parquet')
            jams_csv = os.path.join(self.data_dir, 'waze-jams000-subset000.csv')
            
            # Determine which files to use based on availability and preference
            if os.path.exists(segments_parquet) and not self.use_csv:
                segments_file = segments_parquet
            elif os.path.exists(segments_csv):
                segments_file = segments_csv
            else:
                segments_file = segments_csv  # Default fallback
                
            if os.path.exists(jams_parquet) and not self.use_csv:
                jams_file = jams_parquet
            elif os.path.exists(jams_csv):
                jams_file = jams_csv
            else:
                jams_file = jams_csv  # Default fallback
            
            if os.path.exists(segments_file) and os.path.exists(jams_file):
                # Load data with progress indicators
                if segments_file.endswith('.parquet'):
                    print(f"Reading segments from parquet: {segments_file}")
                    segments_df = pd.read_parquet(segments_file)
                else:
                    print(f"Reading segments from CSV: {segments_file}")
                    segments_df = pd.read_csv(segments_file, low_memory=False)
                
                if jams_file.endswith('.parquet'):
                    print(f"Reading jams from parquet: {jams_file}")
                    jams_df = pd.read_parquet(jams_file)
                else:
                    print(f"Reading jams from CSV: {jams_file}")
                    jams_df = pd.read_csv(jams_file, low_memory=False)
                
                print(f"Loaded {len(segments_df)} segment records and {len(jams_df)} jam records")
                
                if self.sample_size:
                    segments_df = segments_df.sample(min(self.sample_size, len(segments_df)))
                    jams_df = jams_df.sample(min(self.sample_size, len(jams_df)))
                    print(f"Sampled to {len(segments_df)} segment records and {len(jams_df)} jam records")
                
                # Build graph structure with or without node sampling
                if self.full_graph:
                    print("Using full graph training with mini-batches")
                    max_nodes = None  # No limit on nodes
                else:
                    print(f"Using sparse subgraph with maximum {self.subgraph_nodes} nodes")
                    max_nodes = self.subgraph_nodes
                
                print("Building graph structure...")
                nodes_df, edges_df = graph_builder.build_graph_structure(
                    segments_df, jams_df, max_nodes=max_nodes
                )
                
                print("Creating temporal snapshots...")
                snapshots = graph_builder.create_temporal_snapshots(
                    edges_df, 
                    self.interval_minutes, 
                    self.max_snapshots,
                    max_edges_per_snapshot=None if self.full_graph else 100000
                )
                
                # Check if snapshots is empty
                if not snapshots:
                    print("Warning: No snapshots created. Check your time intervals.")
                    self._initialize_minimal_test_data()
                    return self.X, self.A, self.edge_index, self.nodes_df, self.timestamps
                
                print(f"Created {len(snapshots)} temporal snapshots")
                
                # Prepare feature columns
                if self.feature_cols is None:
                    self.feature_cols = []
                    for col in ['speed', 'severity', 'delay', 'length', 
                                'is_accident_related', 'time_since_accident']:
                        if col in edges_df.columns:
                            self.feature_cols.append(col)
                
                print(f"Using features: {self.feature_cols}")
                            
                # Prepare tensor data
                print("Preparing tensor data...")
                X, A = graph_builder.prepare_tensor_data(
                    snapshots, 
                    nodes_df, 
                    self.feature_cols
                )
                
                # Check if tensor creation was successful
                if X is None or A is None:
                    print("Warning: Failed to create tensors from snapshots")
                    self._initialize_minimal_test_data()
                    return self.X, self.A, self.edge_index, self.nodes_df, self.timestamps
                
                # Extract timestamps
                timestamps = [s['timestamp'] for s in snapshots]
                
                # Create edge index for mini-batch training
                if self.full_graph:
                    print("Creating edge index for mini-batch training...")
                    # Store edge index for neighborhood sampling
                    edge_index = self._create_edge_index(edges_df, nodes_df)
                    print("Edge index creation completed.")
                
                return X, A, edge_index, nodes_df, timestamps
                
            else:
                print(f"Waze data files not found. Looked for {segments_file} and {jams_file}")
                raise FileNotFoundError(f"Waze data files not found. Looked for {segments_file} and {jams_file}")
                
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            # Initialize minimal test data
            self._initialize_minimal_test_data()
            return self.X, self.A, self.edge_index, self.nodes_df, self.timestamps
            
    def _create_edge_index(self, edges_df, nodes_df):
        """
        Create edge index tensor from edges DataFrame with chunking for large datasets.
        Uses optimized approach to dramatically reduce memory usage and processing time.
        
        Args:
            edges_df: DataFrame with edge data
            nodes_df: DataFrame with node IDs
            
        Returns:
            edge_index: Edge index tensor [2, E]
        """
        try:
            print(f"Creating edge index from {len(edges_df)} edges...")
            
            # Create node ID mapping
            node_to_idx = {node_id: i for i, node_id in enumerate(nodes_df['node_id'])}
            
            # Process in chunks to avoid memory issues with very large edge lists
            chunk_size = 1000000  # Process 1M edges at a time
            num_chunks = (len(edges_df) + chunk_size - 1) // chunk_size  # Ceiling division
            
            # Pre-allocate lists for sources and targets
            all_sources = []
            all_targets = []
            
            # Create function to process a chunk of edges
            def process_chunk(chunk):
                valid_sources = []
                valid_targets = []
                
                # Use vectorized operations for better performance
                source_mask = chunk['source'].isin(node_to_idx)
                target_mask = chunk['target'].isin(node_to_idx)
                valid_mask = source_mask & target_mask
                
                # Get valid edges
                valid_sources_series = chunk.loc[valid_mask, 'source']
                valid_targets_series = chunk.loc[valid_mask, 'target']
                
                # Map to indices using list comprehension (faster than iterrows)
                sources = [node_to_idx[s] for s in valid_sources_series]
                targets = [node_to_idx[t] for t in valid_targets_series]
                
                return sources, targets
            
            # Process each chunk
            for chunk_idx in tqdm(range(num_chunks), desc="Processing edge chunks"):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(edges_df))
                
                # Get chunk of edges
                chunk = edges_df.iloc[start_idx:end_idx]
                
                # Process the chunk
                sources, targets = process_chunk(chunk)
                
                # Add to main lists
                all_sources.extend(sources)
                all_targets.extend(targets)
                
                # Print progress every few chunks
                if chunk_idx % 5 == 0 and chunk_idx > 0:
                    print(f"  Processed {chunk_idx}/{num_chunks} chunks, collected {len(all_sources)} edges so far")
            
            # Create edge index tensor all at once
            if all_sources:
                print(f"Creating tensor with {len(all_sources)} valid edges...")
                edge_index = torch.tensor([all_sources, all_targets], dtype=torch.long)
                print(f"Created edge index with shape {edge_index.shape}")
                return edge_index
            else:
                print("Warning: No valid edges found for edge index")
                return torch.zeros((2, 0), dtype=torch.long)
        
        except Exception as e:
            print(f"Error creating edge index: {str(e)}")
            # Return empty edge index
            return torch.zeros((2, 0), dtype=torch.long)
    
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
        try:
            batch_start = idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, self.num_nodes)
            
            rng = np.random.RandomState(42 + self.current_epoch + idx)
            seed_nodes = rng.permutation(self.num_nodes)[batch_start:batch_end].tolist()
            
            sampled_nodes, sampled_edge_index = self._sample_neighborhood(seed_nodes)
            
            # Check if we got valid data
            if not sampled_nodes or self.X.shape[0] == 0:
                print(f"Warning: Invalid sampled nodes for batch {idx}")
                # Return empty tensors with correct shapes
                x_shape = (self.sequence_length, 0, self.X.shape[2] if self.X.shape[0] > 0 else 0)
                y_shape = (self.prediction_horizon, 0, self.X.shape[2] if self.X.shape[0] > 0 else 0)
                return {
                    'x_seq': torch.zeros(x_shape),
                    'a_seq': [sampled_edge_index] * self.sequence_length,
                    'y_seq': torch.zeros(y_shape),
                    'sampled_nodes': [],
                    'global_indices': seed_nodes,
                    'batch_idx': idx
                }

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
        except Exception as e:
            print(f"Error in _get_minibatch_item for idx {idx}: {str(e)}")
            # Return empty tensors with correct shapes
            x_shape = (self.sequence_length, 0, self.X.shape[2] if self.X.shape[0] > 0 else 0)
            y_shape = (self.prediction_horizon, 0, self.X.shape[2] if self.X.shape[0] > 0 else 0)
            return {
                'x_seq': torch.zeros(x_shape),
                'a_seq': [torch.zeros((2, 0), dtype=torch.long)] * self.sequence_length,
                'y_seq': torch.zeros(y_shape),
                'sampled_nodes': [],
                'global_indices': [],
                'batch_idx': idx
            }
    
    def _sample_neighborhood(self, seed_nodes):
        """
        Sample multi-hop neighborhood for seed nodes.
        Highly optimized implementation for very large graphs.
        
        Args:
            seed_nodes: List of seed node indices
            
        Returns:
            all_nodes: List of all sampled nodes
            sampled_edge_index: Edge index for the sampled subgraph
        """
        try:
            # Early return if no edges available
            if self.edge_index is None or self.edge_index.shape[1] == 0:
                print("Warning: No edges available for neighborhood sampling")
                return seed_nodes, torch.zeros((2, 0), dtype=torch.long)
            
            print(f"Sampling neighborhood for {len(seed_nodes)} seed nodes with {self.num_hops} hops...")
            
            # For very large graphs, we'll use a more memory-efficient approach
            # by limiting the neighborhood size
            
            # 1. Set a maximum number of neighbors per node to sample
            max_neighbors_per_node = 20  # Adjust this based on your needs
            
            # 2. Initialize visited nodes and edges
            visited_nodes = set(seed_nodes)
            sampled_edges = set()
            
            # 3. Define a helper to efficiently find neighbors from edge_index
            def get_neighbors(nodes, max_per_node=max_neighbors_per_node):
                neighbors = {}
                
                # Search directly in the edge index tensor
                for node in nodes:
                    # Find indices where this node is the source
                    matching_indices = (self.edge_index[0] == node).nonzero(as_tuple=True)[0]
                    
                    # Sample a limited number of neighbors if there are too many
                    if len(matching_indices) > max_per_node:
                        # Randomly sample indices
                        perm = torch.randperm(len(matching_indices))
                        selected_indices = matching_indices[perm[:max_per_node]]
                    else:
                        selected_indices = matching_indices
                    
                    # Get target nodes
                    targets = self.edge_index[1, selected_indices].tolist()
                    neighbors[node] = targets
                    
                return neighbors
            
            # 4. BFS traversal with hop limitation
            current_frontier = set(seed_nodes)
            all_nodes = list(seed_nodes)
            
            # Track progress for each hop
            for hop in range(self.num_hops):
                print(f"  Processing hop {hop+1}/{self.num_hops} with frontier size {len(current_frontier)}")
                
                # Get neighbors for current frontier
                neighbor_dict = get_neighbors(current_frontier)
                
                # Prepare next frontier
                next_frontier = set()
                
                # Add neighbors to visited set and collect new edges
                for source, targets in neighbor_dict.items():
                    for target in targets:
                        # Add the edge
                        sampled_edges.add((source, target))
                        
                        # Check if this is a new node
                        if target not in visited_nodes:
                            visited_nodes.add(target)
                            all_nodes.append(target)
                            next_frontier.add(target)
                
                # Update frontier for next hop
                current_frontier = next_frontier
                if not current_frontier:
                    break
            
            # 5. Create mapping and build edge index
            node_map = {node: i for i, node in enumerate(all_nodes)}
            
            if sampled_edges:
                # Convert to tensor format with remapped indices
                edge_list = [[node_map[src], node_map[tgt]] for src, tgt in sampled_edges]
                sampled_edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                # Empty edge index
                sampled_edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            print(f"Sampled neighborhood with {len(all_nodes)} nodes and {len(sampled_edges)} edges")
            return all_nodes, sampled_edge_index
            
        except Exception as e:
            print(f"Error in neighborhood sampling: {str(e)}")
            return seed_nodes, torch.zeros((2, 0), dtype=torch.long)