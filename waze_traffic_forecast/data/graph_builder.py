"""
Graph construction from Waze data for traffic forecasting.
Builds nodes, edges, and temporal snapshots for graph transformer models.
"""
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm  # Import tqdm for progress bars

class WazeGraphBuilder:
    def __init__(self):
        """Initialize the graph builder."""
        self.nodes = None
        self.edges = None
        self.node_features = None
        self.edge_features = None
        self.timestamps = None
    
    def build_graph_structure(self, segments_df, jams_df=None, max_nodes=5000):
        """
        Build the basic graph structure from segments data.
        
        Args:
            segments_df: DataFrame with jam segments
            jams_df: DataFrame with jam attributes
            max_nodes: Maximum number of nodes to include in the graph
            
        Returns:
            nodes_df, edges_df
        """
        # Extract unique nodes from segments
        from_nodes = segments_df['fromnode'].unique()
        to_nodes = segments_df['tonode'].unique()
        unique_nodes = np.unique(np.concatenate([from_nodes, to_nodes]))
        
        print(f"Extracted {len(unique_nodes)} unique nodes from segments")
        
        # Create node DataFrame
        nodes_df = pd.DataFrame({'node_id': unique_nodes})
        
        # Create edge DataFrame
        edges_df = segments_df[['fromnode', 'tonode', 'jamid', 'scrapedatetime']].copy()
        edges_df.rename(columns={'fromnode': 'source', 'tonode': 'target'}, inplace=True)
        
        print(f"Created {len(edges_df)} edges from segments")
        
        # Join with jams data if available
        if jams_df is not None and len(jams_df) > 0:
            # Select key traffic attributes including accident data
            jam_attrs = ['id', 'severity', 'speed', 'length', 'delay', 'level', 'accident_uuid', 'accident_timestamp']
            existing_attrs = [col for col in jam_attrs if col in jams_df.columns]
            
            if 'id' in existing_attrs and len(existing_attrs) > 1:
                # Rename id to jamid for merging if needed
                if 'jamid' not in jams_df.columns and 'id' in jams_df.columns:
                    jams_merge = jams_df[existing_attrs].copy()
                    jams_merge.rename(columns={'id': 'jamid'}, inplace=True)
                    
                    # NEW: Process binary accident data
                    if 'accident_uuid' in jams_merge.columns:
                        jams_merge['is_accident_related'] = jams_merge['accident_uuid'].notna().astype(int)
                        
                        # Calculate time since accident report
                        if 'accident_timestamp' in jams_merge.columns:
                            # Convert timestamps to datetime if needed
                            accident_time = pd.to_datetime(jams_merge['accident_timestamp'])
                            scrape_time = pd.to_datetime(edges_df['scrapedatetime'])
                            
                            # Calculate minutes since accident
                            time_diff = (scrape_time - accident_time).dt.total_seconds() / 60
                            jams_merge['time_since_accident'] = time_diff.fillna(0).clip(lower=0)
                            
                            # Set accident flag to 0 if too much time has passed (3 hours max)
                            max_duration = 180  # 3 hours
                            expired_mask = jams_merge['time_since_accident'] > max_duration
                            jams_merge.loc[expired_mask, 'is_accident_related'] = 0
                            jams_merge.loc[expired_mask, 'time_since_accident'] = 0
                        else:
                            jams_merge['time_since_accident'] = 0
                    else:
                        # Default values if no accident data
                        jams_merge['is_accident_related'] = 0
                        jams_merge['time_since_accident'] = 0
                    
                    # Merge with edges
                    edges_df = edges_df.merge(
                        jams_merge,
                        on='jamid',
                        how='left'
                    )
                    print(f"Added {len(existing_attrs)-1} traffic attributes to edges")
        
        # Sample a subgraph if there are too many nodes AND max_nodes is not None
        if max_nodes is not None and len(nodes_df) > max_nodes:
            print(f"Graph too large ({len(nodes_df)} nodes). Sampling subgraph...")
            nodes_df, edges_df = self.sample_node_subgraph(nodes_df, edges_df, max_nodes=max_nodes)
        
        self.nodes = nodes_df
        self.edges = edges_df
        return nodes_df, edges_df
    
    def create_temporal_snapshots(self, edges_df, interval_minutes=15, max_snapshots=None, max_edges_per_snapshot=100000):
        """
        Create temporal snapshots of the graph at regular intervals.
        
        Args:
            edges_df: DataFrame with edge data including timestamps
            interval_minutes: Time interval between snapshots
            max_snapshots: Maximum number of snapshots to create
            max_edges_per_snapshot: Maximum number of edges to keep per snapshot
            
        Returns:
            List of snapshot dictionaries
        """
        # Extract timestamps
        timestamps = edges_df['scrapedatetime'].unique()
        
        # Sort timestamps
        sorted_timestamps = np.sort(timestamps)
        
        # Create time intervals
        min_time = sorted_timestamps[0]
        max_time = sorted_timestamps[-1]
        
        # Check if timestamps are datetime-like
        if not isinstance(min_time, (datetime, np.datetime64, pd.Timestamp)):
            # Use integer indices if timestamps aren't datetime
            snapshots = []
            for i in tqdm(range(min(max_snapshots or 10, len(sorted_timestamps))), desc="Creating snapshots"):
                mask = edges_df['scrapedatetime'] == sorted_timestamps[i]
                snapshot_edges = edges_df[mask]
                
                # Sample edges if there are too many
                if max_edges_per_snapshot is not None and len(snapshot_edges) > max_edges_per_snapshot:
                    snapshot_edges = self.sample_important_edges(snapshot_edges, max_edges=max_edges_per_snapshot)
                
                snapshots.append({
                    'timestamp': i,
                    'edges': snapshot_edges,
                    'num_edges': len(snapshot_edges)
                })
            return snapshots
        
        # Convert numpy datetime64 to pandas Timestamp to handle timedelta addition properly
        min_time = pd.Timestamp(min_time)
        max_time = pd.Timestamp(max_time)
        
        # Create time intervals for datetime timestamps using pandas objects
        intervals = []
        current_time = min_time
        
        while current_time <= max_time:
            intervals.append(current_time)
            # Use pandas Timedelta instead of datetime.timedelta
            current_time += pd.Timedelta(minutes=interval_minutes)
            
            if max_snapshots and len(intervals) >= max_snapshots + 1:
                break
        
        # Create graph snapshots for each interval
        snapshots = []
        for i, t_start in tqdm(enumerate(intervals[:-1]), total=len(intervals)-1, desc="Creating snapshots"):
            t_end = intervals[i+1]
            
            # Filter edges for this time interval
            # Convert numpy datetime64 to pandas Timestamp for comparison
            if isinstance(t_start, (np.datetime64)):
                t_start = pd.Timestamp(t_start)
            if isinstance(t_end, (np.datetime64)):
                t_end = pd.Timestamp(t_end)
                
            mask = (edges_df['scrapedatetime'] >= t_start) & (edges_df['scrapedatetime'] < t_end)
            snapshot_edges = edges_df[mask]
            
            # Sample edges if there are too many
            if max_edges_per_snapshot is not None and len(snapshot_edges) > max_edges_per_snapshot:
                snapshot_edges = self.sample_important_edges(snapshot_edges, max_edges=max_edges_per_snapshot)
            
            if len(snapshot_edges) > 0:
                snapshots.append({
                    'timestamp': t_start,
                    'edges': snapshot_edges,
                    'num_edges': len(snapshot_edges)
                })
        
        return snapshots
    
    def prepare_tensor_data(self, snapshots, nodes_df, feature_cols=None):
        """
        Convert graph snapshots to tensor format for STGformer.
        
        Args:
            snapshots: List of graph snapshots
            nodes_df: DataFrame with node information
            feature_cols: List of edge feature columns to use
            
        Returns:
            X: Node feature tensor [T, N, C]
            A: List of sparse adjacency matrices
        """
        if not snapshots:
            print("No snapshots provided to prepare_tensor_data")
            return None, None
        
        # Default feature columns if not specified
        if feature_cols is None:
            # Try common traffic features including accident data
            feature_cols = []
            for col in ['speed', 'severity', 'level', 'delay', 'length', 
                        'is_accident_related', 'time_since_accident']:
                if col in snapshots[0]['edges'].columns:
                    feature_cols.append(col)
            
            if not feature_cols:
                print("No feature columns found or specified")
                return None, None
        
        # Get node mapping
        unique_nodes = nodes_df['node_id'].unique()
        node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
        num_nodes = len(unique_nodes)
        
        # Get number of time steps
        num_timesteps = len(snapshots)
        
        # Initialize feature tensor [T, N, C]
        num_features = len(feature_cols)
        X = np.zeros((num_timesteps, num_nodes, num_features))
        
        # Use a list of sparse tensors instead of a dense 3D tensor
        sparse_adjacency_list = []
        
        # Fill tensors from snapshots using tqdm for progress tracking
        for t, snapshot in tqdm(enumerate(snapshots), total=num_timesteps, desc="Processing snapshots"):
            try:
                edges = snapshot['edges']
                
                # Skip empty snapshots
                if len(edges) == 0:
                    # Add empty sparse tensor
                    indices = torch.LongTensor([[],[]])
                    values = torch.FloatTensor([])
                    # FIX: Use sparse_coo_tensor instead of sparse.FloatTensor
                    sparse_adjacency = torch.sparse_coo_tensor(
                        indices, values, (num_nodes, num_nodes)
                    )
                    sparse_adjacency_list.append(sparse_adjacency)
                    continue
                
                # Create sparse adjacency for this snapshot
                source_indices = []
                target_indices = []
                edge_values = []
                
                for _, edge in edges.iterrows():
                    source = edge['source']
                    target = edge['target']
                    
                    # Skip if source or target not in node mapping
                    if source not in node_to_idx or target not in node_to_idx:
                        continue
                    
                    i, j = node_to_idx[source], node_to_idx[target]
                    source_indices.append(i)
                    target_indices.append(j)
                    edge_values.append(1.0)  # Or any weight value
                    
                    # Add edge features to nodes
                    for f, feat in enumerate(feature_cols):
                        if feat in edge and not pd.isna(edge[feat]):
                            # Add feature to target node
                            X[t, j, f] = edge[feat]
                
                # Create sparse tensor
                if source_indices:  # Only if we have edges
                    indices = torch.LongTensor([source_indices, target_indices])
                    values = torch.FloatTensor(edge_values)
                    # FIX: Use sparse_coo_tensor instead of sparse.FloatTensor
                    sparse_adjacency = torch.sparse_coo_tensor(
                        indices, values, (num_nodes, num_nodes)
                    )
                else:
                    # Empty sparse tensor
                    indices = torch.LongTensor([[],[]])
                    values = torch.FloatTensor([])
                    # FIX: Use sparse_coo_tensor instead of sparse.FloatTensor
                    sparse_adjacency = torch.sparse_coo_tensor(
                        indices, values, (num_nodes, num_nodes)
                    )
                
                sparse_adjacency_list.append(sparse_adjacency)
            except Exception as e:
                print(f"Error processing snapshot {t}: {str(e)}")
                # Create an empty tensor as a fallback
                indices = torch.LongTensor([[],[]])
                values = torch.FloatTensor([])
                sparse_adjacency = torch.sparse_coo_tensor(
                    indices, values, (num_nodes, num_nodes)
                )
                sparse_adjacency_list.append(sparse_adjacency)
        
        # Convert features to tensor
        X_tensor = torch.FloatTensor(X)
        
        print(f"Created feature tensor with shape {X_tensor.shape} and {len(sparse_adjacency_list)} adjacency matrices")
        return X_tensor, sparse_adjacency_list

    def enhance_accident_features(self, edges_df):
        """
        Engineer additional accident-related features from binary data.
        
        Args:
            edges_df: DataFrame with basic accident features
            
        Returns:
            Enhanced DataFrame with derived accident features
        """
        enhanced_df = edges_df.copy()
        
        if 'is_accident_related' in enhanced_df.columns and 'time_since_accident' in enhanced_df.columns:
            # Create accident intensity based on recency
            # More recent accidents have higher intensity
            enhanced_df['accident_intensity'] = (
                enhanced_df['is_accident_related'] * 
                np.exp(-enhanced_df['time_since_accident'] / 60)  # Exponential decay over hours
            )
            
            # Create accident phase indicators
            enhanced_df['accident_fresh'] = (
                (enhanced_df['is_accident_related'] == 1) & 
                (enhanced_df['time_since_accident'] <= 30)
            ).astype(int)
            
            enhanced_df['accident_lingering'] = (
                (enhanced_df['is_accident_related'] == 1) & 
                (enhanced_df['time_since_accident'] > 30) & 
                (enhanced_df['time_since_accident'] <= 120)
            ).astype(int)
        
        return enhanced_df

    def sample_important_edges(self, edges_df, max_edges=100000):
        """
        Sample important edges to reduce memory usage.
        """
        # Check if sampling is needed
        if max_edges is None or len(edges_df) <= max_edges:
            return edges_df
        
        print(f"Sampling {max_edges} edges from {len(edges_df)} total edges...")
        
        try:
            if 'severity' in edges_df.columns:
                # Use pandas' built-in weighted sampling - much more efficient
                return edges_df.sample(max_edges, weights='severity', replace=False)
            elif 'speed' in edges_df.columns:
                # For speed, lower speeds should have higher weights
                # Create a weight column that's inversely proportional to speed
                with pd.option_context('mode.chained_assignment', None):
                    temp_df = edges_df.copy()
                    temp_df['weight'] = 1.0 / temp_df['speed'].clip(lower=0.1)
                    return temp_df.sample(max_edges, weights='weight', replace=False)
            else:
                # No weights available, use random sampling
                return edges_df.sample(max_edges)
        except Exception as e:
            print(f"Warning in edge sampling: {str(e)}. Using random sampling.")
            return edges_df.sample(max_edges)

    def sample_node_subgraph(self, nodes_df, edges_df, max_nodes=1000):
        """
        Sample a subgraph with a limited number of nodes to reduce memory usage.
        
        Args:
            nodes_df: DataFrame with node information
            edges_df: DataFrame with edge data
            max_nodes: Maximum number of nodes to include
            
        Returns:
            sampled_nodes_df, sampled_edges_df
        """
        # Check if sampling is needed
        if max_nodes is None or len(nodes_df) <= max_nodes:
            return nodes_df, edges_df
        
        print(f"Sampling a subgraph with {max_nodes} nodes from {len(nodes_df)} total nodes...")
        
        try:
            # Method 1: Sample based on connectivity (degree centrality)
            if len(edges_df) > 0:
                # Count connections for each node
                source_counts = edges_df['source'].value_counts()
                target_counts = edges_df['target'].value_counts()
                
                # Combine source and target counts
                all_counts = source_counts.add(target_counts, fill_value=0)
                
                # Select top nodes by connection count
                top_nodes = all_counts.nlargest(max_nodes).index
                
                sampled_nodes_df = nodes_df[nodes_df['node_id'].isin(top_nodes)].copy()
            else:
                # No edges, just randomly sample nodes
                sampled_nodes_df = nodes_df.sample(max_nodes).copy()
            
            # Filter edges to only include those between sampled nodes
            sampled_edges_df = edges_df[
                edges_df['source'].isin(sampled_nodes_df['node_id']) & 
                edges_df['target'].isin(sampled_nodes_df['node_id'])
            ].copy()
            
            print(f"Sampled subgraph has {len(sampled_nodes_df)} nodes and {len(sampled_edges_df)} edges")
            return sampled_nodes_df, sampled_edges_df
            
        except Exception as e:
            print(f"Error in subgraph sampling: {str(e)}. Falling back to random sampling.")
            # Fallback: Random node sampling
            sampled_node_ids = nodes_df['node_id'].sample(max_nodes).values
            sampled_nodes_df = nodes_df[nodes_df['node_id'].isin(sampled_node_ids)].copy()
            
            sampled_edges_df = edges_df[
                edges_df['source'].isin(sampled_nodes_df['node_id']) & 
                edges_df['target'].isin(sampled_nodes_df['node_id'])
            ].copy()
            
            return sampled_nodes_df, sampled_edges_df