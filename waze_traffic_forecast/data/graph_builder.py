"""
Graph construction from Waze data for traffic forecasting.
Builds nodes, edges, and temporal snapshots for graph transformer models.
"""
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import pandas as pd

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
            # Select key traffic attributes
            jam_attrs = ['id', 'severity', 'speed', 'length', 'delay', 'level']
            existing_attrs = [col for col in jam_attrs if col in jams_df.columns]
            
            if 'id' in existing_attrs and len(existing_attrs) > 1:
                # Rename id to jamid for merging if needed
                if 'jamid' not in jams_df.columns and 'id' in jams_df.columns:
                    jams_merge = jams_df[existing_attrs].copy()
                    jams_merge.rename(columns={'id': 'jamid'}, inplace=True)
                    
                    # Merge with edges
                    edges_df = edges_df.merge(
                        jams_merge,
                        on='jamid',
                        how='left'
                    )
                    print(f"Added {len(existing_attrs)-1} traffic attributes to edges")
        
        # Sample a subgraph if there are too many nodes
        if len(nodes_df) > max_nodes:
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
            for i in range(min(max_snapshots or 10, len(sorted_timestamps))):
                mask = edges_df['scrapedatetime'] == sorted_timestamps[i]
                snapshot_edges = edges_df[mask]
                
                # Sample edges if there are too many
                if len(snapshot_edges) > max_edges_per_snapshot:
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
        for i, t_start in enumerate(intervals[:-1]):
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
            if len(snapshot_edges) > max_edges_per_snapshot:
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
            return None, None
        
        # Default feature columns if not specified
        if feature_cols is None:
            # Try common traffic features
            feature_cols = []
            for col in ['speed', 'severity', 'level', 'delay', 'length']:
                if col in snapshots[0]['edges'].columns:
                    feature_cols.append(col)
            
            if not feature_cols:
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
        
        # Fill tensors from snapshots
        for t, snapshot in enumerate(snapshots):
            edges = snapshot['edges']
            
            # Skip empty snapshots
            if len(edges) == 0:
                # Add empty sparse tensor
                indices = torch.LongTensor([[],[]])
                values = torch.FloatTensor([])
                sparse_adjacency = torch.sparse.FloatTensor(
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
                sparse_adjacency = torch.sparse.FloatTensor(
                    indices, values, (num_nodes, num_nodes)
                )
            else:
                # Empty sparse tensor
                indices = torch.LongTensor([[],[]])
                values = torch.FloatTensor([])
                sparse_adjacency = torch.sparse.FloatTensor(
                    indices, values, (num_nodes, num_nodes)
                )
            
            sparse_adjacency_list.append(sparse_adjacency)
        
        # Convert features to tensor
        X_tensor = torch.FloatTensor(X)
        
        return X_tensor, sparse_adjacency_list

    def sample_important_edges(self, edges_df, max_edges=100000):
        """
        Sample important edges to reduce memory usage.
        
        Args:
            edges_df: DataFrame with edge data
            max_edges: Maximum number of edges to keep
            
        Returns:
            Sampled edges DataFrame
        """
        if len(edges_df) <= max_edges:
            return edges_df
        
        print(f"Sampling {max_edges} edges from {len(edges_df)} total edges...")
        
        if 'severity' in edges_df.columns:
            severity = edges_df['severity'].fillna(0) + 1e-5
            
            probs = severity / severity.sum()
            
            try:
                sampled_indices = np.random.choice(
                    edges_df.index, 
                    size=max_edges, 
                    replace=False, 
                    p=probs
                )
                return edges_df.loc[sampled_indices]
            except ValueError:
                print("Warning: Could not sample with probability weights. Using random sampling.")
                return edges_df.sample(max_edges)
        
        elif 'speed' in edges_df.columns:
            speed = edges_df['speed'].fillna(edges_df['speed'].mean())
            inverted_speed = 1 / (speed + 1e-5)  
            
            probs = inverted_speed / inverted_speed.sum()
            
            try:
                sampled_indices = np.random.choice(
                    edges_df.index, 
                    size=max_edges, 
                    replace=False, 
                    p=probs
                )
                return edges_df.loc[sampled_indices]
            except ValueError:
                print("Warning: Could not sample with probability weights. Using random sampling.")
                return edges_df.sample(max_edges)
        
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
        if len(nodes_df) <= max_nodes:
            return nodes_df, edges_df
        
        print(f"Sampling a subgraph with {max_nodes} nodes from {len(nodes_df)} total nodes...")
        
        if 'severity' in edges_df.columns:
            source_severity = edges_df.groupby('source')['severity'].mean().fillna(0)
            target_severity = edges_df.groupby('target')['severity'].mean().fillna(0)
            
            node_importance = pd.DataFrame(index=nodes_df['node_id'])
            node_importance['importance'] = 0
            
            for node_id in node_importance.index:
                importance = 0
                if node_id in source_severity:
                    importance += source_severity[node_id]
                if node_id in target_severity:
                    importance += target_severity[node_id]
                node_importance.loc[node_id, 'importance'] = importance
            
            top_nodes = node_importance.sort_values('importance', ascending=False).index[:max_nodes]
            sampled_nodes_df = nodes_df[nodes_df['node_id'].isin(top_nodes)].copy()
            
        elif len(edges_df) > 0:
            node_counts = defaultdict(int)
            for _, edge in edges_df.iterrows():
                node_counts[edge['source']] += 1
                node_counts[edge['target']] += 1
            
            if node_counts:
                start_node = max(node_counts.items(), key=lambda x: x[1])[0]
            else:
                start_node = nodes_df['node_id'].iloc[0]
            
            visited = set([start_node])
            queue = [start_node]
            
            node_connections = defaultdict(list)
            for _, edge in edges_df.iterrows():
                source, target = edge['source'], edge['target']
                node_connections[source].append(target)
                node_connections[target].append(source)  
            
            while queue and len(visited) < max_nodes:
                current = queue.pop(0)
                for neighbor in node_connections[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        if len(visited) >= max_nodes:
                            break
            
            sampled_nodes_df = nodes_df[nodes_df['node_id'].isin(visited)].copy()
            
        else:
            sampled_node_ids = nodes_df['node_id'].sample(max_nodes).values
            sampled_nodes_df = nodes_df[nodes_df['node_id'].isin(sampled_node_ids)].copy()
        
        sampled_edges_df = edges_df[
            edges_df['source'].isin(sampled_nodes_df['node_id']) & 
            edges_df['target'].isin(sampled_nodes_df['node_id'])
        ].copy()
        
        print(f"Sampled subgraph has {len(sampled_nodes_df)} nodes and {len(sampled_edges_df)} edges")
        return sampled_nodes_df, sampled_edges_df