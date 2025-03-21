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
    
    def build_graph_structure(self, segments_df, jams_df=None):
        """
        Build the basic graph structure from segments data.
        
        Args:
            segments_df: DataFrame with jam segments
            jams_df: DataFrame with jam attributes
            
        Returns:
            nodes_df, edges_df
        """
        from_nodes = segments_df['fromnode'].unique()
        to_nodes = segments_df['tonode'].unique()
        unique_nodes = np.unique(np.concatenate([from_nodes, to_nodes]))
        
        nodes_df = pd.DataFrame({'node_id': unique_nodes})
        
        edges_df = segments_df[['fromnode', 'tonode', 'jamid', 'scrapedatetime']].copy()
        edges_df.rename(columns={'fromnode': 'source', 'tonode': 'target'}, inplace=True)
        
        if jams_df is not None and len(jams_df) > 0:
            jam_attrs = ['id', 'severity', 'speed', 'length', 'delay', 'level']
            existing_attrs = [col for col in jam_attrs if col in jams_df.columns]
            
            if 'id' in existing_attrs and len(existing_attrs) > 1:
                if 'jamid' not in jams_df.columns and 'id' in jams_df.columns:
                    jams_merge = jams_df[existing_attrs].copy()
                    jams_merge.rename(columns={'id': 'jamid'}, inplace=True)
                    
                    edges_df = edges_df.merge(
                        jams_merge,
                        on='jamid',
                        how='left'
                    )
        
        self.nodes = nodes_df
        self.edges = edges_df
        return nodes_df, edges_df
    
    def create_temporal_snapshots(self, edges_df, interval_minutes=15, max_snapshots=None):
        """
        Create temporal snapshots of the graph at regular intervals.
        
        Args:
            edges_df: DataFrame with edge data including timestamps
            interval_minutes: Time interval between snapshots
            max_snapshots: Maximum number of snapshots to create
            
        Returns:
            List of snapshot dictionaries
        """
        timestamps = edges_df['scrapedatetime'].unique()
        
        sorted_timestamps = np.sort(timestamps)
        
        min_time = sorted_timestamps[0]
        max_time = sorted_timestamps[-1]
        
        if not isinstance(min_time, (datetime, np.datetime64, pd.Timestamp)):
            snapshots = []
            for i in range(min(max_snapshots or 10, len(sorted_timestamps))):
                mask = edges_df['scrapedatetime'] == sorted_timestamps[i]
                snapshots.append({
                    'timestamp': i,
                    'edges': edges_df[mask],
                    'num_edges': sum(mask)
                })
            return snapshots
        
        min_time = pd.Timestamp(min_time)
        max_time = pd.Timestamp(max_time)
        
        intervals = []
        current_time = min_time
        
        while current_time <= max_time:
            intervals.append(current_time)
            current_time += pd.Timedelta(minutes=interval_minutes)
            
            if max_snapshots and len(intervals) >= max_snapshots + 1:
                break
        
        snapshots = []
        for i, t_start in enumerate(intervals[:-1]):
            t_end = intervals[i+1]
            
            if isinstance(t_start, (np.datetime64)):
                t_start = pd.Timestamp(t_start)
            if isinstance(t_end, (np.datetime64)):
                t_end = pd.Timestamp(t_end)
                
            mask = (edges_df['scrapedatetime'] >= t_start) & (edges_df['scrapedatetime'] < t_end)
            snapshot_edges = edges_df[mask]
            
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
            A: Adjacency tensor [T, N, N]
        """
        if not snapshots:
            return None, None
        
        if feature_cols is None:
            feature_cols = []
            for col in ['speed', 'severity', 'level', 'delay', 'length']:
                if col in snapshots[0]['edges'].columns:
                    feature_cols.append(col)
            
            if not feature_cols:
                return None, None
        
        unique_nodes = nodes_df['node_id'].unique()
        node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
        num_nodes = len(unique_nodes)
        
        num_timesteps = len(snapshots)
        
        num_features = len(feature_cols)
        X = np.zeros((num_timesteps, num_nodes, num_features))
        
        A = np.zeros((num_timesteps, num_nodes, num_nodes))
        
        for t, snapshot in enumerate(snapshots):
            edges = snapshot['edges']
            
            if len(edges) == 0:
                continue
            
            for _, edge in edges.iterrows():
                source = edge['source']
                target = edge['target']
                
                if source not in node_to_idx or target not in node_to_idx:
                    continue
                
                i, j = node_to_idx[source], node_to_idx[target]
                A[t, i, j] = 1
                
                for f, feat in enumerate(feature_cols):
                    if feat in edge and not pd.isna(edge[feat]):
                        X[t, j, f] = edge[feat]
        
        X_tensor = torch.FloatTensor(X)
        A_tensor = torch.FloatTensor(A)
        
        return X_tensor, A_tensor