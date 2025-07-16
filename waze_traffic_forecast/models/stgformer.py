"""
STGformer model implementation.

Based on the paper: "STGformer: Efficient Spatiotemporal Graph Transformer 
for Traffic Forecasting" by Hongjun Wang et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

from waze_traffic_forecast.models.layers import (
    GraphPropagation,
    SpatiotemporalAttention,
    TemporalPositionalEncoding
)

class STGformerLayer(nn.Module):
    """
    Single layer of STGformer combining graph propagation and spatiotemporal attention.
    """
    
    def __init__(self, hidden_dim, **kwargs):
        """
        Initialize a STGformer layer.
        
        Args:
            hidden_dim: Hidden dimension size
            **kwargs: Additional keyword arguments
        """
        super(STGformerLayer, self).__init__()
        
        # Extract parameters from kwargs with defaults
        num_heads = kwargs.get('num_heads', 8)
        dropout = kwargs.get('dropout', 0.1)
        propagation_steps = kwargs.get('propagation_steps', 3)
        use_layer_norm = kwargs.get('use_layer_norm', True)
        use_residual = kwargs.get('use_residual', True)
        
        # Graph propagation module
        self.graph_prop = GraphPropagation(
            hidden_dim, 
            hidden_dim, 
            K=propagation_steps, 
            activation=F.relu
        )
        
        # Spatiotemporal attention module
        self.st_attention = SpatiotemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Residual connection flag
        self.use_residual = use_residual
        
    def forward(self, x, adj_list):
        """
        Forward pass through the STGformer layer with accident-aware processing.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, hidden_dim]
            adj_list: Adjacency matrices - can be:
                - List of sparse matrices for each timestep
                - List of lists of sparse matrices (one list per batch item)
                - Dense tensor [batch_size, num_nodes, num_nodes]
                
        Returns:
            Output tensor of the same shape as input
        """
        batch_size, seq_len, num_nodes, hidden_dim = x.size()
        residual = x
        
        # Apply accident-aware attention if input has accident features
        # Assume last 2 features are accident-related if hidden_dim >= 6
        if hidden_dim >= 6:
            # Extract accident information from input features
            accident_flag = x[:, :, :, -2]      # is_accident_related
            time_since = x[:, :, :, -1]         # time_since_accident
            
            # Create accident attention mask
            # Recent accidents get higher attention weight
            accident_attention = accident_flag * torch.exp(-time_since / 60.0)
            accident_attention = accident_attention.unsqueeze(-1)
            
            # Apply accident-aware weighting
            x = x * (1.0 + 0.3 * accident_attention)  # Boost accident areas by 30%
        
        # Apply graph propagation separately for each time step
        prop_out = []
        
        sparse_mode = isinstance(adj_list, list)
        
        for t in range(seq_len):
            # Get adjacency matrix for this timestep
            if sparse_mode:
                # Handle list of sparse matrices or list of lists
                if isinstance(adj_list[0], list):
                    # We have a list of lists [batch][time]
                    # Extract the adjacency matrices for this timestep across all batches
                    adj_t = []
                    for batch_item in adj_list:
                        if t < len(batch_item):
                            adj_t.append(batch_item[t])
                        else:
                            # Use last available timestep if out of bounds
                            adj_t.append(batch_item[-1])
                else:
                    if t < len(adj_list):
                        adj_t = adj_list[t]
                    else:
                        adj_t = adj_list[-1]
            else:
                # Dense tensor case
                adj_t = adj_list
            
            t_out = self.graph_prop(x[:, t], adj_t)
            prop_out.append(t_out)
        
        prop_out = torch.stack(prop_out, dim=1)  
        
        if self.use_residual:
            prop_out = prop_out + residual
            
        attn_out = self.st_attention(prop_out)
        
        # Apply feed-forward network with layer norm if specified
        if self.use_layer_norm:
            ffn_out = self.norm1(attn_out)
            ffn_out = self.ffn(ffn_out)
            if self.use_residual:
                ffn_out = ffn_out + attn_out
            out = self.norm2(ffn_out)
        else:
            ffn_out = self.ffn(attn_out)
            if self.use_residual:
                ffn_out = ffn_out + attn_out
            out = ffn_out
        
        return out


class STGformer(nn.Module):
    """
    STGformer: Efficient Spatiotemporal Graph Transformer for Traffic Forecasting.
    """
    
    def __init__(self, in_channels, out_channels, num_nodes, **kwargs):
        """
        Initialize the STGformer model.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            num_nodes: Number of nodes in the graph
            **kwargs: Additional keyword arguments
        """
        super(STGformer, self).__init__()
        
        # Extract parameters from kwargs with defaults
        hidden_dim = kwargs.get('hidden_channels', 64)
        num_layers = kwargs.get('num_layers', 3)
        num_heads = kwargs.get('num_heads', 8)
        dropout = kwargs.get('dropout', 0.1)
        propagation_steps = kwargs.get('propagation_steps', 3)
        use_layer_norm = kwargs.get('use_layer_norm', True)
        use_residual = kwargs.get('use_residual', True)
        self.seq_length = kwargs.get('sequence_length', 12)
        self.pred_horizon = kwargs.get('prediction_horizon', 3)
        
        # Input embedding
        self.input_embed = nn.Linear(in_channels, hidden_dim)
        
        # Temporal positional encoding
        self.temporal_pe = TemporalPositionalEncoding(hidden_dim)
        
        # STGformer layers
        self.layers = nn.ModuleList([
            STGformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                propagation_steps=propagation_steps,
                use_layer_norm=use_layer_norm,
                use_residual=use_residual
            ) for _ in range(num_layers)
        ])
        
        # Time-wise fully connected layers (prediction for each future time step)
        self.output_layer = nn.ModuleList([
            nn.Linear(hidden_dim, out_channels) for _ in range(self.pred_horizon)
        ])
        
    def forward(self, x_seq, adj):
        """
        Forward pass through the STGformer.
        
        Args:
            x_seq: Input sequence [batch_size, seq_len, num_nodes, in_channels]
            adj: Adjacency matrices - can be:
                - List of sparse matrices
                - List of lists of sparse matrices (one list per batch item)
                - Dense tensor [batch_size, seq_len, num_nodes, num_nodes]
                
        Returns:
            Predictions for future time steps [batch_size, pred_horizon, num_nodes, out_channels]
        """
        h = self.input_embed(x_seq)  
        
        h = self.temporal_pe(h)
        
        batch_size = x_seq.size(0)
        
        if isinstance(adj, list) and isinstance(adj[0], list):
            adj_batch = adj
        
        elif isinstance(adj, list) and hasattr(adj[0], 'is_sparse') and adj[0].is_sparse:
            adj_batch = [adj] * batch_size
        
        else:
            adj_batch = adj
        
        for layer in self.layers:
            h = layer(h, adj_batch)
        
        h_last = h[:, -1]
        
        predictions = []
        for t in range(self.pred_horizon):
            pred_t = self.output_layer[t](h_last) 
            predictions.append(pred_t)
        
        out = torch.stack(predictions, dim=1)  
        
        return out
    
    def get_loss(self, pred, target, mask=None):
        """
        Calculate loss between predictions and targets.
        
        Args:
            pred: Predictions from the model [batch_size, pred_horizon, num_nodes, out_channels]
            target: Ground truth values [batch_size, pred_horizon, num_nodes, out_channels]
            mask: Optional mask tensor for valid values
            
        Returns:
            Loss value
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)  
            pred = pred * mask
            target = target * mask
            loss = F.mse_loss(pred, target, reduction='sum')
            num_valid = mask.sum()
            loss = loss / (num_valid + 1e-6)
        else:
            loss = F.mse_loss(pred, target)
        
        return loss
    
    def get_accident_aware_loss(self, pred, target, accident_mask=None):
        """
        Calculate loss with higher weight on accident-affected areas.
        
        Args:
            pred: Predictions [batch_size, pred_horizon, num_nodes, features]
            target: Ground truth [batch_size, pred_horizon, num_nodes, features]
            accident_mask: Binary mask for accident-affected nodes
            
        Returns:
            Weighted loss value
        """
        base_loss = F.mse_loss(pred, target, reduction='none')
        
        if accident_mask is not None:
            # Apply higher weight to accident-affected areas
            weights = 1.0 + 2.0 * accident_mask.unsqueeze(-1)  # 3x weight for accident areas
            weighted_loss = base_loss * weights
            return weighted_loss.mean()
        
        return base_loss.mean()


class STGformerModel:
    """
    High-level wrapper for the STGformer model with training and inference methods.
    """
    
    def __init__(self, config):
        """
        Initialize the STGformer model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = config['training']['device']
        
    def build_model(self, in_channels, out_channels, num_nodes):
        """
        Build the STGformer model.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            num_nodes: Number of nodes in the graph
            
        Returns:
            Built model
        """
        model_config = self.config['model']
        data_config = self.config['data']
        
        self.model = STGformer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_nodes=num_nodes,
            hidden_channels=model_config['hidden_channels'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout'],
            use_layer_norm=model_config['use_layer_norm'],
            use_residual=model_config['use_residual'],
            sequence_length=data_config['sequence_length'],
            prediction_horizon=data_config['prediction_horizon']
        )
        
        self.model = self.model.to(self.device)
        
        training_config = self.config['training']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        if training_config['lr_scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config['lr_scheduler_params']['step_size'],
                gamma=training_config['lr_scheduler_params']['gamma']
            )
        elif training_config['lr_scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs']
            )
        elif training_config['lr_scheduler'] == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=training_config['lr_scheduler_params']['factor'],
                patience=training_config['lr_scheduler_params']['patience'],
                verbose=True
            )
        
        return self.model
    
    def train_step(self, x_seq, adj, y_true, mask=None):
        """
        Perform one training step.
        
        Args:
            x_seq: Input sequence [batch_size, seq_len, num_nodes, in_channels]
            adj: Adjacency matrices (can be in various formats)
            y_true: Ground truth values [batch_size, pred_horizon, num_nodes, out_channels]
            mask: Optional mask tensor for valid values
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        x_seq = x_seq.to(self.device)
        
        if isinstance(adj, list):
            if isinstance(adj[0], list):
                adj_device = []
                for batch_item in adj:
                    batch_item_device = []
                    for sparse_mat in batch_item:
                        batch_item_device.append(sparse_mat.to(self.device))
                    adj_device.append(batch_item_device)
            else:
                adj_device = [sparse_mat.to(self.device) for sparse_mat in adj]
        else:
            adj_device = adj.to(self.device)
        
        y_true = y_true.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        y_pred = self.model(x_seq, adj_device)
        
        loss = self.model.get_loss(y_pred, y_true, mask)
        
        loss.backward()
        
        if self.config['training']['grad_clip_value'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip_value']
            )
        
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader, return_predictions=False):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader with evaluation data
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                x_seq = batch['x_seq'].to(self.device)
                adj = batch['a_seq'][-1].to(self.device) 
                y_true = batch['y_seq'].to(self.device)
                mask = batch.get('mask')
                if mask is not None:
                    mask = mask.to(self.device)
                
                y_pred = self.model(x_seq, adj)
                
                loss = self.model.get_loss(y_pred, y_true, mask)
                
                batch_size = x_seq.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                if return_predictions:
                    all_predictions.append(y_pred.detach().cpu())
                    all_targets.append(y_true.detach().cpu())
        
        avg_loss = total_loss / max(1, total_samples)
        
        result = {'loss': avg_loss}
        
        if return_predictions:
            result['predictions'] = torch.cat(all_predictions, dim=0)
            result['targets'] = torch.cat(all_targets, dim=0)
        
        return result
    
    def predict(self, x_seq, adj):
        """
        Make predictions with the model.
        
        Args:
            x_seq: Input sequence [batch_size, seq_len, num_nodes, in_channels]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Predictions for future time steps
        """
        self.model.eval()
        with torch.no_grad():
            x_seq = x_seq.to(self.device)
            adj = adj.to(self.device)
            predictions = self.model(x_seq, adj)
        
        return predictions.cpu()
    
    def save_checkpoint(self, path, epoch=None, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch
            **kwargs: Additional information to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': epoch,
            **kwargs
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path, **kwargs):
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint
            **kwargs: Additional keyword arguments
        
        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if not kwargs.get('skip_optimizer', False) and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if not kwargs.get('skip_scheduler', False) and 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint