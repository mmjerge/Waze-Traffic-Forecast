"""
Base Graph Transformer model implementation.

This module provides a more general Graph Transformer architecture that can be
extended for various graph-based prediction tasks.
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


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer Encoder that combines graph structure with transformer architecture.
    """
    
    def __init__(self, hidden_dim, **kwargs):
        """
        Initialize a Graph Transformer Encoder.
        
        Args:
            hidden_dim: Hidden dimension size
            **kwargs: Additional keyword arguments
        """
        super(GraphTransformerEncoder, self).__init__()
        
        # Extract parameters from kwargs with defaults
        num_layers = kwargs.get('num_layers', 3)
        num_heads = kwargs.get('num_heads', 8)
        dropout = kwargs.get('dropout', 0.1)
        attention_dropout = kwargs.get('attention_dropout', 0.1)
        propagation_steps = kwargs.get('propagation_steps', 3)
        use_layer_norm = kwargs.get('use_layer_norm', True)
        use_residual = kwargs.get('use_residual', True)
        
        # Input embedding and positional encoding
        self.input_embed = nn.Linear(kwargs.get('in_channels', hidden_dim), hidden_dim)
        self.temporal_pe = TemporalPositionalEncoding(hidden_dim)
        
        # Graph propagation layers (one for each transformer layer)
        self.graph_prop_layers = nn.ModuleList([
            GraphPropagation(
                hidden_dim, 
                hidden_dim, 
                K=propagation_steps
            ) for _ in range(num_layers)
        ])
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            SpatiotemporalAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
                use_layer_norm=use_layer_norm,
                use_residual=use_residual
            ) for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
            ])
        
        # Residual connection flag
        self.use_residual = use_residual
        
    def forward(self, x, adj, mask=None, **kwargs):
        """
        Forward pass through the Graph Transformer Encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
            mask: Optional attention mask
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor after encoding
        """
        # Embed input if dimensions don't match
        if x.size(-1) != self.input_embed.weight.size(1):
            x = self.input_embed(x)
        
        # Add temporal positional encoding
        x = self.temporal_pe(x)
        
        # Pass through each layer
        for i, (graph_prop, attention, ff) in enumerate(
            zip(self.graph_prop_layers, self.attention_layers, self.ff_layers)
        ):
            # Apply graph propagation separately for each time step
            batch_size, seq_len, num_nodes, hidden_dim = x.size()
            prop_out = []
            for t in range(seq_len):
                t_out = graph_prop(x[:, t], adj)  # [batch_size, num_nodes, hidden_dim]
                prop_out.append(t_out)
            
            # Stack back into sequence
            prop_out = torch.stack(prop_out, dim=1)  # [batch_size, seq_len, num_nodes, hidden_dim]
            
            # Apply residual connection after graph propagation if specified
            if self.use_residual:
                prop_out = prop_out + x
            
            # Apply layer norm if specified
            if self.use_layer_norm:
                prop_out = self.norm_layers[i*2](prop_out)
            
            # Apply attention
            attn_out = attention(prop_out, mask)
            
            # Apply feed-forward network with layer norm if specified
            if self.use_layer_norm:
                ff_in = self.norm_layers[i*2+1](attn_out)
                ff_out = ff(ff_in)
                if self.use_residual:
                    x = ff_out + attn_out
                else:
                    x = ff_out
            else:
                ff_out = ff(attn_out)
                if self.use_residual:
                    x = ff_out + attn_out
                else:
                    x = ff_out
        
        return x


class GraphTransformerDecoder(nn.Module):
    """
    Graph Transformer Decoder for generating predictions over future time steps.
    """
    
    def __init__(self, hidden_dim, out_channels, pred_horizon, **kwargs):
        """
        Initialize a Graph Transformer Decoder.
        
        Args:
            hidden_dim: Hidden dimension size
            out_channels: Number of output channels
            pred_horizon: Number of future time steps to predict
            **kwargs: Additional keyword arguments
        """
        super(GraphTransformerDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.pred_horizon = pred_horizon
        
        # Extract parameters from kwargs with defaults
        decoder_type = kwargs.get('decoder_type', 'mlp')
        
        if decoder_type == 'mlp':
            # Simple MLP decoder (one for each prediction step)
            self.output_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, out_channels)
                ) for _ in range(pred_horizon)
            ])
        elif decoder_type == 'rnn':
            # RNN-based decoder
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            )
            self.output_layer = nn.Linear(hidden_dim, out_channels)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        
        self.decoder_type = decoder_type
        
    def forward(self, x, **kwargs):
        """
        Forward pass through the Graph Transformer Decoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, hidden_dim]
            **kwargs: Additional keyword arguments
            
        Returns:
            Predictions for future time steps [batch_size, pred_horizon, num_nodes, out_channels]
        """
        batch_size, seq_len, num_nodes, hidden_dim = x.size()
        
        if self.decoder_type == 'mlp':
            # Use the last time step as context for prediction
            context = x[:, -1]  # [batch_size, num_nodes, hidden_dim]
            
            # Generate predictions for each future time step
            predictions = []
            for i in range(self.pred_horizon):
                pred_t = self.output_layers[i](context)  # [batch_size, num_nodes, out_channels]
                predictions.append(pred_t)
            
            # Stack predictions along time dimension
            predictions = torch.stack(predictions, dim=1)  # [batch_size, pred_horizon, num_nodes, out_channels]
            
        elif self.decoder_type == 'rnn':
            # Reshape for RNN
            x_rnn = x.reshape(batch_size * num_nodes, seq_len, hidden_dim)
            
            # Initial hidden state
            h0 = torch.zeros(1, batch_size * num_nodes, hidden_dim, device=x.device)
            
            # RNN forward pass
            output, _ = self.rnn(x_rnn, h0)
            
            # Use the last output as seed
            last_output = output[:, -1].unsqueeze(1)  # [batch_size*num_nodes, 1, hidden_dim]
            
            # Autoregressive generation
            outputs = [last_output]
            
            for _ in range(self.pred_horizon - 1):
                next_output, _ = self.rnn(outputs[-1])
                outputs.append(next_output)
            
            # Concatenate all outputs
            decoder_outputs = torch.cat(outputs[1:], dim=1)  # [batch_size*num_nodes, pred_horizon, hidden_dim]
            
            # Reshape back
            decoder_outputs = decoder_outputs.reshape(batch_size, num_nodes, self.pred_horizon, hidden_dim)
            decoder_outputs = decoder_outputs.permute(0, 2, 1, 3)  # [batch_size, pred_horizon, num_nodes, hidden_dim]
            
            # Project to output channels
            predictions = self.output_layer(decoder_outputs)  # [batch_size, pred_horizon, num_nodes, out_channels]
        
        return predictions


class GraphTransformer(nn.Module):
    """
    Complete Graph Transformer model for spatiotemporal prediction.
    """
    
    def __init__(self, in_channels, out_channels, num_nodes, **kwargs):
        """
        Initialize the Graph Transformer model.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            num_nodes: Number of nodes in the graph
            **kwargs: Additional keyword arguments
        """
        super(GraphTransformer, self).__init__()
        
        # Extract parameters from kwargs with defaults
        hidden_dim = kwargs.get('hidden_channels', 64)
        num_encoder_layers = kwargs.get('num_encoder_layers', 3)
        num_decoder_layers = kwargs.get('num_decoder_layers', 2)
        num_heads = kwargs.get('num_heads', 8)
        dropout = kwargs.get('dropout', 0.1)
        attention_dropout = kwargs.get('attention_dropout', 0.1)
        propagation_steps = kwargs.get('propagation_steps', 3)
        use_layer_norm = kwargs.get('use_layer_norm', True)
        use_residual = kwargs.get('use_residual', True)
        pred_horizon = kwargs.get('prediction_horizon', 3)
        decoder_type = kwargs.get('decoder_type', 'mlp')
        
        # Input embedding
        self.input_embed = nn.Linear(in_channels, hidden_dim)
        
        # Encoder
        self.encoder = GraphTransformerEncoder(
            hidden_dim=hidden_dim,
            in_channels=hidden_dim,  # After input embedding
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            propagation_steps=propagation_steps,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual
        )
        
        # Decoder
        self.decoder = GraphTransformerDecoder(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            pred_horizon=pred_horizon,
            decoder_type=decoder_type
        )
        
    def forward(self, x_seq, adj, **kwargs):
        """
        Forward pass through the Graph Transformer.
        
        Args:
            x_seq: Input sequence [batch_size, seq_len, num_nodes, in_channels]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
            **kwargs: Additional keyword arguments
            
        Returns:
            Predictions for future time steps [batch_size, pred_horizon, num_nodes, out_channels]
        """
        # Embed input
        x_embed = self.input_embed(x_seq)
        
        # Encoder
        encoder_output = self.encoder(x_embed, adj, **kwargs)
        
        # Decoder
        predictions = self.decoder(encoder_output, **kwargs)
        
        return predictions
    
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
            mask = mask.unsqueeze(-1)  # Add feature dimension
            # Apply mask
            pred = pred * mask
            target = target * mask
            loss = F.mse_loss(pred, target, reduction='sum')
            # Normalize by the number of valid elements
            num_valid = mask.sum()
            loss = loss / (num_valid + 1e-6)
        else:
            loss = F.mse_loss(pred, target)
        
        return loss