"""
Custom layers for graph transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphPropagation(nn.Module):
    """
    Graph propagation module for spatial message passing.
    Based on simplified graph convolution inspired by Chebyshev polynomial approximation.
    """
    
    def __init__(self, in_channels, out_channels, K=3, **kwargs):
        """
        Initialize the graph propagation layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            K: Order of Chebyshev polynomial (hop steps)
            **kwargs: Additional keyword arguments
        """
        super(GraphPropagation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        
        # Linear transformations for each order
        self.weights = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(K+1)
        ])
        
        # Activation function
        self.activation = kwargs.get('activation', F.relu)
        
    def forward(self, x, adj):
        """
        Forward pass through the graph propagation layer.
        
        Args:
            x: Node features tensor [batch_size, num_nodes, in_channels]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Output tensor [batch_size, num_nodes, out_channels]
        """
        batch_size, num_nodes, _ = x.size()
        
        # Identity matrix
        identity = torch.eye(num_nodes, device=x.device)
        if batch_size > 1:
            identity = identity.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Normalized Laplacian
        degree = torch.sum(adj, dim=-1)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
        
        laplacian = identity - torch.bmm(
            torch.bmm(degree_inv_sqrt, adj),
            degree_inv_sqrt
        )
        
        # Scaled Laplacian for numerical stability
        lambda_max = 2.0  # Maximum eigenvalue of normalized Laplacian
        scaled_laplacian = (2.0 * laplacian / lambda_max) - identity
        
        # Chebyshev polynomial approximation
        Tx_0 = x  # Order 0
        out = self.weights[0](Tx_0)
        
        if self.K >= 1:
            Tx_1 = torch.bmm(scaled_laplacian, x)  # Order 1
            out = out + self.weights[1](Tx_1)
            
        for k in range(2, self.K + 1):
            # Recurrence relation: T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
            Tx_k = 2 * torch.bmm(scaled_laplacian, Tx_1) - Tx_0
            out = out + self.weights[k](Tx_k)
            Tx_0, Tx_1 = Tx_1, Tx_k
        
        if self.activation is not None:
            out = self.activation(out)
        
        return out


class SpatiotemporalAttention(nn.Module):
    """
    Unified spatiotemporal attention module that captures both spatial and temporal patterns.
    Uses linear attention for efficiency on large graphs.
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, **kwargs):
        """
        Initialize the spatiotemporal attention layer.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            dropout: Dropout probability
            **kwargs: Additional keyword arguments
        """
        super(SpatiotemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        # Linear projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # Layer normalization if specified
        self.use_layer_norm = kwargs.get('use_layer_norm', True)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Residual connection if specified
        self.use_residual = kwargs.get('use_residual', True)
        
    def _linearized_attention(self, q, k, v, mask=None):
        """
        Compute efficient linear attention.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention output
        """
        # Apply ELU activation and add 1 for positive values
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention: instead of computing (Q*K^T)*V, compute Q*(K^T*V)
        # This reduces complexity from O(N^2) to O(N)
        if mask is not None:
            k = k * mask.unsqueeze(-1)
            
        kv = torch.einsum("...nd,...ne->...de", k, v)
        out = torch.einsum("...nd,...de->...ne", q, kv)
        
        # Normalize by the sum of attention weights
        normalizer = torch.einsum("...nd,...d->...n", q, k.sum(dim=-2))
        out = out / (normalizer.unsqueeze(-1) + 1e-6)
        
        return out
    
    def forward(self, x, mask=None):
        """
        Forward pass through the spatiotemporal attention layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor of the same shape as input
        """
        batch_size, seq_len, num_nodes, _ = x.size()
        residual = x
        
        # Apply layer normalization if specified
        if self.use_layer_norm:
            x = self.layer_norm(x)
        
        # Reshape for spatiotemporal attention
        # Merge sequence and nodes for unified spatiotemporal attention
        x = x.reshape(batch_size, seq_len * num_nodes, self.hidden_dim)
        
        # Linear projections and reshape for multi-head attention
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape mask if provided
        if mask is not None:
            mask = mask.view(batch_size, -1)
        
        # Apply linearized attention
        attn_out = self._linearized_attention(q, k, v, mask)
        
        # Combine heads and apply output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len * num_nodes, self.hidden_dim
        )
        out = self.out_proj(attn_out)
        out = self.out_dropout(out)
        
        # Reshape back to [batch_size, seq_len, num_nodes, hidden_dim]
        out = out.view(batch_size, seq_len, num_nodes, self.hidden_dim)
        
        # Add residual connection if specified
        if self.use_residual:
            out = out + residual
        
        return out


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding for sequence data.
    """
    
    def __init__(self, hidden_dim, max_len=1000, **kwargs):
        """
        Initialize temporal positional encoding.
        
        Args:
            hidden_dim: Dimension of embeddings
            max_len: Maximum sequence length
            **kwargs: Additional keyword arguments
        """
        super(TemporalPositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-np.log(10000.0) / hidden_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, hidden_dim]
        
        # Register as buffer (not parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, hidden_dim]
            
        Returns:
            Output tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]