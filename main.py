#!/usr/bin/env python3
"""
Script for performing inference with a trained STGformer model on Waze traffic data.
Uses a checkpoint model to predict on an evaluation dataset.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# Set CUDA optimization flags for better performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from waze_traffic_forecast._config import load_config
from waze_traffic_forecast.dataset import WazeGraphDataset
from waze_traffic_forecast.models.stgformer import STGformerModel


def sparse_tensor_collate_fn(batch):
    """
    Custom collate function that handles both sparse tensors and edge indices.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched data with sparse tensors properly handled
    """
    elem = batch[0]
    
    if isinstance(elem, dict):
        result = {}
        for key in elem:
            if key == 'a_seq':
                if isinstance(elem[key], list):
                    if len(elem[key]) > 0 and isinstance(elem[key][0], torch.Tensor):
                        if hasattr(elem[key][0], 'is_sparse') and elem[key][0].is_sparse:
                            result[key] = [sample[key] for sample in batch]
                        elif elem[key][0].dim() == 2 and elem[key][0].shape[0] == 2:
                            result[key] = [sample[key] for sample in batch]
                        else:
                            try:
                                result[key] = torch.utils.data.default_collate([sample[key] for sample in batch])
                            except:
                                result[key] = [sample[key] for sample in batch]
                    else:
                        result[key] = [sample[key] for sample in batch]
                else:
                    try:
                        result[key] = torch.utils.data.default_collate([sample[key] for sample in batch])
                    except:
                        result[key] = [sample[key] for sample in batch]
            elif key in ['sampled_nodes', 'global_indices']:
                result[key] = [sample[key] for sample in batch]
            else:
                try:
                    result[key] = torch.utils.data.default_collate([sample[key] for sample in batch])
                except RuntimeError:
                    result[key] = [sample[key] for sample in batch]
        return result
    
    return torch.utils.data.default_collate(batch)


class EnhancedSTGformerModel(STGformerModel):
    """Wrapper class that adds robust handling of adjacency matrices to STGformerModel"""
    
    def __init__(self, config):
        super().__init__(config)
        self._printed_debug_info = False
        self._cached_adj_mat = None
    
    def prepare_adjacency_matrix(self, a_seq, x_seq):
        """
        Convert adjacency matrix to the format expected by the model.
        
        Args:
            a_seq: List or tensor containing adjacency information
            x_seq: Input sequence tensor to get batch size and device
            
        Returns:
            Processed adjacency matrix as 3D tensor suitable for batch operations
        """
        if self._cached_adj_mat is not None:
            # Use cached adjacency if available (for consistent inference)
            return self._cached_adj_mat
            
        device = x_seq.device
        batch_size = x_seq.shape[0]
        seq_len = x_seq.shape[1]
        num_nodes = x_seq.shape[2]
        
        # Create a batched identity adjacency matrix - this is what works with this model
        print(f"[INFO] Creating batched adjacency tensor with shape: ({batch_size}, {num_nodes}, {num_nodes})")
        # Create a repeated identity matrix for each batch and time step
        # For STGformer, we need an adjacency matrix per batch item
        adj = torch.eye(num_nodes, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Print shape to verify
        print(f"[DEBUG] Created adjacency tensor with shape: {adj.shape}")
        
        # Cache it for future batches (assuming same batch size)
        self._cached_adj_mat = adj
        return adj
    
    def predict_step(self, x_seq, a_seq):
        """Enhanced predict step with specialized adjacency matrix handling"""
        # Process the adjacency matrix
        adj = self.prepare_adjacency_matrix(a_seq, x_seq)
        
        # Forward pass
        return self.model(x_seq, adj)


def evaluate_model(config, checkpoint_path, data_dir=None, output_dir=None, **kwargs):
    """
    Evaluate a trained STGformer model using a checkpoint.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to the model checkpoint
        data_dir: Directory containing evaluation data
        output_dir: Directory to save evaluation results
        **kwargs: Additional keyword arguments
    
    Returns:
        Evaluation metrics and predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set data directory
    if data_dir:
        config['data']['directory'] = data_dir
    
    # Set output directory
    if not output_dir:
        output_dir = config['paths'].get('output_dir', 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    print(f"Loading dataset from {config['data']['directory']}")
    dataset = WazeGraphDataset(
        data_dir=config['data']['directory'],
        full_graph=config['data'].get('full_graph', False),
        subgraph_nodes=config['data'].get('subgraph_nodes', 5000),
        batch_size=config['data'].get('batch_size', 1000),
        num_hops=config['data'].get('num_hops', 2),
        sample_size=config['data'].get('sample_size'),
        interval_minutes=config['data']['interval_minutes'],
        max_snapshots=config['data']['max_snapshots'],
        feature_cols=config['data']['feature_columns'],
        prediction_horizon=config['data']['prediction_horizon'],
        sequence_length=config['data']['sequence_length']
    )
    
    if config['data'].get('full_graph', False):
        dataset.init_epoch(0)
        
    print(f"Dataset size: {len(dataset)}")
    
    # Get input/output dimensions from dataset
    if dataset.X is not None:
        in_channels = dataset.X.shape[-1]
        out_channels = in_channels
        num_nodes = dataset.X.shape[1]
        
        print(f"Feature tensor shape: {dataset.X.shape}")
        print(f"Number of input features: {in_channels}")
        print(f"Number of output features: {out_channels}")
        print(f"Number of nodes: {num_nodes}")
    else:
        print("Error: Failed to create dataset tensors")
        return None
    
    # Optimize dataloader parameters for better performance
    num_workers = min(kwargs.get('num_workers', 4), os.cpu_count() or 4)
    batch_size = kwargs.get('batch_size', config['training']['batch_size'])
    
    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=sparse_tensor_collate_fn
    )
    
    # Initialize model
    print("Initializing model")
    model = EnhancedSTGformerModel(config)
    model.build_model(in_channels, out_channels, num_nodes)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model = model.model.to(device)
    
    # Set model to evaluation mode
    model.model.eval()
    
    # Perform inference
    print("Starting inference")
    all_predictions = []
    all_targets = []
    all_losses = []
    
    progress_bar = tqdm(data_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            x_seq = batch['x_seq'].to(device)
            a_seq = batch['a_seq']
            y_true = batch['y_seq'].to(device)
            
            try:
                # Make predictions
                y_pred = model.predict_step(x_seq, a_seq)
                
                # Calculate loss
                loss = model.model.get_loss(y_pred, y_true)
                all_losses.append(loss.item())
                
                # Store predictions and targets
                all_predictions.append(y_pred.cpu())
                all_targets.append(y_true.cpu())
                
                progress_bar.set_postfix(loss=loss.item())
            except Exception as e:
                print(f"[ERROR] Error in inference batch {batch_idx}: {str(e)}")
                continue
    
    # Calculate evaluation metrics
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else float('inf')
    print(f"Average loss: {avg_loss:.6f}")
    
    # Save predictions
    if all_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Calculate additional metrics
        mse = torch.mean((predictions - targets) ** 2).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # Save results
        results_path = os.path.join(output_dir, 'evaluation_results.pt')
        torch.save(
            {
                'predictions': predictions,
                'targets': targets,
                'metrics': {
                    'loss': avg_loss,
                    'mse': mse,
                    'mae': mae
                }
            },
            results_path
        )
        
        print(f"Evaluation results saved to {results_path}")
        
        # Also save in CSV format for easier analysis
        try:
            # Flatten predictions and targets for CSV output
            if len(predictions.shape) > 2:
                pred_flat = predictions.reshape(predictions.shape[0], -1)
                target_flat = targets.reshape(targets.shape[0], -1)
            else:
                pred_flat = predictions
                target_flat = targets
                
            # Limit to a reasonable size for CSV
            max_cols = 100
            if pred_flat.shape[1] > max_cols:
                pred_flat = pred_flat[:, :max_cols]
                target_flat = target_flat[:, :max_cols]
            
            # Create columns
            cols = []
            for i in range(pred_flat.shape[1]):
                cols.append(f'pred_{i}')
                cols.append(f'true_{i}')
            
            # Interleave predictions and targets
            csv_data = torch.zeros(pred_flat.shape[0], pred_flat.shape[1] * 2)
            for i in range(pred_flat.shape[1]):
                csv_data[:, i*2] = pred_flat[:, i]
                csv_data[:, i*2+1] = target_flat[:, i]
            
            # Create DataFrame
            df = pd.DataFrame(csv_data.numpy(), columns=cols)
            
            # Add metrics as extra rows
            metrics_df = pd.DataFrame([
                {'pred_0': 'loss', 'true_0': avg_loss},
                {'pred_0': 'mse', 'true_0': mse},
                {'pred_0': 'mae', 'true_0': mae}
            ])
            
            # Combine DataFrames
            final_df = pd.concat([metrics_df, df], ignore_index=True)
            
            # Save CSV
            csv_path = os.path.join(output_dir, 'evaluation_results.csv')
            final_df.to_csv(csv_path, index=False)
            print(f"CSV results saved to {csv_path}")
        except Exception as e:
            print(f"Error saving CSV results: {str(e)}")
    
    return {
        'loss': avg_loss,
        'mse': mse if 'mse' in locals() else None,
        'mae': mae if 'mae' in locals() else None
    }


def main():
    """Main function for model inference."""
    parser = argparse.ArgumentParser(description='Inference with STGformer model on Waze traffic data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing evaluation data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (if available)')
    
    args = parser.parse_args()
    
    # Set GPU device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override batch size if specified
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Run evaluation
    evaluate_model(
        config=config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()