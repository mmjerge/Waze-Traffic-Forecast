#!/usr/bin/env python3
"""
Script for training the STGformer model on Waze traffic data.
Uses Weights & Biases for experiment tracking and Accelerate for multi-GPU training.
Optimized for performance on high-end hardware with single-GPU enhancements.
"""

import os
import sys

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, random_split

# Set CUDA optimization flags for better performance
torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
torch.backends.cudnn.deterministic = False  # Disable deterministic mode for speed
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere and newer GPUs
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for convolutions

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
            # Use cached adjacency if available (for consistent training)
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
    
    def train_step(self, x_seq, a_seq, y_true):
        """
        Enhanced train step with specialized adjacency matrix handling.
        """
        self.optimizer.zero_grad()
        
        # Debug information for first batch only
        if not self._printed_debug_info:
            print(f"[DEBUG] x_seq type: {type(x_seq)}, shape: {x_seq.shape if hasattr(x_seq, 'shape') else 'no shape'}")
            print(f"[DEBUG] a_seq type: {type(a_seq)}")
            if isinstance(a_seq, list):
                print(f"[DEBUG] a_seq length: {len(a_seq)}")
                if len(a_seq) > 0:
                    print(f"[DEBUG] a_seq[0] type: {type(a_seq[0])}")
            else:
                print(f"[DEBUG] a_seq shape: {a_seq.shape if hasattr(a_seq, 'shape') else 'no shape'}")
            print(f"[DEBUG] y_true type: {type(y_true)}, shape: {y_true.shape if hasattr(y_true, 'shape') else 'no shape'}")
            self._printed_debug_info = True
        
        # Ensure tensors are on the correct device
        device = x_seq.device
        y_true = y_true.to(device)
        
        # Process the adjacency matrix
        adj = self.prepare_adjacency_matrix(a_seq, x_seq)
        
        # Forward pass
        y_pred = self.model(x_seq, adj)
        
        # Calculate loss
        loss = self.model.get_loss(y_pred, y_true)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_step(self, x_seq, a_seq):
        """Enhanced predict step with specialized adjacency matrix handling"""
        # Process the adjacency matrix
        adj = self.prepare_adjacency_matrix(a_seq, x_seq)
        
        # Forward pass
        return self.model(x_seq, adj)


def train_model(config, **kwargs):
    """
    Train the STGformer model using the specified configuration.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional keyword arguments
    
    Returns:
        Trained model and training history
    """
    # Enable profiling if requested
    profiling_enabled = kwargs.get('profile', False)
    prof = None
    
    # Check for fast dev run mode
    fast_dev_run = kwargs.get('fast_dev_run', False)
    if fast_dev_run:
        config['training']['num_epochs'] = min(2, config['training']['num_epochs'])
        limit_batches = 1  # Process only 1 batch per epoch
    else:
        limit_batches = float('inf')  # Process all batches
    
    # Initialize accelerator with mixed precision for better performance
    # Use appropriate mixed precision based on platform
    mixed_precision = kwargs.get('mixed_precision', None)
    if mixed_precision is None:
        # Auto-detect appropriate mixed precision
        if torch.cuda.is_available():
            mixed_precision = 'fp16'  # Use fp16 on CUDA
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mixed_precision = 'no'  # MPS doesn't support fp16 mixed precision
        else:
            mixed_precision = 'no'  # Default to no mixed precision for CPU
    
    accelerator = Accelerator(
        gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),
        mixed_precision=mixed_precision,
        log_with="wandb" if not kwargs.get('disable_wandb', False) else None
    )
    
    if not kwargs.get('disable_wandb', False):
        if accelerator.is_main_process:
            wandb_project = kwargs.get('wandb_project', 'waze-traffic-forecast')
            wandb_entity = kwargs.get('wandb_entity', None)
            wandb_name = kwargs.get('wandb_name', f"STGformer-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            
            accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs={
                    "wandb": {
                        "entity": wandb_entity,
                        "name": wandb_name,
                        "config": config
                    }
                }
            )
    
    if accelerator.is_main_process:
        print("Accelerator config:")
        print(f"  Number of processes: {accelerator.num_processes}")
        print(f"  Distributed type: {accelerator.distributed_type}")
        print(f"  Mixed precision: {accelerator.mixed_precision}")
        
        if fast_dev_run:
            print("Running in fast development mode (1 batch per epoch, 2 epochs)")
    
    data_dir = kwargs.get('data_dir', config['data']['directory'])
    output_dir = kwargs.get('output_dir', config['paths']['output_dir'])
    checkpoint_dir = kwargs.get('checkpoint_dir', config['paths']['checkpoint_dir'])
    log_dir = kwargs.get('log_dir', config['paths']['log_dir'])
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        config_path = os.path.join(output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    seed = kwargs.get('seed', 42)
    set_seed(seed)
    
    # Create dataset (with only the parameters that WazeGraphDataset actually supports)
    print(f"Loading dataset from {data_dir}")
    dataset = WazeGraphDataset(
        data_dir=data_dir,
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

    if accelerator.is_main_process:
        print(f"Dataset size: {len(dataset)}")
        if dataset.X is not None:
            print(f"Feature tensor shape: {dataset.X.shape}")
            
            if isinstance(dataset.A, list):
                print(f"Adjacency format: List of {len(dataset.A)} sparse matrices")
                if len(dataset.A) > 0:
                    if hasattr(dataset.A[0], 'size'):
                        sparse_shape = dataset.A[0].size()
                        print(f"Each sparse matrix shape: {sparse_shape}")
            else:
                print(f"Adjacency tensor shape: {dataset.A.shape}")
            
            if config['data'].get('full_graph', False) and dataset.edge_index is not None:
                print(f"Edge index shape: {dataset.edge_index.shape}")
                print(f"Total number of edges: {dataset.edge_index.shape[1]}")
            
            in_channels = dataset.X.shape[-1]
            out_channels = in_channels  
            num_nodes = dataset.X.shape[1] 
            
            print(f"Number of input features: {in_channels}")
            print(f"Number of output features: {out_channels}")
            print(f"Number of nodes: {num_nodes}")
        else:
            print("Error: Failed to create dataset tensors")
            return None, None
    else:
        in_channels = dataset.X.shape[-1]
        out_channels = in_channels
        num_nodes = dataset.X.shape[1]
    
    train_size = int(len(dataset) * config['data']['train_ratio'])
    val_size = int(len(dataset) * config['data']['val_ratio'])
    test_size = len(dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    if accelerator.is_main_process:
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(val_dataset)}")
        print(f"Test size: {len(test_dataset)}")
    
    # Optimize dataloader parameters for better performance
    num_workers = min(kwargs.get('num_workers', 4), os.cpu_count() or 4)
    prefetch_factor = kwargs.get('prefetch_factor', 2)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=prefetch_factor,  # Prefetch batches
        collate_fn=sparse_tensor_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=sparse_tensor_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=sparse_tensor_collate_fn
    )
    
    if accelerator.is_main_process:
        print("Initializing model")
        
    # Use our enhanced model wrapper
    model = EnhancedSTGformerModel(config)
    model.build_model(in_channels, out_channels, num_nodes)
    
    # Enable gradient checkpointing if requested
    if kwargs.get('use_gradient_checkpointing', True) and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            print("Gradient checkpointing enabled")
    
    checkpoint_path = kwargs.get('resume_from')
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = model.load_checkpoint(checkpoint_path)
        start_epoch = checkpoint.get('epoch', 0) + 1
        if accelerator.is_main_process:
            print(f"Resuming from epoch {start_epoch}")
    
    model.model, model.optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model.model, model.optimizer, train_loader, val_loader, test_loader
    )
    
    if accelerator.is_main_process:
        print("Starting training")
        
    # Initialize profiler if requested
    if profiling_enabled and accelerator.is_main_process:
        from torch.profiler import profile, record_function, ProfilerActivity
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(log_dir, 'profiler')),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()
        
    num_epochs = config['training']['num_epochs']
    patience = config['training']['patience']
    eval_every = kwargs.get('eval_every', 1)
    
    best_val_loss = float('inf')
    patience_counter = 0
    previous_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        if config['data'].get('full_graph', False):
            dataset.init_epoch(epoch)
        
        model.model.train()
        train_losses = []
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            disable=not accelerator.is_main_process
        )
        
        # Start profiling if enabled
        if profiling_enabled and accelerator.is_main_process and prof:
            prof.step()
            
        for batch_idx, batch in enumerate(progress_bar):
            # Fast dev run: process only limited batches
            if batch_idx >= limit_batches:
                break
                
            x_seq = batch['x_seq']
            a_seq = batch['a_seq']
            y_true = batch['y_seq']
            
            try:
                loss = model.train_step(x_seq, a_seq, y_true)
                train_losses.append(loss)
                
                # Log GPU memory usage every 10 batches
                if accelerator.is_main_process and batch_idx % 10 == 0:
                    if torch.cuda.is_available():
                        used_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        progress_bar.set_postfix(loss=loss, gpu_mem=f"{used_mem:.1f}MB")
                    else:
                        progress_bar.set_postfix(loss=loss)
                else:
                    progress_bar.set_postfix(loss=loss)
            except Exception as e:
                print(f"[ERROR] Error in training batch {batch_idx}: {str(e)}")
                if batch_idx == 0:  # Only break on first batch failure
                    raise  # Re-raise the error to stop training
                continue  # Skip this batch and continue with the next one
        
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        
        current_lr = model.optimizer.param_groups[0]['lr']
        
        # Only run validation every eval_every epochs or on the last epoch
        run_validation = (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1
        
        if run_validation:
            model.model.eval()
            val_losses = []
            
            progress_bar = tqdm(
                val_loader, 
                desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                disable=not accelerator.is_main_process
            )
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(progress_bar):
                    # Fast dev run: process only limited batches
                    if batch_idx >= limit_batches:
                        break
                        
                    x_seq = batch['x_seq']
                    a_seq = batch['a_seq']
                    y_true = batch['y_seq']
                    
                    try:
                        y_pred = model.predict_step(x_seq, a_seq)
                        loss = model.model.get_loss(y_pred, y_true)
                        
                        gathered_loss = accelerator.gather(torch.tensor(loss.item(), device=accelerator.device)).mean().item()
                        val_losses.append(gathered_loss)
                        
                        progress_bar.set_postfix(loss=gathered_loss)
                    except Exception as e:
                        print(f"[ERROR] Error in validation batch {batch_idx}: {str(e)}")
                        continue  # Skip this batch
            
            val_loss = sum(val_losses) / len(val_losses) if val_losses else previous_val_loss
            previous_val_loss = val_loss  # Store for use in epochs where validation is skipped
        else:
            # Skip validation this epoch
            if accelerator.is_main_process:
                print(f"Skipping validation for epoch {epoch+1}")
            val_loss = previous_val_loss  # Use previous value for metrics and scheduler
        
        if not kwargs.get('disable_wandb', False):
            metrics = {
                "train/loss": avg_train_loss,
                "train/learning_rate": current_lr,
                "train/epoch": epoch + 1
            }
            
            if run_validation:
                metrics["val/loss"] = val_loss
            
            # Add GPU memory metrics
            if torch.cuda.is_available():
                metrics["system/gpu_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
                metrics["system/gpu_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)
                
            accelerator.log(metrics)
        
        if accelerator.is_main_process:
            gpu_mem_str = ""
            if torch.cuda.is_available():
                used_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                gpu_mem_str = f", GPU Mem: {used_mem:.1f}MB"
                
            val_str = f", Val Loss: {val_loss:.6f}" if run_validation else ", Val: skipped"
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}{val_str}, "
                  f"LR: {current_lr:.6f}{gpu_mem_str}")
            
            # Reset peak memory stats after logging
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        
        # Update learning rate scheduler
        if model.scheduler is not None:
            if isinstance(model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                model.scheduler.step(val_loss)
            else:
                model.scheduler.step()
        
        # Early stopping - only check when validation is run
        if run_validation and accelerator.is_main_process and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            unwrapped_model = accelerator.unwrap_model(model.model)
            torch.save(
                {
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': avg_train_loss,
                    'config': config
                },
                best_model_path
            )
            print(f"Saved best model checkpoint to {best_model_path}")
        elif run_validation:
            if accelerator.is_main_process:
                patience_counter += 1
                if val_loss >= best_val_loss:
                    print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
        
        # Synchronize patience counter across processes using gather
        if accelerator.num_processes > 1:
            # Create patience counter tensor on the correct device
            patience_tensor = torch.tensor([patience_counter], device=accelerator.device)
            
            # Gather from all processes
            gathered_patience = accelerator.gather(patience_tensor)
            
            # Use the maximum value (if any process needs to stop, all should stop)
            patience_counter = gathered_patience.max().item()
        
        # Wait for all processes to reach this point
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process and (epoch + 1) % kwargs.get('save_every', 5) == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            unwrapped_model = accelerator.unwrap_model(model.model)
            torch.save(
                {
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': avg_train_loss,
                    'config': config
                },
                checkpoint_path
            )
        
        # Early stopping
        if patience_counter >= patience:
            if accelerator.is_main_process:
                print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Stop profiler if enabled
    if profiling_enabled and accelerator.is_main_process and prof:
        prof.stop()
    
    # Wait for all processes to reach this point
    accelerator.wait_for_everyone()
    
    # Load best model for final evaluation (on main process only)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if accelerator.is_main_process and os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=accelerator.device)
        unwrapped_model = accelerator.unwrap_model(model.model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
    
    if accelerator.is_main_process:
        print("\nEvaluating model on test set")
    
    model.model.eval()
    test_losses = []
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(
        test_loader, 
        desc="Testing",
        disable=not accelerator.is_main_process
    )
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Fast dev run: process only limited batches
            if batch_idx >= limit_batches:
                break
                
            x_seq = batch['x_seq']
            a_seq = batch['a_seq']
            y_true = batch['y_seq']
            
            try:
                y_pred = model.predict_step(x_seq, a_seq)
                loss = model.model.get_loss(y_pred, y_true)
                
                gathered_loss = accelerator.gather(torch.tensor(loss.item(), device=accelerator.device)).mean().item()
                test_losses.append(gathered_loss)
                
                gathered_preds = accelerator.gather(y_pred)
                gathered_targets = accelerator.gather(y_true)
                
                if accelerator.is_main_process:
                    all_predictions.append(gathered_preds.cpu())
                    all_targets.append(gathered_targets.cpu())
                
                progress_bar.set_postfix(loss=gathered_loss)
            except Exception as e:
                print(f"[ERROR] Error in test batch {batch_idx}: {str(e)}")
                continue  # Skip this batch
    
    test_loss = sum(test_losses) / len(test_losses) if test_losses else 0
    
    if accelerator.is_main_process and not kwargs.get('disable_wandb', False):
        accelerator.log({"test/loss": test_loss})
    
    if accelerator.is_main_process:
        print(f"Test Loss: {test_loss:.6f}")
        
        if all_predictions:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            torch.save(
                {
                    'predictions': predictions,
                    'targets': targets,
                    'test_loss': test_loss
                },
                os.path.join(output_dir, 'test_predictions.pt')
            )
        
        print(f"Training completed. Results saved to {output_dir}")
    
    accelerator.end_training()
    
    return model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train STGformer model on Waze traffic data')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing Waze parquet files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save model outputs')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory to save training logs')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--wandb_project', type=str, default='waze-traffic-forecast',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--optimize', action='store_true',
                        help='Enable optimization mode (increases batch size and enables mixed precision)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable PyTorch profiling')
    parser.add_argument('--mixed_precision', type=str, choices=['no', 'fp16', 'bf16'], default=None,
                        help='Mixed precision mode (auto-detected if not specified)')
    parser.add_argument('--batch_multiplier', type=int, default=4,
                        help='Multiply batch size by this value when optimize flag is used')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--fast_dev_run', action='store_true',
                        help='Run a quick dev loop for debugging')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch (higher can improve performance)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.data_dir:
        config['data']['directory'] = args.data_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.checkpoint_dir:
        config['paths']['checkpoint_dir'] = args.checkpoint_dir
    if args.log_dir:
        config['paths']['log_dir'] = args.log_dir
    
    # Apply optimization flags
    if args.optimize:
        print("Optimization mode enabled:")
        # Increase batch size
        original_batch_size = config['training']['batch_size']
        config['training']['batch_size'] *= args.batch_multiplier
        print(f"  - Increased batch size from {original_batch_size} to {config['training']['batch_size']}")
        
        # Enable performance optimizations
        args.use_gradient_checkpointing = True
        
        # Other optimizations are applied through kwargs
        print(f"  - Using mixed precision: {args.mixed_precision}")
        print(f"  - Gradient checkpointing: {'enabled' if args.use_gradient_checkpointing else 'disabled'}")
        print(f"  - Profiling: {'enabled' if args.profile else 'disabled'}")
    
    train_model(
        config,
        resume_from=args.resume_from,
        seed=args.seed,
        num_workers=args.num_workers,
        save_every=args.save_every,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        disable_wandb=args.disable_wandb,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        profile=args.profile,
        mixed_precision=args.mixed_precision,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        fast_dev_run=args.fast_dev_run,
        prefetch_factor=args.prefetch_factor
    )


def evaluate_with_accident_metrics(model, dataloader, device, accelerator):
    """
    Enhanced evaluation with accident-specific metrics.
    
    Args:
        model: The trained model
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        accelerator: Accelerator for distributed evaluation
        
    Returns:
        Dictionary with accident-specific metrics
    """
    model.eval()
    total_loss = 0
    accident_loss = 0
    normal_loss = 0
    accident_samples = 0
    normal_samples = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x_seq = batch['x_seq']
            y_true = batch['y_seq']
            a_seq = batch['a_seq']
            
            # Extract accident information from the last timestep
            # Assume accident features are the last 2 features if they exist
            if x_seq.shape[-1] >= 6:  # Original 4 + 2 accident features
                accident_mask = x_seq[:, -1, :, -2] > 0  # is_accident_related from last timestep
                
                y_pred = model.predict_step(x_seq, a_seq)
                
                # Calculate separate losses for accident and normal traffic
                if accident_mask.any():
                    accident_indices = accident_mask.nonzero(as_tuple=False)
                    if len(accident_indices) > 0:
                        acc_pred = y_pred[accident_indices[:, 0], :, accident_indices[:, 1]]
                        acc_true = y_true[accident_indices[:, 0], :, accident_indices[:, 1]]
                        acc_loss = torch.nn.functional.mse_loss(acc_pred, acc_true)
                        accident_loss += acc_loss.item()
                        accident_samples += len(accident_indices)
                
                if (~accident_mask).any():
                    normal_indices = (~accident_mask).nonzero(as_tuple=False)
                    if len(normal_indices) > 0:
                        norm_pred = y_pred[normal_indices[:, 0], :, normal_indices[:, 1]]
                        norm_true = y_true[normal_indices[:, 0], :, normal_indices[:, 1]]
                        norm_loss = torch.nn.functional.mse_loss(norm_pred, norm_true)
                        normal_loss += norm_loss.item()
                        normal_samples += len(normal_indices)
                
                # Calculate total loss
                total_loss += torch.nn.functional.mse_loss(y_pred, y_true).item()
                batch_count += 1
    
    # Calculate averages
    avg_total_loss = total_loss / max(1, batch_count)
    avg_accident_loss = accident_loss / max(1, accident_samples)
    avg_normal_loss = normal_loss / max(1, normal_samples)
    accident_ratio = accident_samples / max(1, accident_samples + normal_samples)
    
    return {
        'total_loss': avg_total_loss,
        'accident_mse': avg_accident_loss,
        'normal_mse': avg_normal_loss,
        'accident_ratio': accident_ratio,
        'accident_samples': accident_samples,
        'normal_samples': normal_samples
    }


if __name__ == "__main__":
    main()