#!/usr/bin/env python3
"""
Script for training the STGformer model on Waze traffic data.
Uses Weights & Biases for experiment tracking and Accelerate for multi-GPU training.
"""

import os
import sys
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


def train_model(config, **kwargs):
    """
    Train the STGformer model using the specified configuration.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional keyword arguments
    
    Returns:
        Trained model and training history
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=True,
        collate_fn=sparse_tensor_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=True,
        collate_fn=sparse_tensor_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=True,
        collate_fn=sparse_tensor_collate_fn
    )
    
    if accelerator.is_main_process:
        print("Initializing model")
        
    model = STGformerModel(config)
    model.build_model(in_channels, out_channels, num_nodes)
    
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
        
    num_epochs = config['training']['num_epochs']
    patience = config['training']['patience']
    
    best_val_loss = float('inf')
    patience_counter = 0
    
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
        
        for batch_idx, batch in enumerate(progress_bar):
            x_seq = batch['x_seq']
            a_seq = batch['a_seq']
            y_true = batch['y_seq']
            
            loss = model.train_step(x_seq, a_seq, y_true)
            train_losses.append(loss)
            
            progress_bar.set_postfix(loss=loss)
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        current_lr = model.optimizer.param_groups[0]['lr']
        
        model.model.eval()
        val_losses = []
        
        progress_bar = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
            disable=not accelerator.is_main_process
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                x_seq = batch['x_seq']
                a_seq = batch['a_seq']
                y_true = batch['y_seq']
                
                y_pred = model.model(x_seq, a_seq)
                
                loss = model.model.get_loss(y_pred, y_true)
                
                gathered_loss = accelerator.gather(torch.tensor(loss.item(), device=accelerator.device)).mean().item()
                val_losses.append(gathered_loss)
                
                progress_bar.set_postfix(loss=gathered_loss)
        
        val_loss = sum(val_losses) / len(val_losses)
        
        if not kwargs.get('disable_wandb', False):
            metrics = {
                "train/loss": avg_train_loss,
                "val/loss": val_loss,
                "train/learning_rate": current_lr,
                "train/epoch": epoch + 1
            }
            accelerator.log(metrics)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Update learning rate scheduler
        if model.scheduler is not None:
            if isinstance(model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                model.scheduler.step(val_loss)
            else:
                model.scheduler.step()
        
        # Early stopping
        if accelerator.is_main_process and val_loss < best_val_loss:
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
        else:
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
        for batch in progress_bar:
            x_seq = batch['x_seq']
            a_seq = batch['a_seq']
            y_true = batch['y_seq']
            
            y_pred = model.model(x_seq, a_seq)
            
            loss = model.model.get_loss(y_pred, y_true)
            
            gathered_loss = accelerator.gather(torch.tensor(loss.item(), device=accelerator.device)).mean().item()
            test_losses.append(gathered_loss)
            
            gathered_preds = accelerator.gather(y_pred)
            gathered_targets = accelerator.gather(y_true)
            
            if accelerator.is_main_process:
                all_predictions.append(gathered_preds.cpu())
                all_targets.append(gathered_targets.cpu())
            
            progress_bar.set_postfix(loss=gathered_loss)
    
    test_loss = sum(test_losses) / len(test_losses)
    
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
    
    train_model(
        config,
        resume_from=args.resume_from,
        seed=args.seed,
        num_workers=args.num_workers,
        save_every=args.save_every,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        disable_wandb=args.disable_wandb,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


if __name__ == "__main__":
    main()