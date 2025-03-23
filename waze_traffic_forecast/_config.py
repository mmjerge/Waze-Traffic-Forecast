"""
Configuration loading and handling for Waze traffic forecasting.
"""

import os
import yaml
import torch
from typing import Dict, Any, Optional

def get_default_config():
    """
    Get default configuration settings.
    
    Returns:
        Dict containing default configuration parameters
    """
    return {
        # Data settings
        "data": {
            "directory": "/scratch/mj6ux/data/waze",
            "sample_size": None,
            "interval_minutes": 15,
            "max_snapshots": 1000,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "sequence_length": 12,
            "prediction_horizon": 3,
            "feature_columns": ["speed", "severity", "delay", "length"],
        },
        
        # Model settings
        "model": {
            "name": "STGformer",
            "hidden_channels": 64,
            "num_layers": 3,
            "num_heads": 8,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "use_layer_norm": True,
            "use_residual": True,
        },
        
        # Training settings
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 100,
            "patience": 10,
            "lr_scheduler": "cosine",
            "lr_scheduler_params": {
                "step_size": 10,
                "gamma": 0.1,
                "factor": 0.5,
                "patience": 5,
            },
            "grad_clip_value": 5.0,
            "device": "auto",
        },
        
        "paths": {
            "output_dir": "./output",
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
        },
    }

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict containing configuration parameters
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if config["training"]["device"] == "auto":
        config["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the YAML configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def get_default_config_path() -> str:
    """
    Get path to the default configuration file.
    
    Returns:
        Path to the default config.yaml file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    package_config = os.path.join(current_dir, "config.yaml")
    if os.path.exists(package_config):
        return package_config
    
    repo_config = os.path.join(os.path.dirname(current_dir), "config.yaml")
    if os.path.exists(repo_config):
        return repo_config
    
    return os.path.join(os.getcwd(), "config.yaml")