# Waze Traffic Forecast

Graph transformer-based traffic prediction using Waze data.

## Overview

This project implements a graph transformer architecture for traffic forecasting using Waze data. The model represents road topology and traffic patterns as dynamic graphs, where nodes represent junctions and endpoints, and edges capture traffic flow characteristics.

## Features

- Graph transformer architecture for traffic prediction
- Support for both subgraph and full graph training
- Sparse tensor implementation for memory efficiency
- Temporal snapshot creation for time-series analysis
- Multi-GPU training via HuggingFace Accelerate
- Experiment tracking with Weights & Biases

## Installation

### Setup Environment

```bash
# Create and activate conda environment
conda env create -f environment.yaml
conda activate waze-traffic
```

### Install Package

```bash
# Install in development mode
pip install -e .
```

## Usage

### Configuration

The model configuration is controlled through a YAML file (`config.yaml`). Important settings include:

```yaml
data:
  # Training on full graph vs subgraph
  full_graph: false  # Set to true for full graph training
  subgraph_nodes: 5000  # Max nodes when using subgraph
  batch_size: 1000  # Mini-batch size for full graph training
```

### Training

To train the model using the default sparse subgraph approach:

```bash
python scripts/train_model.py --config config.yaml
```

To train on the full graph using mini-batch training:

```bash
# Edit config.yaml to set data.full_graph: true
python scripts/train_model.py --config config.yaml
```

### Distributed Training

For multi-GPU training:

```bash
accelerate launch scripts/train_model.py --config config.yaml
```

### Additional Options

```bash
# Track experiments with Weights & Biases
python scripts/train_model.py --config config.yaml --wandb_project "waze-traffic"

# Resume from checkpoint
python scripts/train_model.py --config config.yaml --resume_from checkpoints/best_model.pt

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 accelerate launch scripts/train_model.py --config config.yaml
```

## Model Architecture

The implementation is based on the STGformer architecture, which combines:

1. **Graph Propagation Layer**: Models spatial dependencies through message passing
2. **Spatiotemporal Attention**: Captures both local and global dependencies
3. **Temporal Positional Encoding**: Preserves temporal ordering

## Dataset

The model can work with two training approaches:

1. **Sparse Subgraph (Default)**:
   - Samples a connected subgraph of important nodes
   - Uses sparse tensors for memory efficiency
   - Suitable for limited hardware resources

2. **Full Graph with Mini-Batches**:
   - Processes the entire graph through node neighborhood sampling
   - Trains on all nodes and edges
   - Requires more computational resources

## Project Structure

```
waze-traffic-forecast/
├── waze_traffic_forecast/          # Main package
│   ├── data/                      # Data processing modules
│   │   ├── graph_builder.py       # Graph construction
│   │   ├── preprocessor.py        # Data preprocessing
│   │   └── inspector.py           # Schema inspection
│   ├── models/                    # Model implementations
│   │   ├── layers.py              # Model layers
│   │   └── stgformer.py           # STGformer implementation
│   ├── dataset.py                 # Dataset implementation
│   └── _config.py                 # Configuration handling
├── scripts/                        # Executable scripts
│   ├── train_model.py             # Training script
│   ├── build_waze_graph.py        # Graph building script
│   └── inspect_waze_schema.py     # Schema inspection script
├── config.yaml                     # Configuration file
├── environment.yaml                # Conda environment file
├── setup.py                        # Package installation
└── README.md                       # This file
```

## Citation

```
@inproceedings{waze-traffic-forecast,
  title={Graph Transformers for Traffic Forecasting},
  author={Potluri, Sravanth and Jerge, Michael M. and Sahay, Shreejeet},
  year={2025},
  organization={University of Virginia}
}
```

## License

This project is licensed under the terms of the MIT license.