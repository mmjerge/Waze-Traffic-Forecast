 # Scripts Directory

 This directory contains utility scripts for data processing, schema inspection, graph building, and model training.

 ## Scripts

 - `build_waze_graph.py`: Builds the Waze graph dataset from raw data files. Parses command-line arguments for data directory, output path, sampling, and snapshot settings. Saves processed tensors (`X`, `A`, `timestamps`) to a `.pt` file.
  
   Usage:
   ```bash
   python scripts/build_waze_graph.py --data_dir /path/to/waze/data --output_dir ./output --sample 10000 --interval 15 --max_snapshots 100
   ```

 - `inspect_waze_schema.py`: Inspects Parquet files in a specified directory, printing row counts, column names and types, and common fields across files.
  
   Usage:
   ```bash
   python scripts/inspect_waze_schema.py /path/to/waze/data
   ```

 - `train_model.py`: Trains the STGformer model on the Waze traffic dataset. Supports configuration via YAML, Weights & Biases tracking, and multi-GPU training with Accelerate.
  
   Usage:
   ```bash
   python scripts/train_model.py --config config.yaml
   ```

   For distributed training:
   ```bash
   accelerate launch scripts/train_model.py --config config.yaml
   ```