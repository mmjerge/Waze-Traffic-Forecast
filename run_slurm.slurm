#!/bin/bash

#SBATCH --job-name=gml_train         # Job name
#SBATCH --partition=gpu               # Partition (queue) name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=8                    # Number of cores (tasks)
#SBATCH --mem=512G                    # Memory in GB
#SBATCH --gres=gpu:a100:1             # GPU resources: 1x NVIDIA A100 GPUs
#SBATCH --time=16:00:00 
#SBATCH --output=%A_%a.out            # Standard output log
#SBATCH --error=%A_%a.err             # Standard error log
#SBATCH --account=uvasrg_paid         # Allocation account

# Load necessary modules (adjust as needed)
module load cuda/12.4.1
module load miniforge

# Optional: Set up environment (if using conda or virtualenv)
conda activate traffic

python scripts/train_model.py --config config.yaml