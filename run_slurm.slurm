#!/bin/bash

#SBATCH --job-name=gml_train         # Job name
#SBATCH --partition=gpu               # Partition (queue) name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=8                    # Number of cores (tasks)
#SBATCH --mem=512G                    # Memory in GB
#SBATCH --gres=gpu:a100:1             # GPU resources: 1x NVIDIA A100 GPUs
#SBATCH --time=36:00:00 
#SBATCH --output=%x_%A_%a.out         # Standard output log with job name, job ID, and array ID
#SBATCH --error=%x_%A_%a.err          # Standard error log with job name, job ID, and array ID
#SBATCH --account=uvasrg_paid         # Allocation account

# Load necessary modules (adjust as needed)
module load cuda/12.4.1
module load miniforge

# Optional: Set up environment (if using conda or virtualenv)
conda activate traffic

# python main.py --config config.yaml --checkpoint ./checkpoints/best_model.pt --data_dir /scratch/mj6ux/data/waze --output_dir ./res
ults

python scripts/train_model.py --config config.yaml --optimize