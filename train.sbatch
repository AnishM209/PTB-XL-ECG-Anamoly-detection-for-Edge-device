#!/bin/bash
#SBATCH --job-name=ptbxl_train
#SBATCH --partition=l40   # see: sinfo -o "%P %G"
#SBATCH --qos=l40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log

# Usage: sbatch train.sbatch --task superclass --leads all --fs 500 --win_sec 2.5
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ecg
cd ~/stage1
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py --workers 6 "$@"
