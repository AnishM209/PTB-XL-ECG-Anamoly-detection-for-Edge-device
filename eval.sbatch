#!/bin/bash
#SBATCH --job-name=ptbxl_eval
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log

# Usage: sbatch eval.sbatch --task af --leads V2-V1 V3
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ecg
cd ~/stage1
python evaluate.py "$@"
