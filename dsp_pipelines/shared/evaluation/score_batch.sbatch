#!/bin/bash
#SBATCH --job-name=dsp_score
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dsp_score_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ecg

cd ~/dsp_test
python score_batch.py "$@"
