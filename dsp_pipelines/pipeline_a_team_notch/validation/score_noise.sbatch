#!/bin/bash
#SBATCH --job-name=score_noise
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/agipml/25m1097/dsp_test/logs/score_noise_%j.out
set -e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate ecg
cd ~/dsp_test
python score_noise.py
