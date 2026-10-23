#!/bin/bash
#SBATCH --job-name=hw_filter
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/agipml/25m1097/dsp_test/logs/hw_filter_%j.out
set -e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate ecg
cd ~/dsp_test
python hw_filter_test.py
