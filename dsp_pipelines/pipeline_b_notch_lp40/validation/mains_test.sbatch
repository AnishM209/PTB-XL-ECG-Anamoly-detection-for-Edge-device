#!/bin/bash
#SBATCH --job-name=mains_test
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/mains_test_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ecg
cd ~/dsp_test
python mains_test.py
