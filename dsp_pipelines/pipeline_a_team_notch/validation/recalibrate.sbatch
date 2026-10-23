#!/bin/bash
#SBATCH --job-name=recal06
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/recal06_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ecg
cd ~/dsp_test
python recalibrate.py --workers "${SLURM_CPUS_PER_TASK}"
