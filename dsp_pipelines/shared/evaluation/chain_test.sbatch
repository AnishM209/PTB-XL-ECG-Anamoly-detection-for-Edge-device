#!/bin/bash
#SBATCH --job-name=chain_test
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/agipml/25m1097/dsp_test/logs/chain_test_%j.out
set -e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate ecg
cd ~/dsp_test
python chain_test.py --workers $SLURM_CPUS_PER_TASK
