#!/bin/bash
#SBATCH --job-name=device_chain
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/agipml/25m1097/dsp_test/logs/device_chain_%j.out
set -e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate ecg
cd ~/dsp_test
python device_chain.py validate
for f in ~/recordings/*.npz; do
  python device_chain.py recording "$f" --q 5
  python device_chain.py recording "$f" --q 30
done
