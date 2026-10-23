#!/bin/bash
#SBATCH --job-name=team_dev
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/team_on_device_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ecg
cd ~/dsp_test
python team_on_device.py thresholds --q 5 --workers 16
for f in ~/recordings/still_01.npz ~/recordings/move_01.npz; do
  [ -f "$f" ] && python team_on_device.py recording "$f" --q 5
done
