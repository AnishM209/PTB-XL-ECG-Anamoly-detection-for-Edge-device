#!/bin/bash
#SBATCH --job-name=ptbxl_seeds
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log

# Runs several seeds of one configuration back to back in a single job, so a
# whole configuration uses one slot of the per-user job limit.
# Usage: sbatch train_seeds.sbatch "0 1 2" --task superclass --leads V1 --model resnet1d_wang
seeds=$1; shift
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ecg
cd ~/stage1
nvidia-smi --query-gpu=name,memory.total --format=csv
for s in $seeds; do
  echo "================ seed $s ================"
  python train.py --workers 6 --seed "$s" "$@" || echo "seed $s FAILED"
done
