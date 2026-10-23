#!/bin/bash
#SBATCH --job-name=noise_stress
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/agipml/25m1097/dsp_test/logs/noise_stress_%j.out
set -e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate ecg
cd ~/dsp_test
W=$SLURM_CPUS_PER_TASK

# 1. fold 10, all noise conditions, original code (also saves noisy raw + self-check vs team06/raw)
python noise_stress.py --code ~/test_cleaning --tag orig --workers $W

# 2. fold 10, same noise, detector reset patch (noisy raw already saved)
if [ -d ~/test_cleaning_reset ]; then
  python noise_stress.py --code ~/test_cleaning_reset --tag reset --no_raw --workers $W
fi

# 3. fold 9, no noise, for re-choosing thresholds of the pipelines that have none yet
python noise_stress.py --code ~/test_cleaning --tag orig --fold 9 --pipes med03 hp05zp hp05c --workers $W
