#!/bin/bash
#SBATCH --job-name=dsp_team06
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dsp_team06_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ecg
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -e

# 1. The teammate's own dataset cleaner, current settings (0.6 s median, A7 thresholded, bank processing)
cd ~/test_cleaning
python clean_dataset.py ~/data/ptb-xl/records500 ~/data/ptb-xl-clean06/records500 --workers "${SLURM_CPUS_PER_TASK}"

# 2. Pack the test fold and measure T-wave retention
cd ~/dsp_test
python pack_cleaned.py --clean_root ~/data/ptb-xl-clean06 --name team06 --workers "${SLURM_CPUS_PER_TASK}"

# 3. Score raw vs old settings vs our fix vs the team's current pipeline
python score_batch.py --variants teammate hp05_noA7 team06
