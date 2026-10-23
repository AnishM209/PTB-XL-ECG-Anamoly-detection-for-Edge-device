#!/bin/bash
#SBATCH --job-name=dsp_batch
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=agipml
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dsp_batch_%j.out

# REPLACE the next two lines with the conda activation lines from ~/stage1/train.sbatch
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ecg

export MPLBACKEND=Agg      # SWT_python imports matplotlib; no display on compute nodes
export OMP_NUM_THREADS=1   # one thread per worker process
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/dsp_test
python batch_clean.py --workers "${SLURM_CPUS_PER_TASK}" "$@"
