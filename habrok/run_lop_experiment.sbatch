#!/bin/bash
#SBATCH --job-name=lop_experiment
#SBATCH --output=log/lop_%A_%a.out
#SBATCH --error=log/lop_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2%3
#
# 3-way parallel sweep over learning rate via job array.
# Edit partition/account/GPU/time for your quota.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# #SBATCH --account=YOUR_ACCOUNT

set -euo pipefail
mkdir -p log

cd "${SLURM_SUBMIT_DIR}"

# --- Edit for your environment (see habrok/env.example) ---
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

LRS=(0.00001 0.0001 0.0003)
LR="${LRS[$SLURM_ARRAY_TASK_ID]}"

OUT="${SCRATCH:-$HOME}/lop_runs/lr_${LR}"
mkdir -p "${OUT}"

python lop_experiment.py \
  --output-dir "${OUT}" \
  --plot-dir "${OUT}" \
  --seed 1 \
  --tasks 800 \
  --steps-per-task 60000 \
  --optimizer adam \
  --lr "${LR}" \
  --fc-layers 4 \
  --fc-units 2000 \
  --fc-drop 0.0

echo "finished lr=${LR} -> ${OUT}"
