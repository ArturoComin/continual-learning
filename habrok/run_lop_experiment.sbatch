#!/bin/bash
#SBATCH --job-name=lop_experiment
#SBATCH --output=log/lop_%A_%a.out
#SBATCH --error=log/lop_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-11%12
#
# 12-way parallel sweep over learning rate x seed via job array.
# Edit partition/account/GPU/time for your quota.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p log

cd "${SLURM_SUBMIT_DIR}"
PROJECT_LOP_RUNS="/home2/s6068553/Documents/continual-learning/"

# Disable tqdm progress bars in non-interactive Slurm logs (prevents multi-GB .err files).
export TQDM_DISABLE=1

# --- Edit for your environment (see habrok/env.example) ---
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

# Use one shared virtualenv across array jobs to avoid reinstalling large CUDA wheels 12x.
VENV_PATH="/scratch/$USER/venvs/lop"
source "${VENV_PATH}/bin/activate"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

LOCK_DIR="${VENV_PATH}.lock"
REQ_MARKER="${VENV_PATH}/.requirements_installed"

# Guard dependency install so only one task does it.
if mkdir "${LOCK_DIR}" 2>/dev/null; then
  trap 'rmdir "${LOCK_DIR}"' EXIT
  if [ ! -f "${REQ_MARKER}" ]; then
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install --upgrade \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    touch "${REQ_MARKER}"
  else
    echo "dependency marker found; skipping reinstall"
  fi
  rmdir "${LOCK_DIR}"
  trap - EXIT
else
  echo "waiting for dependency install lock..."
  until [ ! -d "${LOCK_DIR}" ] && [ -f "${REQ_MARKER}" ]; do
    sleep 5
  done
fi

LRS=(0.00001 0.0001 0.0003)
SEEDS=(1 2 3 4)

NUM_LRS=${#LRS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_RUNS=$((NUM_LRS * NUM_SEEDS))

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if [ "${TASK_ID}" -ge "${TOTAL_RUNS}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0..$((TOTAL_RUNS-1))"
  exit 1
fi

LR_INDEX=$((TASK_ID / NUM_SEEDS))
SEED_INDEX=$((TASK_ID % NUM_SEEDS))
LR="${LRS[$LR_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
print("device count", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA not available in this job; aborting to avoid CPU-only run.")
PY

echo "starting lr=${LR}, seed=${SEED}"

python lop_experiment.py \
  --seed "${SEED}" \
  --tasks 800 \
  --steps-per-task 60000 \
  --online-testing \
  --optimizer adam \
  --init-weight kaiming_normal \
  --lr "${LR}" \
  --fc-layers 4 \
  --fc-units 2000 \
  --fc-drop 0.0

echo "finished lr=${LR}, seed=${SEED} -> default ./store paths"

# After all array tasks complete, merge lr runs into one 2x2 comparison plot (run once):
# python combine_lop_runs.py --plot-files \
#   store/plots/lop-summary-permMNIST8200-task--F-1024x2000x2000x2000_c10--i60000-lr1e-05-b1-adam-s1.png \
#   store/plots/lop-summary-permMNIST800-task--F-1024x2000x2000x2000_c10--i60000-lr0.0001-b1-adam-s1.png \
#   store/plots/lop-summary-permMNIST800-task--F-1024x2000x2000x2000_c10--i60000-lr0.0003-b1-adam-s1.png \
#   --acc-ymin 0.85 --acc-ymax 0.96 \
#   --output store/plots/lop-summary-permMNIST200-lr-comparison.png
