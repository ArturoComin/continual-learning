#!/bin/bash
#SBATCH --job-name=drift_sweep
#SBATCH --output=log/drift_sweep_%A_%a.out
#SBATCH --error=log/drift_sweep_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p log

SWEEP_ID="${SWEEP_ID:?SWEEP_ID is required}"
SWEEP_STAGE="${SWEEP_STAGE:?SWEEP_STAGE is required}"
CANDIDATE_FILE="${CANDIDATE_FILE:?CANDIDATE_FILE is required}"
WORKERS_PER_JOB="${WORKERS_PER_JOB:-1}"
BASE_SEED="${BASE_SEED:-1}"
START_OFFSET="${START_OFFSET:-0}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "SLURM_ARRAY_TASK_ID is required."
  exit 1
fi

task_index="${SLURM_ARRAY_TASK_ID}"
base_candidate_idx=$((START_OFFSET + task_index * WORKERS_PER_JOB))

echo "============================================================"
echo "Drift sweep job:"
echo "  sweep_id=${SWEEP_ID} stage=${SWEEP_STAGE}"
echo "  candidate_file=${CANDIDATE_FILE}"
echo "  task_index=${task_index} workers_per_job=${WORKERS_PER_JOB}"
echo "  base_candidate_idx=${base_candidate_idx} base_seed=${BASE_SEED}"
echo "============================================================"

pids=()
for ((w=0; w<WORKERS_PER_JOB; w++)); do
  candidate_idx=$((base_candidate_idx + w))
  if [ "${WORKERS_PER_JOB}" -gt 1 ]; then
    (
      python habrok/sweep_drift.py run-candidate \
        --sweep-id "${SWEEP_ID}" \
        --stage "${SWEEP_STAGE}" \
        --candidate-file "${CANDIDATE_FILE}" \
        --candidate-idx "${candidate_idx}" \
        --base-seed "${BASE_SEED}"
    ) &
    pids+=($!)
  else
    python habrok/sweep_drift.py run-candidate \
      --sweep-id "${SWEEP_ID}" \
      --stage "${SWEEP_STAGE}" \
      --candidate-file "${CANDIDATE_FILE}" \
      --candidate-idx "${candidate_idx}" \
      --base-seed "${BASE_SEED}"
  fi
done

if [ "${#pids[@]}" -gt 0 ]; then
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
fi

echo "Sweep array task ${task_index} done."
