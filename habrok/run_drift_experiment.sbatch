#!/bin/bash
#SBATCH --job-name=drift_experiment
#SBATCH --output=log/drift_%j.out
#SBATCH --error=log/drift_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p log

# Disable tqdm progress bars in non-interactive Slurm logs.
export TQDM_DISABLE=1

# --- Edit for your environment (see habrok/env.example) ---
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

# Shared virtualenv to avoid re-installing CUDA wheels on every job.
VENV_PATH="/scratch/$USER/venvs/lop"
source "${VENV_PATH}/bin/activate"

LOCK_DIR="${VENV_PATH}.lock"
REQ_MARKER="${VENV_PATH}/.requirements_installed"

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

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
print("device count", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA not available in this job; aborting to avoid CPU-only run.")
PY

# ----------------------------
# Experiment toggles
# ----------------------------
# All values can be overridden at submit time, e.g.:
# sbatch --export=ALL,SEED=10,N_SEEDS=3,CONTEXTS=20,RUN_ER=1,RUN_SI=0 habrok/run_drift_experiment.sbatch

SEED="${SEED:-1}"
N_SEEDS="${N_SEEDS:-3}"
CONTEXTS="${CONTEXTS:-3}"
FC_LAYERS="${FC_LAYERS:-2}"
FC_UNITS="${FC_UNITS:-400}"
FC_DROP="${FC_DROP:-0.0}"
OPTIMIZER="${OPTIMIZER:-adam}"
LR="${LR:-0.0001}"
BATCH="${BATCH:-128}"

# Drift-specific controls.
DRIFT_REF_CONTEXTS="${DRIFT_REF_CONTEXTS:-1}"   # space-separated, e.g. "1 2 5"
DRIFT_REF_CONTEXT="${DRIFT_REF_CONTEXT:-1}"     # primary ref used by some summaries
NAIVE_FT_LR_ADAM="${NAIVE_FT_LR_ADAM:-}"        # empty => inherit LR
NAIVE_FT_LR_SGD="${NAIVE_FT_LR_SGD:-0.01}"
EVAL_ALL_CONTEXTS="${EVAL_ALL_CONTEXTS:-0}"     # 1 => add --eval-all-contexts
ER_REPLAY="${ER_REPLAY:-buffer}"                # buffer|all
REPLAY_CONTEXTS="${REPLAY_CONTEXTS:-}"          # space-separated context ids for ER replay=all
FULL_REPLAY_REFERENCE_TASKS="${FULL_REPLAY_REFERENCE_TASKS:-0}"  # 1 => ER replay=all on DRIFT_REF_CONTEXTS

# Strategy toggles.
RUN_NAIVE_ADAM="${RUN_NAIVE_ADAM:-1}"
RUN_NAIVE_SGD="${RUN_NAIVE_SGD:-1}"
RUN_ER="${RUN_ER:-1}"
RUN_ER_TURNOVER="${RUN_ER_TURNOVER:-1}"
RUN_ER_BUFFER="${RUN_ER_BUFFER:-0}"
RUN_ER_BUFFER_TURNOVER="${RUN_ER_BUFFER_TURNOVER:-0}"
RUN_ER_FULL_REPLAY="${RUN_ER_FULL_REPLAY:-0}"
RUN_SI="${RUN_SI:-1}"
RUN_SI_TURNOVER="${RUN_SI_TURNOVER:-1}"

# Catch-all passthrough for any additional drift_experiment.py CLI args.
EXTRA_ARGS="${EXTRA_ARGS:-}"

strategy_filter=()
[ "${RUN_NAIVE_ADAM}" = "1" ] && strategy_filter+=("Naive FT (Adam)")
[ "${RUN_NAIVE_SGD}" = "1" ] && strategy_filter+=("Naive FT (SGD)")
[ "${RUN_ER}" = "1" ] && strategy_filter+=("ER")
[ "${RUN_ER_TURNOVER}" = "1" ] && strategy_filter+=("ER+Turnover")
[ "${RUN_ER_BUFFER}" = "1" ] && strategy_filter+=("ER (Buffer)")
[ "${RUN_ER_BUFFER_TURNOVER}" = "1" ] && strategy_filter+=("ER (Buffer)+Turnover")
[ "${RUN_ER_FULL_REPLAY}" = "1" ] && strategy_filter+=("ER (Full Replay)")
[ "${RUN_SI}" = "1" ] && strategy_filter+=("SI")
[ "${RUN_SI_TURNOVER}" = "1" ] && strategy_filter+=("SI+Turnover")

if [ "${#strategy_filter[@]}" -eq 0 ]; then
  echo "No strategies selected. Set at least one RUN_* toggle to 1."
  exit 1
fi

cmd=(
  python drift_experiment.py
  --seed "${SEED}"
  --n-seeds "${N_SEEDS}"
  --contexts "${CONTEXTS}"
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --batch "${BATCH}"
  --fc-layers "${FC_LAYERS}"
  --fc-units "${FC_UNITS}"
  --fc-drop "${FC_DROP}"
  --drift-ref-context "${DRIFT_REF_CONTEXT}"
  --drift-ref-contexts ${DRIFT_REF_CONTEXTS}
  --naive-ft-lr-sgd "${NAIVE_FT_LR_SGD}"
  --strategy-filter
)
cmd+=("${strategy_filter[@]}")

if [ -n "${NAIVE_FT_LR_ADAM}" ]; then
  cmd+=(--naive-ft-lr-adam "${NAIVE_FT_LR_ADAM}")
fi
if [ "${EVAL_ALL_CONTEXTS}" = "1" ]; then
  cmd+=(--eval-all-contexts)
fi
if [ "${FULL_REPLAY_REFERENCE_TASKS}" = "1" ]; then
  ER_REPLAY="all"
  REPLAY_CONTEXTS="${DRIFT_REF_CONTEXTS}"
fi
cmd+=(--er-replay "${ER_REPLAY}")
if [ -n "${REPLAY_CONTEXTS}" ]; then
  cmd+=(--replay-contexts ${REPLAY_CONTEXTS})
fi
if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  extra_arr=(${EXTRA_ARGS})
  cmd+=("${extra_arr[@]}")
fi

echo "============================================================"
echo "Drift run config:"
echo "  seed=${SEED} n_seeds=${N_SEEDS} contexts=${CONTEXTS}"
echo "  optimizer=${OPTIMIZER} lr=${LR} batch=${BATCH}"
echo "  fc_layers=${FC_LAYERS} fc_units=${FC_UNITS} fc_drop=${FC_DROP}"
echo "  drift_ref_context=${DRIFT_REF_CONTEXT}"
echo "  drift_ref_contexts=${DRIFT_REF_CONTEXTS}"
echo "  naive_ft_lr_adam=${NAIVE_FT_LR_ADAM:-<inherits --lr>}"
echo "  naive_ft_lr_sgd=${NAIVE_FT_LR_SGD}"
echo "  eval_all_contexts=${EVAL_ALL_CONTEXTS}"
echo "  er_replay=${ER_REPLAY}"
echo "  replay_contexts=${REPLAY_CONTEXTS:-<all previous contexts>}"
echo "  full_replay_reference_tasks=${FULL_REPLAY_REFERENCE_TASKS}"
echo "  strategy_filter=${strategy_filter[*]}"
echo "  extra_args=${EXTRA_ARGS:-<none>}"
echo "============================================================"

printf 'Executing:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

# Quick sanity checks:
#  1) python drift_experiment.py --help
#  2) sbatch --export=ALL,N_SEEDS=1,CONTEXTS=3,RUN_ER=1,RUN_SI=0,RUN_NAIVE_ADAM=0,RUN_NAIVE_SGD=0,RUN_ER_TURNOVER=0,RUN_SI_TURNOVER=0 habrok/run_drift_experiment.sbatch
