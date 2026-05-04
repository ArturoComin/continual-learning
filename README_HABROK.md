# Running `lop_experiment` on RUG Habrok from your laptop

This guide assumes you have **SSH access** to the Habrok login node and a clone of this repository. Official cluster documentation: [Habrok (CIT)](https://docs.hpc.rug.nl/habrok/).

## 1. One-time: SSH and VPN

- If your faculty requires VPN for off-campus SSH, connect first.
- Test login: `ssh <your_username>@habrok.hpc.rug.nl` (exact hostname may differ; use what RUG documents for your cohort).

## 2. Copy the repository to Habrok

From your **local machine** (PowerShell or Git Bash), in the parent folder of this repo:

```powershell
rsync -avz --exclude ".git" --exclude "__pycache__" --exclude "*.pyc" `
  ./continual-learning/ <your_username>@habrok.hpc.rug.nl:~/continual-learning/
```

If `rsync` is not available, use `scp -r` for the same paths.

## 3. One-time on Habrok: Python environment

SSH in, then:

```bash
cd ~/continual-learning
# Load modules you use for PyTorch + CUDA (see `habrok/env.example` and `module avail`).
# module load ...

python3 -m venv ~/venvs/continual
source ~/venvs/continual/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you use **conda** instead, create an environment and `pip install -r requirements.txt` there.

## 4. Configure the batch script

Edit [`habrok/run_lop_experiment.sbatch`](habrok/run_lop_experiment.sbatch):

- `#SBATCH --partition=...` and `#SBATCH --account=...` to match your project.
- `#SBATCH --array=0-29` if you want 30 seeds (array index 0 → seed 1 in the template).
- `#SBATCH --time=...` — full `800 × 60000` Adam steps per job can take many hours; use a smoke run first (`--tasks 5 --steps-per-task 500` added to the `python` line) to validate the pipeline.
- Uncomment and fill in `module load` / `source .../activate` before the `python lop_experiment.py` line.

## 5. Submit from Habrok

```bash
cd ~/continual-learning
mkdir -p log
sbatch habrok/run_lop_experiment.sbatch
```

Note the printed job id (e.g. `Submitted batch job 12345`).

## 6. Monitor

```bash
squeue -u $USER
tail -f log/lop_12345_0.out
```

## 7. Pull results back to your laptop

After jobs finish, each run writes `metrics.jsonl` and `run_meta.json` under the `--output-dir` used in the sbatch file (default template: `$SCRATCH/lop_runs/seed_*`).

From your **local machine**:

```powershell
rsync -avz <your_username>@habrok.hpc.rug.nl:~/lop_runs/ ./lop_runs_habrok/
```

Then plot locally:

```powershell
cd continual-learning
python lop_experiment_plot.py ../lop_runs_habrok/seed_1 ../lop_runs_habrok/seed_2 --output lop_compare.png
```

## 8. Local test (no cluster)

```powershell
cd continual-learning
python lop_experiment.py --output-dir ./lop_local --seed 0 --tasks 3 --steps-per-task 200 --metric-samples 128 --no-gpu --download
python lop_experiment_plot.py ./lop_local --output lop_local.png
```

Use `--no-gpu` only for tiny smoke tests; full runs should use CUDA on Habrok or a workstation GPU.

## 9. Two-stage drift hyperparameter sweeps on Habrok

The repository now includes:
- sweep tool: `habrok/sweep_drift.py`
- sweep sbatch wrapper: `habrok/run_drift_sweep.sbatch`

### Stage-1 (broad, training-only; contexts=1)

Generate candidates:

```bash
cd ~/continual-learning
python habrok/sweep_drift.py generate \
  --sweep-id drift-sweep-001 \
  --stage1-candidates 96 \
  --stage1-seeds 1 \
  --stage1-iters 1000,2000,4000,6000
```

Submit stage-1 (high array fan-out, one worker per job by default):

```bash
python habrok/sweep_drift.py submit \
  --sweep-id drift-sweep-001 \
  --stage stage1 \
  --candidate-file store/results/sweeps/drift-sweep-001/candidates_stage1.csv \
  --max-array-concurrency 64 \
  --workers-per-job 1 \
  --gpus-per-job 1 \
  --cpus-per-task 4 \
  --mem 16G \
  --time-limit 24:00:00
```

Collect + report (ranking metric: `stage1_single_task_acc`):

```bash
python habrok/sweep_drift.py collect --sweep-id drift-sweep-001 --stage stage1
python habrok/sweep_drift.py report  --sweep-id drift-sweep-001 --stage stage1
```

### Stage-2 (narrow, CL-focused; contexts=5)

Generate stage-2 candidates from stage-1 ranking:

```bash
python habrok/sweep_drift.py generate \
  --sweep-id drift-sweep-001 \
  --stage1-candidates 1 \
  --stage2-from-stage1-summary store/results/sweeps/drift-sweep-001/stage1_ranked.csv \
  --stage2-top-k 12 \
  --stage2-candidates 48 \
  --stage2-seeds 3 \
  --stage2-contexts 5
```

Submit stage-2:

```bash
python habrok/sweep_drift.py submit \
  --sweep-id drift-sweep-001 \
  --stage stage2 \
  --candidate-file store/results/sweeps/drift-sweep-001/candidates_stage2.csv \
  --max-array-concurrency 48 \
  --workers-per-job 1 \
  --gpus-per-job 1 \
  --cpus-per-task 4 \
  --mem 16G \
  --time-limit 48:00:00
```

Collect + report (ranking metric: `final_avg_acc`):

```bash
python habrok/sweep_drift.py collect --sweep-id drift-sweep-001 --stage stage2
python habrok/sweep_drift.py report  --sweep-id drift-sweep-001 --stage stage2
```

### Parallelism controls

- Inter-job: `--max-array-concurrency`
- In-job: `--workers-per-job`
- Resource sizing: `--gpus-per-job`, `--cpus-per-task`, `--mem`, `--time-limit`

Use `--dry-run` with `submit` to inspect generated `sbatch` commands before launching.

### Output locations

All sweep artifacts are under:

`store/results/sweeps/<sweep_id>/`

Typical files:
- `candidates_stage1.csv`
- `candidates_stage2.csv`
- `stage1_results_long.csv`
- `stage2_results_long.csv`
- `stage1_ranked.csv`
- `stage2_ranked.csv`
- `stage1_ranked_training_params.md`
- `stage2_ranked_cl_params.md`
