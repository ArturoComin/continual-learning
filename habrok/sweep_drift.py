#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import shlex
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORE_ROOT = PROJECT_ROOT / "store" / "results" / "sweeps"
DEFAULT_STAGE1_CANDIDATES = 64
DEFAULT_STAGE2_CANDIDATES = 40


def _now_stamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_float(value: Optional[str], fallback: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


@dataclass
class Candidate:
    candidate_id: str
    stage: str
    optimizer: str
    lr: float
    batch: int
    iters: int
    momentum: Optional[float]
    adam_beta1: Optional[float]
    adam_beta2: Optional[float]
    adam_eps: Optional[float]
    weight_decay: Optional[float]
    contexts: int
    n_seeds: int
    drift_ref_context: int
    drift_ref_contexts: List[int]
    strategy_filter: List[str]
    run_naive_adam: int
    run_naive_sgd: int
    run_er_buffer: int
    run_er_buffer_turnover: int
    run_er_full_replay: int
    run_er_reference_replay: int
    run_si: int
    run_si_turnover: int
    extra_args: str
    parent_candidate_id: str = ""
    notes: str = ""

    def to_row(self) -> Dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "optimizer": self.optimizer,
            "lr": f"{self.lr:.8g}",
            "batch": str(self.batch),
            "iters": str(self.iters),
            "momentum": "" if self.momentum is None else f"{self.momentum:.8g}",
            "adam_beta1": "" if self.adam_beta1 is None else f"{self.adam_beta1:.8g}",
            "adam_beta2": "" if self.adam_beta2 is None else f"{self.adam_beta2:.8g}",
            "adam_eps": "" if self.adam_eps is None else f"{self.adam_eps:.8g}",
            "weight_decay": "" if self.weight_decay is None else f"{self.weight_decay:.8g}",
            "contexts": str(self.contexts),
            "n_seeds": str(self.n_seeds),
            "drift_ref_context": str(self.drift_ref_context),
            "drift_ref_contexts": " ".join(str(x) for x in self.drift_ref_contexts),
            "strategy_filter": " | ".join(self.strategy_filter),
            "run_naive_adam": str(self.run_naive_adam),
            "run_naive_sgd": str(self.run_naive_sgd),
            "run_er_buffer": str(self.run_er_buffer),
            "run_er_buffer_turnover": str(self.run_er_buffer_turnover),
            "run_er_full_replay": str(self.run_er_full_replay),
            "run_er_reference_replay": str(self.run_er_reference_replay),
            "run_si": str(self.run_si),
            "run_si_turnover": str(self.run_si_turnover),
            "extra_args": self.extra_args,
            "parent_candidate_id": self.parent_candidate_id,
            "notes": self.notes,
        }


def _candidate_fields() -> List[str]:
    return list(Candidate.__dataclass_fields__.keys())


def _candidate_from_row(row: Dict[str, str]) -> Candidate:
    return Candidate(
        candidate_id=row["candidate_id"],
        stage=row["stage"],
        optimizer=row["optimizer"],
        lr=float(row["lr"]),
        batch=int(row["batch"]),
        iters=int(row["iters"]),
        momentum=_safe_float(row.get("momentum")),
        adam_beta1=_safe_float(row.get("adam_beta1")),
        adam_beta2=_safe_float(row.get("adam_beta2")),
        adam_eps=_safe_float(row.get("adam_eps")),
        weight_decay=_safe_float(row.get("weight_decay")),
        contexts=int(row["contexts"]),
        n_seeds=int(row["n_seeds"]),
        drift_ref_context=int(row["drift_ref_context"]),
        drift_ref_contexts=[int(x) for x in row.get("drift_ref_contexts", "").split() if x],
        strategy_filter=[x.strip() for x in row.get("strategy_filter", "").split("|") if x.strip()],
        run_naive_adam=int(row["run_naive_adam"]),
        run_naive_sgd=int(row["run_naive_sgd"]),
        run_er_buffer=int(row["run_er_buffer"]),
        run_er_buffer_turnover=int(row["run_er_buffer_turnover"]),
        run_er_full_replay=int(row["run_er_full_replay"]),
        run_er_reference_replay=int(row["run_er_reference_replay"]),
        run_si=int(row["run_si"]),
        run_si_turnover=int(row["run_si_turnover"]),
        extra_args=row.get("extra_args", ""),
        parent_candidate_id=row.get("parent_candidate_id", ""),
        notes=row.get("notes", ""),
    )


def _write_candidates(path: Path, candidates: List[Candidate]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_candidate_fields())
        writer.writeheader()
        for c in candidates:
            writer.writerow(c.to_row())


def _read_candidates(path: Path) -> List[Candidate]:
    candidates: List[Candidate] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(_candidate_from_row(row))
    return candidates


def _sample_stage1_candidates(
    rng: random.Random,
    n_candidates: int,
    n_seeds: int,
    iters_values: List[int],
    strategy_filter: List[str],
) -> List[Candidate]:
    candidates: List[Candidate] = []
    batches = [32, 64, 128, 256]
    for idx in range(n_candidates):
        optimizer = rng.choice(["adam", "sgd"])
        lr = _log_uniform(rng, 1e-5, 5e-2)
        batch = rng.choice(batches)
        iters = rng.choice(iters_values)
        momentum = None
        beta1 = None
        beta2 = None
        eps = None
        wd = 0.0 if rng.random() < 0.5 else _log_uniform(rng, 1e-7, 1e-3)
        if optimizer == "sgd":
            momentum = rng.uniform(0.0, 0.99)
        else:
            beta1 = rng.uniform(0.8, 0.99)
            beta2 = rng.uniform(0.95, 0.9999)
            eps = _log_uniform(rng, 1e-9, 1e-7)
        cid = f"s1-{idx:04d}"
        candidates.append(
            Candidate(
                candidate_id=cid,
                stage="stage1",
                optimizer=optimizer,
                lr=lr,
                batch=batch,
                iters=iters,
                momentum=momentum,
                adam_beta1=beta1,
                adam_beta2=beta2,
                adam_eps=eps,
                weight_decay=wd,
                contexts=1,
                n_seeds=n_seeds,
                drift_ref_context=1,
                drift_ref_contexts=[1],
                strategy_filter=strategy_filter,
                run_naive_adam=1,
                run_naive_sgd=1,
                run_er_buffer=1,
                run_er_buffer_turnover=0,
                run_er_full_replay=0,
                run_er_reference_replay=0,
                run_si=1,
                run_si_turnover=0,
                extra_args="",
                notes="training-only sweep",
            )
        )
    return candidates


def _mean_std_ci(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0, 0.0
    stdev = statistics.stdev(values)
    ci95 = 1.96 * stdev / math.sqrt(len(values))
    return mean, stdev, ci95


def _load_stage1_ranking(path: Path, top_k: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: float(r.get("metric_mean", "nan")), reverse=True)
    return rows[:top_k]


def _cl_specific_variants() -> List[Tuple[str, str]]:
    return [
        ("si-low", "--si --reg-strength 1"),
        ("si-mid", "--si --reg-strength 10"),
        ("si-high", "--si --reg-strength 100"),
        ("er-budget-10", "--budget 10 --er-replay buffer"),
        ("er-budget-50", "--budget 50 --er-replay buffer"),
        ("er-budget-100", "--budget 100 --er-replay buffer"),
        ("er-full-replay", "--er-replay all"),
    ]


def _sample_stage2_from_stage1(
    rng: random.Random,
    stage1_top_rows: List[Dict[str, str]],
    n_candidates: int,
    n_seeds: int,
    contexts: int,
    strategy_filter: List[str],
) -> List[Candidate]:
    if not stage1_top_rows:
        raise ValueError("No stage-1 top rows available to build stage-2 candidates.")
    variants = _cl_specific_variants()
    out: List[Candidate] = []
    idx = 0
    while len(out) < n_candidates:
        parent = stage1_top_rows[len(out) % len(stage1_top_rows)]
        variant_name, variant_args = variants[idx % len(variants)]
        parent_lr = float(parent["lr"])
        jittered_lr = max(1e-6, min(1e-1, parent_lr * math.exp(rng.uniform(-0.35, 0.35))))
        parent_batch = int(parent["batch"])
        batch_candidates = sorted({max(16, parent_batch // 2), parent_batch, min(512, parent_batch * 2)})
        batch = rng.choice(batch_candidates)
        cid = f"s2-{len(out):04d}"
        out.append(
            Candidate(
                candidate_id=cid,
                stage="stage2",
                optimizer=parent["optimizer"],
                lr=jittered_lr,
                batch=batch,
                iters=int(parent["iters"]),
                momentum=_safe_float(parent.get("momentum")),
                adam_beta1=_safe_float(parent.get("adam_beta1")),
                adam_beta2=_safe_float(parent.get("adam_beta2")),
                adam_eps=_safe_float(parent.get("adam_eps")),
                weight_decay=_safe_float(parent.get("weight_decay"), 0.0),
                contexts=contexts,
                n_seeds=n_seeds,
                drift_ref_context=1,
                drift_ref_contexts=[1],
                strategy_filter=strategy_filter,
                run_naive_adam=1,
                run_naive_sgd=1,
                run_er_buffer=1,
                run_er_buffer_turnover=1,
                run_er_full_replay=1,
                run_er_reference_replay=0,
                run_si=1,
                run_si_turnover=1,
                extra_args=variant_args,
                parent_candidate_id=parent["candidate_id"],
                notes=f"from {parent['candidate_id']} ({variant_name})",
            )
        )
        idx += 1
    return out


def _build_env_for_candidate(
    c: Candidate, base_seed: int, sweep_id: str, stage: str
) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(base_seed),
            "N_SEEDS": str(c.n_seeds),
            "CONTEXTS": str(c.contexts),
            "OPTIMIZER": c.optimizer,
            "LR": f"{c.lr:.8g}",
            "BATCH": str(c.batch),
            "ITERS": str(c.iters),
            "DRIFT_REF_CONTEXT": str(c.drift_ref_context),
            "DRIFT_REF_CONTEXTS": " ".join(str(x) for x in c.drift_ref_contexts),
            "MOMENTUM": "" if c.momentum is None else f"{c.momentum:.8g}",
            "WEIGHT_DECAY": "0.0" if c.weight_decay is None else f"{c.weight_decay:.8g}",
            "ADAM_BETA1": "0.9" if c.adam_beta1 is None else f"{c.adam_beta1:.8g}",
            "ADAM_BETA2": "0.999" if c.adam_beta2 is None else f"{c.adam_beta2:.8g}",
            "ADAM_EPS": "1e-08" if c.adam_eps is None else f"{c.adam_eps:.8g}",
            "RUN_NAIVE_ADAM": str(c.run_naive_adam),
            "RUN_NAIVE_SGD": str(c.run_naive_sgd),
            "RUN_ER_BUFFER": str(c.run_er_buffer),
            "RUN_ER_BUFFER_TURNOVER": str(c.run_er_buffer_turnover),
            "RUN_ER_FULL_REPLAY": str(c.run_er_full_replay),
            "RUN_ER_REFERENCE_REPLAY": str(c.run_er_reference_replay),
            "RUN_SI": str(c.run_si),
            "RUN_SI_TURNOVER": str(c.run_si_turnover),
            "EXTRA_ARGS": c.extra_args.strip(),
            "SWEEP_ID": sweep_id,
            "SWEEP_STAGE": stage,
            "SWEEP_CANDIDATE_ID": c.candidate_id,
        }
    )
    return env


def _sweep_dir(sweep_id: str) -> Path:
    return STORE_ROOT / sweep_id


def cmd_generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.random_seed)
    sweep_id = args.sweep_id or f"sweep-{_now_stamp()}"
    sweep_dir = _sweep_dir(sweep_id)
    _ensure_dir(sweep_dir)
    strategy_filter = args.strategy_filter
    iters_values = [int(x) for x in args.stage1_iters.split(",")]
    stage1 = _sample_stage1_candidates(
        rng=rng,
        n_candidates=args.stage1_candidates,
        n_seeds=args.stage1_seeds,
        iters_values=iters_values,
        strategy_filter=strategy_filter,
    )
    stage1_path = sweep_dir / "candidates_stage1.csv"
    _write_candidates(stage1_path, stage1)
    meta = {
        "sweep_id": sweep_id,
        "stage1_candidates": args.stage1_candidates,
        "stage1_seeds": args.stage1_seeds,
        "stage2_candidates": args.stage2_candidates,
        "stage2_seeds": args.stage2_seeds,
        "stage2_contexts": args.stage2_contexts,
        "strategy_filter": strategy_filter,
        "random_seed": args.random_seed,
    }
    (sweep_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote stage-1 candidates: {stage1_path}")
    if args.stage2_from_stage1_summary:
        top_rows = _load_stage1_ranking(Path(args.stage2_from_stage1_summary), args.stage2_top_k)
        stage2 = _sample_stage2_from_stage1(
            rng=rng,
            stage1_top_rows=top_rows,
            n_candidates=args.stage2_candidates,
            n_seeds=args.stage2_seeds,
            contexts=args.stage2_contexts,
            strategy_filter=strategy_filter,
        )
        stage2_path = sweep_dir / "candidates_stage2.csv"
        _write_candidates(stage2_path, stage2)
        print(f"Wrote stage-2 candidates: {stage2_path}")
    else:
        print("Skipped stage-2 generation (provide --stage2-from-stage1-summary to enable).")
    print(f"sweep_id={sweep_id}")


def cmd_run_candidate(args: argparse.Namespace) -> None:
    candidate_file = Path(args.candidate_file)
    candidates = _read_candidates(candidate_file)
    if args.candidate_idx < 0 or args.candidate_idx >= len(candidates):
        raise IndexError(f"candidate_idx {args.candidate_idx} out of bounds [0, {len(candidates)-1}]")
    c = candidates[args.candidate_idx]
    stage = args.stage or c.stage
    sweep_id = args.sweep_id
    run_dir = _sweep_dir(sweep_id) / stage / c.candidate_id
    _ensure_dir(run_dir)
    agg_dir = run_dir / f"seeds_{args.base_seed}_{args.base_seed + c.n_seeds - 1}"
    _ensure_dir(agg_dir)
    env = _build_env_for_candidate(
        c, base_seed=args.base_seed, sweep_id=sweep_id, stage=stage
    )
    env["RESULTS_DIR"] = str(agg_dir / "results")
    env["PLOT_DIR"] = str(agg_dir / "plots")
    env["MODEL_DIR"] = str(agg_dir / "models")
    env["SWEEP_RUN_DIR"] = str(agg_dir)
    # Prevent nested array scripts from reading the outer sweep task id as a seed index.
    env.pop("SLURM_ARRAY_TASK_ID", None)
    _ensure_dir(Path(env["RESULTS_DIR"]))
    _ensure_dir(Path(env["PLOT_DIR"]))
    _ensure_dir(Path(env["MODEL_DIR"]))
    cmd = ["bash", str(PROJECT_ROOT / "habrok" / "run_drift_experiment.sbatch")]
    print(
        f"[run-candidate] idx={args.candidate_idx} candidate={c.candidate_id} "
        f"seed_range={args.base_seed}-{args.base_seed + c.n_seeds - 1}"
    )
    if args.dry_run:
        print(
            f"[dry-run] env overrides: "
            f"{json.dumps({k: env[k] for k in sorted(env) if k.startswith('SWEEP_')}, indent=2)}"
        )
        return
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Candidate run failed: {c.candidate_id}, seed_range="
            f"{args.base_seed}-{args.base_seed + c.n_seeds - 1}, rc={result.returncode}"
        )


def _chunk_bounds(total: int, chunk_size: int) -> List[Tuple[int, int]]:
    out = []
    start = 0
    while start < total:
        end = min(total, start + chunk_size) - 1
        out.append((start, end))
        start = end + 1
    return out


def cmd_submit(args: argparse.Namespace) -> None:
    candidate_file = Path(args.candidate_file)
    candidates = _read_candidates(candidate_file)
    n = len(candidates)
    if n == 0:
        raise ValueError("candidate file has no rows")
    sweep_dir = _sweep_dir(args.sweep_id)
    _ensure_dir(sweep_dir)
    jobs_manifest = sweep_dir / f"submitted_jobs_{args.stage}.jsonl"
    chunk_size = max(1, args.chunk_size)
    chunks = _chunk_bounds(n, chunk_size)
    submitted = []
    for chunk_index, (start_idx, end_idx) in enumerate(chunks):
        candidates_in_chunk = end_idx - start_idx + 1
        array_tasks = math.ceil(candidates_in_chunk / args.workers_per_job)
        array_clause = f"0-{array_tasks-1}%{args.max_array_concurrency}"
        sbatch_cmd = [
            "sbatch",
            f"--array={array_clause}",
            f"--cpus-per-task={args.cpus_per_task}",
            f"--mem={args.mem}",
            f"--time={args.time_limit}",
            f"--gres=gpu:{args.gpus_per_job}",
            "--export=ALL,"
            + ",".join(
                [
                    f"SWEEP_ID={args.sweep_id}",
                    f"SWEEP_STAGE={args.stage}",
                    f"CANDIDATE_FILE={candidate_file}",
                    f"WORKERS_PER_JOB={args.workers_per_job}",
                    f"BASE_SEED={args.base_seed}",
                    f"START_OFFSET={start_idx}",
                ]
            ),
            str(PROJECT_ROOT / "habrok" / "run_drift_sweep.sbatch"),
        ]
        if args.dry_run:
            print(" ".join(shlex.quote(x) for x in sbatch_cmd))
            continue
        proc = subprocess.run(sbatch_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
        stdout = proc.stdout.strip()
        job_id = stdout.split()[-1] if stdout else "unknown"
        row = {
            "stage": args.stage,
            "chunk_index": chunk_index,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "array_clause": array_clause,
            "array_tasks": array_tasks,
            "job_id": job_id,
            "sbatch_stdout": stdout,
        }
        submitted.append(row)
        with jobs_manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"submitted chunk {chunk_index}: {start_idx}-{end_idx} -> job {job_id}")
    if submitted:
        print(f"Submitted {len(submitted)} arrays ({n} candidates total).")


def _read_run_summaries(sweep_dir: Path, stage: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    stage_dir = sweep_dir / stage
    if not stage_dir.exists():
        return records
    for candidate_dir in sorted(stage_dir.iterdir()):
        if not candidate_dir.is_dir():
            continue
        for seed_dir in sorted(candidate_dir.iterdir()):
            summary_jsonl = seed_dir / "results" / "drift_run_summary.jsonl"
            summary_json = seed_dir / "results" / "drift_run_summary.json"
            if summary_jsonl.exists():
                for line in summary_jsonl.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    payload["candidate_id"] = candidate_dir.name
                    payload["seed_dir"] = str(seed_dir)
                    payload["stage"] = stage
                    records.append(payload)
            elif summary_json.exists():
                try:
                    payload = json.loads(summary_json.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                rows = payload.get("rows", [])
                for row in rows:
                    row["candidate_id"] = candidate_dir.name
                    row["seed_dir"] = str(seed_dir)
                    row["stage"] = stage
                    records.append(row)
    return records


def cmd_collect(args: argparse.Namespace) -> None:
    sweep_dir = _sweep_dir(args.sweep_id)
    stage = args.stage
    records = _read_run_summaries(sweep_dir, stage)
    out_long = sweep_dir / f"{stage}_results_long.csv"
    if not records:
        print(f"No run summaries found for stage={stage} under {sweep_dir}")
        return
    fieldnames = sorted({k for r in records for k in r.keys()})
    with out_long.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Wrote long results: {out_long}")


def _group_by_candidate(records: List[Dict[str, object]], metric_key: str) -> List[Dict[str, object]]:
    by_candidate: Dict[str, List[Dict[str, object]]] = {}
    for r in records:
        by_candidate.setdefault(str(r["candidate_id"]), []).append(r)
    rows: List[Dict[str, object]] = []
    for cid, recs in by_candidate.items():
        values = []
        for r in recs:
            val = r.get(metric_key)
            if val is None:
                continue
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                continue
        mean, stdev, ci95 = _mean_std_ci(values)
        sample = recs[0]
        rows.append(
            {
                "candidate_id": cid,
                "stage": sample.get("stage", ""),
                "n_runs": len(values),
                "metric_name": metric_key,
                "metric_mean": mean,
                "metric_std": stdev,
                "metric_ci95": ci95,
                "optimizer": sample.get("optimizer", ""),
                "lr": sample.get("lr", ""),
                "batch": sample.get("batch", ""),
                "iters": sample.get("iters", ""),
                "momentum": sample.get("momentum", ""),
                "adam_beta1": sample.get("adam_beta1", ""),
                "adam_beta2": sample.get("adam_beta2", ""),
                "adam_eps": sample.get("adam_eps", ""),
                "weight_decay": sample.get("weight_decay", ""),
                "final_param_drift": sample.get("final_param_drift", ""),
                "final_repr_drift": sample.get("final_repr_drift", ""),
            }
        )
    rows.sort(key=lambda x: float(x["metric_mean"]), reverse=True)
    return rows


def _write_ranked_markdown(path: Path, rows: List[Dict[str, object]], title: str) -> None:
    lines = [f"# {title}", "", "| rank | candidate_id | metric_mean | ci95 | optimizer | lr | batch | iters |", "|---:|---|---:|---:|---|---:|---:|---:|"]
    for i, r in enumerate(rows, 1):
        lines.append(
            "| {rank} | {cid} | {mean:.4f} | {ci:.4f} | {opt} | {lr} | {batch} | {iters} |".format(
                rank=i,
                cid=r["candidate_id"],
                mean=float(r["metric_mean"]),
                ci=float(r["metric_ci95"]),
                opt=r["optimizer"],
                lr=r["lr"],
                batch=r["batch"],
                iters=r["iters"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_report(args: argparse.Namespace) -> None:
    sweep_dir = _sweep_dir(args.sweep_id)
    stage = args.stage
    long_path = sweep_dir / f"{stage}_results_long.csv"
    if not long_path.exists():
        raise FileNotFoundError(f"Missing {long_path}. Run collect first.")
    records: List[Dict[str, object]] = []
    with long_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    metric_key = "final_avg_acc" if stage == "stage2" else "stage1_single_task_acc"
    ranked = _group_by_candidate(records, metric_key=metric_key)
    ranked_path = sweep_dir / f"{stage}_ranked.csv"
    with ranked_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)
    md_name = "stage2_ranked_cl_params.md" if stage == "stage2" else "stage1_ranked_training_params.md"
    _write_ranked_markdown(
        sweep_dir / md_name,
        ranked,
        title=f"{stage} ranking by {metric_key}",
    )
    print(f"Wrote ranked CSV: {ranked_path}")
    print(f"Wrote ranked markdown: {sweep_dir / md_name}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-stage drift sweep tooling for Habrok.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate stage-1/stage-2 candidate files.")
    g.add_argument("--sweep-id", type=str, default="")
    g.add_argument("--random-seed", type=int, default=42)
    g.add_argument("--stage1-candidates", type=int, default=DEFAULT_STAGE1_CANDIDATES)
    g.add_argument("--stage1-seeds", type=int, default=1)
    g.add_argument("--stage1-iters", type=str, default="1000,2000,4000,6000")
    g.add_argument("--stage2-candidates", type=int, default=DEFAULT_STAGE2_CANDIDATES)
    g.add_argument("--stage2-seeds", type=int, default=3)
    g.add_argument("--stage2-contexts", type=int, default=5)
    g.add_argument("--stage2-top-k", type=int, default=12)
    g.add_argument("--stage2-from-stage1-summary", type=str, default="")
    g.add_argument(
        "--strategy-filter",
        nargs="+",
        default=["Naive FT (Adam)", "Naive FT (SGD)", "ER (Buffer)", "SI"],
    )
    g.set_defaults(func=cmd_generate)

    rc = sub.add_parser("run-candidate", help="Run one candidate index (for sbatch worker).")
    rc.add_argument("--sweep-id", type=str, required=True)
    rc.add_argument("--stage", type=str, default="")
    rc.add_argument("--candidate-file", type=str, required=True)
    rc.add_argument("--candidate-idx", type=int, required=True)
    rc.add_argument("--base-seed", type=int, default=1)
    rc.add_argument("--dry-run", action="store_true")
    rc.set_defaults(func=cmd_run_candidate)

    s = sub.add_parser("submit", help="Submit Slurm arrays for a candidate file.")
    s.add_argument("--sweep-id", type=str, required=True)
    s.add_argument("--stage", type=str, choices=["stage1", "stage2"], required=True)
    s.add_argument("--candidate-file", type=str, required=True)
    s.add_argument("--max-array-concurrency", type=int, default=64)
    s.add_argument("--workers-per-job", type=int, default=1)
    s.add_argument("--chunk-size", type=int, default=1000)
    s.add_argument("--cpus-per-task", type=int, default=4)
    s.add_argument("--mem", type=str, default="16G")
    s.add_argument("--gpus-per-job", type=int, default=1)
    s.add_argument("--time-limit", type=str, default="48:00:00")
    s.add_argument("--base-seed", type=int, default=1)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_submit)

    c = sub.add_parser("collect", help="Collect JSON run summaries into long CSV.")
    c.add_argument("--sweep-id", type=str, required=True)
    c.add_argument("--stage", type=str, choices=["stage1", "stage2"], required=True)
    c.set_defaults(func=cmd_collect)

    r = sub.add_parser("report", help="Build ranked tables for stage1/stage2.")
    r.add_argument("--sweep-id", type=str, required=True)
    r.add_argument("--stage", type=str, choices=["stage1", "stage2"], required=True)
    r.set_defaults(func=cmd_report)

    return p


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
