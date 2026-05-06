#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mean_std_ci(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(len(values))
    return mean, std, ci95


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate drift summaries across seed runs.")
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob for per-seed drift_run_summary.jsonl files.",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Output path for aggregated JSON summary.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output path for aggregated CSV summary.",
    )
    parser.add_argument(
        "--out-plot",
        default="",
        help="Optional output path for a PDF with aggregated metric plots.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    rows: List[Dict] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source"] = path
                rows.append(row)

    if not rows:
        raise RuntimeError("Matched files were empty; no rows to aggregate.")

    numeric_keys = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_keys.add(key)

    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("strategy_name", "unknown"), int(row.get("drift_ref_context", -1)))].append(row)

    out_rows: List[Dict] = []
    for (strategy_name, drift_ref_context), group in sorted(grouped.items()):
        out = {
            "strategy_name": strategy_name,
            "drift_ref_context": drift_ref_context,
            "n_rows": len(group),
        }
        for key in sorted(numeric_keys):
            vals = [float(r[key]) for r in group if key in r and r[key] is not None]
            mean, std, ci95 = _mean_std_ci(vals)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            out[f"{key}_ci95"] = ci95
        out_rows.append(out)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_glob": args.input_glob,
                "matched_files": files,
                "n_input_rows": len(rows),
                "aggregated_rows": out_rows,
            },
            f,
            indent=2,
        )

    fieldnames = sorted({k for row in out_rows for k in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    if args.out_plot:
        out_plot = Path(args.out_plot)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        _write_plots(out_rows, out_plot)

    print(f"Matched files: {len(files)}")
    print(f"Input rows: {len(rows)}")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote CSV: {out_csv}")
    if args.out_plot:
        print(f"Wrote plots: {args.out_plot}")


def _write_plots(aggregated_rows: List[Dict], out_plot: Path) -> None:
    tracked_metrics = [
        ("stage1_single_task_acc_mean", "Stage-1 single task accuracy"),
        ("final_avg_acc_mean", "Final average accuracy"),
        ("final_ref_task_acc_mean", "Final reference-task accuracy"),
        ("final_param_drift_mean", "Final parameter drift"),
        ("final_repr_drift_mean", "Final representational drift"),
        ("final_dead_unit_fraction_mean", "Final dead unit fraction"),
        ("final_effective_rank_mean", "Final effective rank"),
    ]

    grouped = defaultdict(list)
    for row in aggregated_rows:
        grouped[int(row.get("drift_ref_context", -1))].append(row)

    refs = sorted(grouped.keys())
    n_rows = len(tracked_metrics)
    n_cols = max(len(refs), 1)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for col_idx, ref_context in enumerate(refs):
        rows_for_ref = sorted(grouped[ref_context], key=lambda r: r.get("strategy_name", ""))
        strategy_names = [r.get("strategy_name", "unknown") for r in rows_for_ref]
        for row_idx, (metric_key, ylabel) in enumerate(tracked_metrics):
            ax = axes[row_idx][col_idx]
            means = [float(r.get(metric_key, float("nan"))) for r in rows_for_ref]
            ci_key = metric_key.replace("_mean", "_ci95")
            cis = [float(r.get(ci_key, 0.0) or 0.0) for r in rows_for_ref]
            x = list(range(len(strategy_names)))
            ax.bar(x, means, yerr=cis, capsize=3)
            ax.set_title(f"Ref task {ref_context}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(strategy_names, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
