#!/usr/bin/env python3
import argparse
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_series_by_context(plotting_dict, section, field, n_contexts):
    x_context = plotting_dict.get(section, {}).get("x_context", [])
    values = plotting_dict.get(section, {}).get(field, [])
    output = []
    for context_id in range(1, n_contexts + 1):
        indices = [i for i, x in enumerate(x_context) if x == context_id]
        output.append(values[indices[-1]] if len(indices) > 0 else np.nan)
    return output


def _extract_acc_series(plotting_dict, n_contexts):
    online = plotting_dict.get("online_acc", {})
    online_values = online.get("per_context", [])
    online_x_context = online.get("x_context", [])
    if len(online_values) > 0 and len(online_x_context) == len(online_values):
        values = []
        for context_id in range(1, n_contexts + 1):
            indices = [i for i, x in enumerate(online_x_context) if x == context_id]
            values.append(online_values[indices[-1]] if len(indices) > 0 else np.nan)
        return values

    x_context = plotting_dict.get("x_context", [])
    per_context = plotting_dict.get("acc per context", {})
    values = []
    for context_id in range(1, n_contexts + 1):
        indices = [i for i, x in enumerate(x_context) if x == context_id]
        if len(indices) == 0:
            values.append(np.nan)
            continue
        values.append(per_context["context {}".format(context_id)][indices[-1]])
    return values


def _infer_n_contexts(plotting_dict):
    candidates = []
    for section in ("online_acc", "lop"):
        candidates.extend(plotting_dict.get(section, {}).get("x_context", []))
    candidates.extend(plotting_dict.get("x_context", []))
    return int(max(candidates)) if len(candidates) > 0 else 0


def _lr_label_from_dict_file(dict_file):
    filename = os.path.basename(dict_file)
    match = re.search(r"-lr(.+)-b\d+-", filename)
    if match is None:
        return filename.replace("dict-", "").replace(".pkl", "")
    return "lr={}".format(match.group(1))


def _dict_file_from_plot_file(plot_file, results_dir):
    filename = os.path.basename(plot_file)
    if not filename.startswith("lop-summary-") or not filename.endswith(".png"):
        raise ValueError("Unexpected plot filename format: {}".format(filename))
    stamp = filename[len("lop-summary-") : -len(".png")]
    return os.path.join(results_dir, "dict-{}--n1024.pkl".format(stamp))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine multiple LOP dict runs into a single 2x2 comparison plot."
    )
    parser.add_argument(
        "--dict-files",
        nargs="+",
        default=None,
        help="explicit dict cache files (e.g., store/results/dict-...pkl)",
    )
    parser.add_argument(
        "--plot-files",
        nargs="+",
        default=None,
        help="lop-summary png files; corresponding dict files are inferred",
    )
    parser.add_argument(
        "--results-dir",
        default="store/results",
        help="used with --plot-files to infer dict file paths",
    )
    parser.add_argument(
        "--output",
        default="store/plots/lop-summary-lr-comparison.png",
        help="output path for combined plot",
    )
    parser.add_argument("--acc-ymin", type=float, default=0.85)
    parser.add_argument("--acc-ymax", type=float, default=0.96)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dict_files is None and args.plot_files is None:
        raise ValueError("Provide either --dict-files or --plot-files.")

    dict_files = []
    if args.dict_files is not None:
        dict_files.extend(args.dict_files)
    if args.plot_files is not None:
        for plot_file in args.plot_files:
            dict_files.append(_dict_file_from_plot_file(plot_file, args.results_dir))

    runs = []
    for dict_file in dict_files:
        if not os.path.isfile(dict_file):
            raise FileNotFoundError("Missing dict file: {}".format(dict_file))
        with open(dict_file, "rb") as handle:
            plotting_dict = pickle.load(handle)
        n_contexts = _infer_n_contexts(plotting_dict)
        if n_contexts <= 0:
            raise ValueError("Could not infer contexts from {}".format(dict_file))
        contexts = list(range(1, n_contexts + 1))
        runs.append(
            {
                "label": _lr_label_from_dict_file(dict_file),
                "contexts": contexts,
                "acc": _extract_acc_series(plotting_dict, n_contexts),
                "dead": _extract_series_by_context(plotting_dict, "lop", "dead_unit_fraction", n_contexts),
                "rank": _extract_series_by_context(plotting_dict, "lop", "effective_rank", n_contexts),
                "weight": _extract_series_by_context(plotting_dict, "lop", "weight_magnitude", n_contexts),
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_acc, ax_dead, ax_rank, ax_weight = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for run in runs:
        ax_acc.plot(run["contexts"], run["acc"], label=run["label"])
        ax_dead.plot(run["contexts"], run["dead"], label=run["label"])
        ax_rank.plot(run["contexts"], run["rank"], label=run["label"])
        ax_weight.plot(run["contexts"], run["weight"], label=run["label"])

    ax_acc.set_title("Online task accuracy")
    ax_acc.set_xlabel("task")
    ax_acc.set_ylim(args.acc_ymin, args.acc_ymax)

    ax_dead.set_title("Dead unit fraction")
    ax_dead.set_xlabel("task")

    ax_rank.set_title("Effective rank")
    ax_rank.set_xlabel("task")

    ax_weight.set_title("Weight magnitude")
    ax_weight.set_xlabel("task")

    for ax in (ax_acc, ax_dead, ax_rank, ax_weight):
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("LOP summary by learning rate", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print("Generated plot: {}".format(args.output))


if __name__ == "__main__":
    main()
