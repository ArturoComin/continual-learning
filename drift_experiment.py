#!/usr/bin/env python3
import copy
import json
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main
import utils
from utils import checkattr
from params import options
from params.param_stamp import get_param_stamp_from_args
from params.param_values import set_default_values, check_for_errors
from visual import visual_plt


def handle_inputs():
    kwargs = {"comparison": True}
    parser = options.define_args(
        filename="drift_experiment",
        description="Run ER/SI drift experiment on permMNIST and plot accuracy + drift metrics.",
    )
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    parser.add_argument(
        "--drift-ref-contexts",
        type=int,
        nargs="+",
        default=None,
        help="one or more reference contexts for post-hoc drift plots (e.g. --drift-ref-contexts 1 2 10)",
    )
    parser.add_argument(
        "--naive-ft-lr-adam",
        type=float,
        default=None,
        help="learning rate for the Naive FT (Adam) baseline (default: uses --lr)",
    )
    parser.add_argument(
        "--naive-ft-lr-sgd",
        type=float,
        default=0.01,
        help="learning rate for the Naive FT (SGD) baseline",
    )
    parser.add_argument(
        "--eval-all-contexts",
        action="store_true",
        help="use evaluate.test_all instead of test_all_so_far (off by default)",
    )
    parser.add_argument(
        "--er-replay",
        type=str,
        default="buffer",
        choices=["buffer", "all"],
        help="replay mode used by ER-based strategies in this script",
    )
    parser.add_argument(
        "--replay-contexts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "optional context ids to replay when using --er-replay all "
            "(e.g. --replay-contexts 1 2 10)"
        ),
    )
    parser.add_argument(
        "--strategy-filter",
        type=str,
        nargs="+",
        default=None,
        help=(
            "optional subset of strategies to run; accepts names with spaces when quoted "
            '(e.g. --strategy-filter "ER (Buffer)" "SI+Turnover") or comma-separated chunks '
            '(e.g. --strategy-filter "ER (Buffer),SI+Turnover")'
        ),
    )
    args = parser.parse_args()

    # Fixed experiment setup for this script.
    args.experiment = "permMNIST"
    args.contexts = 10 if args.contexts is None else args.contexts
    args.scenario = "task"
    args.drift_metrics = True
    args.drift_ref_context = (
        1 if args.drift_ref_context is None else args.drift_ref_context
    )
    if args.drift_ref_contexts is None:
        args.drift_ref_contexts = [args.drift_ref_context]
    for ref_context in args.drift_ref_contexts:
        if ref_context < 1 or ref_context > args.contexts:
            raise ValueError(
                "All entries in '--drift-ref-contexts' should be in [1, --contexts]."
            )
    if args.replay_contexts is not None:
        args.replay_contexts = sorted(set(args.replay_contexts))
        for replay_context in args.replay_contexts:
            if replay_context < 1 or replay_context > args.contexts:
                raise ValueError(
                    "All entries in '--replay-contexts' should be in [1, --contexts]."
                )
    args.results_dict = True
    args.pdf = False

    set_default_values(args, also_hyper_params=True)
    check_for_errors(args, **kwargs)
    return args


def _dict_file_prefix(args):
    param_stamp = get_param_stamp_from_args(args)
    suffix = "--S{}".format(args.eval_s) if checkattr(args, "gen_classifier") else ""
    suffix += "--evalAll" if checkattr(args, "eval_all_contexts") else ""
    dict_prefix = "{}/dict-{}--n{}{}".format(
        args.r_dir, param_stamp, "All" if args.acc_n is None else args.acc_n, suffix
    )
    return dict_prefix, param_stamp


def _acc_file(args, param_stamp):
    suffix = "--S{}".format(args.eval_s) if checkattr(args, "gen_classifier") else ""
    return "{}/acc-{}{}.txt".format(args.r_dir, param_stamp, suffix)


def _get_unique_pdf_path(base_path):
    if not os.path.isfile(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    counter = 2
    while True:
        candidate = "{}-v{}{}".format(root, counter, ext)
        if not os.path.isfile(candidate):
            return candidate
        counter += 1


def _has_valid_plotting_data(plotting_dict, expected_contexts):
    if not isinstance(plotting_dict, dict):
        return False
    x_context = plotting_dict.get("x_context", [])
    averages = plotting_dict.get("average", [])
    per_context = plotting_dict.get("acc per context", {})
    if len(x_context) == 0 or len(averages) == 0:
        return False
    if len(x_context) != len(averages):
        return False
    if expected_contexts is not None and len(x_context) < expected_contexts:
        return False
    if not isinstance(per_context, dict):
        return False
    for context_id in range(1, expected_contexts + 1):
        key = "context {}".format(context_id)
        if key not in per_context:
            return False
        if len(per_context[key]) != len(x_context):
            return False
    return True


def run_and_collect(args):
    start_time = time.time()
    dict_prefix, param_stamp = _dict_file_prefix(args)
    acc_file = _acc_file(args, param_stamp)
    model_file = "{}/mM-{}".format(args.m_dir, param_stamp)

    if os.path.isfile(dict_prefix + ".pkl"):
        print(" already run (dict): {}".format(param_stamp))
        plotting_dict = utils.load_object(dict_prefix)
        if not _has_valid_plotting_data(plotting_dict, expected_contexts=args.contexts):
            print(
                " cached dict invalid/empty -> rebuilding: {}.pkl".format(dict_prefix)
            )
            os.remove(dict_prefix + ".pkl")
            args.train = True
            main.run(args)
    elif os.path.isfile(acc_file) or os.path.isfile(model_file):
        source = "acc" if os.path.isfile(acc_file) else "checkpoint"
        print(
            " found {} but missing dict -> rerunning training: {}".format(
                source, param_stamp
            )
        )
        args.train = True
        main.run(args)
    else:
        print(" ...running: {}".format(param_stamp))
        args.train = True
        main.run(args)

    print(" loading plotting dict: {}.pkl".format(dict_prefix))
    plotting_dict = utils.load_object(dict_prefix)
    print(" run_and_collect finished in {:.1f}s".format(time.time() - start_time))
    return plotting_dict


def extract_current_task_accuracy(plotting_dict, n_contexts):
    x_context = plotting_dict["x_context"]
    per_context = plotting_dict["acc per context"]
    values = []
    for context_id in range(1, n_contexts + 1):
        indices = [i for i, x in enumerate(x_context) if x == context_id]
        if len(indices) == 0:
            values.append(np.nan)
            continue
        index = indices[-1]
        values.append(per_context["context {}".format(context_id)][index])
    return values


def extract_series_by_context(plotting_dict, series, n_contexts, x_key="x_context"):
    x_context = plotting_dict[x_key]
    values = []
    for context_id in range(1, n_contexts + 1):
        indices = [i for i, x in enumerate(x_context) if x == context_id]
        if len(indices) == 0:
            values.append(np.nan)
            continue
        values.append(series[indices[-1]])
    return values


def extract_series_over_contexts(
    plotting_dict, series, n_contexts, x_key="x_context", fill_before_first=np.nan
):
    x_context = plotting_dict[x_key]
    values = []
    latest_seen = fill_before_first
    for context_id in range(1, n_contexts + 1):
        indices = [i for i, x in enumerate(x_context) if x == context_id]
        if len(indices) > 0:
            latest_seen = series[indices[-1]]
        values.append(latest_seen)
    return values


def extract_average_acc_so_far(plotting_dict, n_contexts):
    x_context = plotting_dict["x_context"]
    per_context = plotting_dict["acc per context"]
    values = []
    for context_id in range(1, n_contexts + 1):
        indices = [i for i, x in enumerate(x_context) if x == context_id]
        if len(indices) == 0:
            values.append(np.nan)
            continue
        index = indices[-1]
        accs_so_far = [
            per_context["context {}".format(task_id)][index]
            for task_id in range(1, context_id + 1)
        ]
        values.append(float(np.mean(accs_so_far)))
    return values


def aggregate_series(series_list):
    data = np.array(series_list, dtype=float)
    mean = np.nanmean(data, axis=0)
    ci = None
    if data.shape[0] > 1:
        std = np.nanstd(data, axis=0, ddof=1)
        ci = 1.96 * std / np.sqrt(data.shape[0])
    return mean.tolist(), None if ci is None else ci.tolist()


def _cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def recompute_drift_from_snapshots(plotting_dict, reference_context, n_contexts):
    snapshots = plotting_dict.get("drift", {}).get("snapshots", {})
    ref_key = str(reference_context)
    if ref_key not in snapshots:
        return None, None
    ref_state = snapshots[ref_key]
    param_drift = []
    repr_drift = []
    for context_id in range(1, n_contexts + 1):
        key = str(context_id)
        if key not in snapshots:
            param_drift.append(np.nan)
            repr_drift.append(np.nan)
            continue
        cur_state = snapshots[key]
        param_drift.append(
            1.0 - _cosine(cur_state["param_vector"], ref_state["param_vector"])
        )
        repr_values = []
        for dataset_id in range(context_id):
            if dataset_id >= len(cur_state["repr_vectors"]) or dataset_id >= len(
                ref_state["repr_vectors"]
            ):
                continue
            cur_repr = cur_state["repr_vectors"][dataset_id]
            ref_repr = ref_state["repr_vectors"][dataset_id]
            if (cur_repr is None) or (ref_repr is None):
                continue
            repr_values.append(1.0 - _cosine(cur_repr, ref_repr))
        repr_drift.append(
            float(np.mean(repr_values)) if len(repr_values) > 0 else np.nan
        )
    return param_drift, repr_drift


def set_er_buffer_args(args):
    args.replay = "buffer"
    args.replay_contexts = None
    args.sample_selection = "random"
    args.use_replay = "normal"
    args.weight_penalty = False
    args.precondition = False
    args.importance_weighting = None
    return args


def set_er_full_replay_args(args):
    args.replay = "all"
    args.replay_contexts = None
    args.sample_selection = "random"
    args.use_replay = "normal"
    args.weight_penalty = False
    args.precondition = False
    args.importance_weighting = None
    return args


def set_er_reference_replay_args(args):
    args.replay = "all"
    args.replay_contexts = sorted(set(args.drift_ref_contexts))
    args.sample_selection = "random"
    args.use_replay = "normal"
    args.weight_penalty = False
    args.precondition = False
    args.importance_weighting = None
    return args


def set_si_args(args):
    args.replay = "none"
    args.weight_penalty = True
    args.precondition = False
    args.importance_weighting = "si"
    args.reg_strength = args.si_c
    return args


def set_turnover_args(args):
    args.syn_turnover = True
    return args


def set_naive_finetune_adam_args(args):
    args.replay = "none"
    args.weight_penalty = False
    args.precondition = False
    args.importance_weighting = None
    args.optimizer = "adam"
    if hasattr(args, "naive_ft_lr_adam") and args.naive_ft_lr_adam is not None:
        args.lr = args.naive_ft_lr_adam
    return args


def set_naive_finetune_sgd_args(args):
    args.replay = "none"
    args.weight_penalty = False
    args.precondition = False
    args.importance_weighting = None
    args.optimizer = "sgd"
    if hasattr(args, "naive_ft_lr_sgd") and args.naive_ft_lr_sgd is not None:
        args.lr = args.naive_ft_lr_sgd
    args.momentum = 0.9
    return args


if __name__ == "__main__":
    script_start = time.time()
    args = handle_inputs()

    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    strategies = {
        "Naive FT (Adam)": [set_naive_finetune_adam_args],
        "Naive FT (SGD)": [set_naive_finetune_sgd_args],
        "ER (Buffer)": [set_er_buffer_args],
        "ER (Buffer)+Turnover": [set_er_buffer_args, set_turnover_args],
        "ER (Full Replay)": [set_er_full_replay_args],
        "ER (Reference Replay)": [set_er_reference_replay_args],
        "SI": [set_si_args],
        "SI+Turnover": [set_si_args, set_turnover_args],
    }
    if args.strategy_filter is not None:
        requested = []
        for token in args.strategy_filter:
            parts = [part.strip() for part in token.split(",") if part.strip()]
            requested.extend(parts)
        unknown = sorted(set(requested) - set(strategies.keys()))
        if len(unknown) > 0:
            raise ValueError(
                "Unknown entries in '--strategy-filter': {}. Available: {}".format(
                    ", ".join(unknown), ", ".join(strategies.keys())
                )
            )
        selected = [name for name in strategies if name in requested]
        if len(selected) == 0:
            raise ValueError(
                "'--strategy-filter' provided but no valid strategies selected."
            )
        strategies = {name: strategies[name] for name in selected}

    seed_list = list(range(args.seed, args.seed + args.n_seeds))
    base_metrics = {
        name: {
            "current_task_acc": [],
            "average_acc_so_far": [],
        }
        for name in strategies
    }
    drift_metrics_by_ref = {
        ref_context: {
            name: {
                "param_cos_similarity": [],
                "representational_cos_similarity": [],
            }
            for name in strategies
        }
        for ref_context in args.drift_ref_contexts
    }
    task_n_acc_by_ref = {
        ref_context: {name: [] for name in strategies}
        for ref_context in args.drift_ref_contexts
    }

    for name, config_fns in strategies.items():
        print("\n------{}------".format(name))
        strategy_start = time.time()
        for seed in seed_list:
            seed_start = time.time()
            print(" [{}] seed {} | preparing run args".format(name, seed))
            run_args = copy.deepcopy(args)
            for config_fn in config_fns:
                run_args = config_fn(run_args)
            run_args.seed = seed
            print(" [{}] seed {} | run_and_collect start".format(name, seed))
            plotting_dict = run_and_collect(run_args)
            print(" [{}] seed {} | extracting base metrics".format(name, seed))

            base_metrics[name]["current_task_acc"].append(
                extract_current_task_accuracy(
                    plotting_dict, n_contexts=run_args.contexts
                )
            )
            base_metrics[name]["average_acc_so_far"].append(
                extract_average_acc_so_far(plotting_dict, n_contexts=run_args.contexts)
            )
            param_cos = extract_series_by_context(
                {"x_context": plotting_dict["drift"]["x_context"]},
                plotting_dict["drift"]["param_cos_similarity"],
                n_contexts=run_args.contexts,
            )
            repr_cos = extract_series_by_context(
                {"x_context": plotting_dict["drift"]["x_context"]},
                plotting_dict["drift"]["representational_cos_similarity"],
                n_contexts=run_args.contexts,
            )
            for ref_context in args.drift_ref_contexts:
                print(
                    " [{}] seed {} | ref {} | recomputing task-n acc + drift".format(
                        name, seed, ref_context
                    )
                )
                if checkattr(run_args, "eval_all_contexts"):
                    task_n_acc = extract_series_by_context(
                        plotting_dict,
                        plotting_dict["acc per context"][
                            "context {}".format(ref_context)
                        ],
                        n_contexts=run_args.contexts,
                    )
                else:
                    task_n_acc = extract_series_over_contexts(
                        plotting_dict,
                        plotting_dict["acc per context"][
                            "context {}".format(ref_context)
                        ],
                        n_contexts=run_args.contexts,
                        fill_before_first=0.0,
                    )
                task_n_acc_by_ref[ref_context][name].append(task_n_acc)
                param_drift, repr_drift = recompute_drift_from_snapshots(
                    plotting_dict,
                    reference_context=ref_context,
                    n_contexts=run_args.contexts,
                )
                if param_drift is None:
                    param_drift = [
                        1.0 - val if not np.isnan(val) else np.nan for val in param_cos
                    ]
                if repr_drift is None:
                    repr_drift = [
                        1.0 - val if not np.isnan(val) else np.nan for val in repr_cos
                    ]
                drift_metrics_by_ref[ref_context][name]["param_cos_similarity"].append(
                    param_drift
                )
                drift_metrics_by_ref[ref_context][name][
                    "representational_cos_similarity"
                ].append(repr_drift)
            print(
                " [{}] seed {} | done in {:.1f}s".format(
                    name, seed, time.time() - seed_start
                )
            )
        print(
            " [{}] strategy done in {:.1f}s".format(name, time.time() - strategy_start)
        )

    contexts = list(range(1, args.contexts + 1))
    summary_base = {}
    for strategy_name in strategies:
        summary_base[strategy_name] = {}
        for metric_name in base_metrics[strategy_name]:
            mean_vals, ci_vals = aggregate_series(
                base_metrics[strategy_name][metric_name]
            )
            summary_base[strategy_name][metric_name] = {
                "mean": mean_vals,
                "ci": ci_vals,
            }

    summary_drift_by_ref = {}
    for ref_context in args.drift_ref_contexts:
        summary_drift_by_ref[ref_context] = {}
        for strategy_name in strategies:
            summary_drift_by_ref[ref_context][strategy_name] = {}
            for metric_name in drift_metrics_by_ref[ref_context][strategy_name]:
                mean_vals, ci_vals = aggregate_series(
                    drift_metrics_by_ref[ref_context][strategy_name][metric_name]
                )
                summary_drift_by_ref[ref_context][strategy_name][metric_name] = {
                    "mean": mean_vals,
                    "ci": ci_vals,
                }
    summary_task_n_by_ref = {}
    for ref_context in args.drift_ref_contexts:
        summary_task_n_by_ref[ref_context] = {}
        for strategy_name in strategies:
            mean_vals, ci_vals = aggregate_series(
                task_n_acc_by_ref[ref_context][strategy_name]
            )
            summary_task_n_by_ref[ref_context][strategy_name] = {
                "mean": mean_vals,
                "ci": ci_vals,
            }

    ref_label = "-".join([str(r) for r in args.drift_ref_contexts])
    base_pdf_name = "{}/drift-summary-{}{}-refs{}.pdf".format(
        args.p_dir, args.experiment, args.contexts, ref_label
    )
    pdf_name = _get_unique_pdf_path(base_pdf_name)
    pp = visual_plt.open_pdf(pdf_name)
    strategy_names = list(strategies.keys())
    color_map = plt.get_cmap("tab10")
    colors = [color_map(i % color_map.N) for i in range(len(strategy_names))]

    metric_specs = [
        ("current_task_acc", "Current-task accuracy", None),
        ("average_acc_so_far", "Average accuracy (all tasks so far)", (0.6, 1)),
        ("task_n_acc", "Accuracy on reference task", (0.6, 1)),
        ("param_cos_similarity", "Parameter drift (1 - cosine similarity)", (0, 1)),
        (
            "representational_cos_similarity",
            "Representational drift (1 - cosine similarity)",
            (0, 1),
        ),
    ]

    for ref_context in args.drift_ref_contexts:
        print(" plotting page for reference context {}".format(ref_context))
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for idx, (metric_name, ylabel, ylim) in enumerate(metric_specs):
            ax = axes[idx]
            for strategy_idx, strategy_name in enumerate(strategy_names):
                if metric_name in ("current_task_acc", "average_acc_so_far"):
                    mean_vals = summary_base[strategy_name][metric_name]["mean"]
                    ci_vals = summary_base[strategy_name][metric_name]["ci"]
                elif metric_name == "task_n_acc":
                    mean_vals = summary_task_n_by_ref[ref_context][strategy_name][
                        "mean"
                    ]
                    ci_vals = summary_task_n_by_ref[ref_context][strategy_name]["ci"]
                else:
                    mean_vals = summary_drift_by_ref[ref_context][strategy_name][
                        metric_name
                    ]["mean"]
                    ci_vals = summary_drift_by_ref[ref_context][strategy_name][
                        metric_name
                    ]["ci"]
                ax.plot(
                    contexts,
                    mean_vals,
                    linewidth=2.5,
                    color=colors[strategy_idx],
                    label=strategy_name,
                )
                if ci_vals is not None:
                    upper = np.array(mean_vals) + np.array(ci_vals)
                    lower = np.array(mean_vals) - np.array(ci_vals)
                    ax.fill_between(
                        contexts, upper, lower, color=colors[strategy_idx], alpha=0.2
                    )
            ax.set_title(ylabel)
            ax.set_xlabel("Context")
            ax.set_ylabel(ylabel)
            if metric_name == "current_task_acc":
                y_min = 0.0
                finite_vals = []
                for line in ax.get_lines():
                    ydata = np.array(line.get_ydata(), dtype=float)
                    finite_vals.extend(ydata[np.isfinite(ydata)].tolist())
                if len(finite_vals) > 0:
                    y_min = max(0.0, min(finite_vals) - 0.05)
                ax.set_ylim((y_min, 1.0))
            elif ylim is not None:
                ax.set_ylim(ylim)
            ax.grid(alpha=0.25)
            if metric_name == "task_n_acc":
                ax.axvline(
                    ref_context,
                    linestyle=":",
                    color="black",
                    linewidth=1.5,
                    alpha=0.8,
                )
            ax.legend()

        axes[5].axis("off")

        fig.suptitle(
            "permMNIST-{} | drift wrt task {}".format(args.contexts, ref_context)
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pp.savefig(fig)
        plt.close(fig)
    pp.close()

    print("\n" + "#" * 60)
    print(
        "DRIFT EXPERIMENT SUMMARY (permMNIST, {} contexts, ref task {})".format(
            args.contexts, args.drift_ref_context
        )
    )
    print("#" * 60)
    for strategy_name in strategy_names:
        print("\n{}:".format(strategy_name))
        for metric_name, _, _ in metric_specs:
            if metric_name in ("current_task_acc", "average_acc_so_far"):
                final_value = summary_base[strategy_name][metric_name]["mean"][-1]
            elif metric_name == "task_n_acc":
                final_value = summary_task_n_by_ref[args.drift_ref_contexts[0]][
                    strategy_name
                ]["mean"][-1]
            else:
                final_value = summary_drift_by_ref[args.drift_ref_contexts[0]][
                    strategy_name
                ][metric_name]["mean"][-1]
            print(" - {:32s} {:.4f}".format(metric_name, final_value))
    summary_rows = []
    for strategy_name in strategy_names:
        stage1_single_task_acc = summary_base[strategy_name]["current_task_acc"]["mean"][-1]
        final_avg_acc = summary_base[strategy_name]["average_acc_so_far"]["mean"][-1]
        ref_context = args.drift_ref_contexts[0]
        final_ref_task_acc = summary_task_n_by_ref[ref_context][strategy_name]["mean"][-1]
        final_param_drift = summary_drift_by_ref[ref_context][strategy_name][
            "param_cos_similarity"
        ]["mean"][-1]
        final_repr_drift = summary_drift_by_ref[ref_context][strategy_name][
            "representational_cos_similarity"
        ]["mean"][-1]
        summary_rows.append(
            {
                "strategy_name": strategy_name,
                "seed": args.seed,
                "n_seeds": args.n_seeds,
                "contexts": args.contexts,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "batch": args.batch,
                "iters": args.iters,
                "momentum": args.momentum if hasattr(args, "momentum") else None,
                "weight_decay": args.weight_decay if hasattr(args, "weight_decay") else 0.0,
                "adam_beta1": args.adam_beta1 if hasattr(args, "adam_beta1") else 0.9,
                "adam_beta2": args.adam_beta2 if hasattr(args, "adam_beta2") else 0.999,
                "adam_eps": args.adam_eps if hasattr(args, "adam_eps") else 1e-8,
                "drift_ref_context": ref_context,
                "stage1_single_task_acc": stage1_single_task_acc,
                "final_avg_acc": final_avg_acc,
                "final_ref_task_acc": final_ref_task_acc,
                "final_param_drift": final_param_drift,
                "final_repr_drift": final_repr_drift,
                "summary_pdf": pdf_name,
            }
        )
    summary_json = "{}/drift_run_summary.json".format(args.r_dir)
    with open(summary_json, "w") as f:
        json.dump({"rows": summary_rows}, f, indent=2)
    summary_jsonl = "{}/drift_run_summary.jsonl".format(args.r_dir)
    with open(summary_jsonl, "w") as f:
        for row in summary_rows:
            f.write(json.dumps(row) + "\n")
    print("Wrote run summary: {}".format(summary_json))
    print("\nGenerated plot: {}\n".format(pdf_name))
    plt.close("all")
    print("Total script time: {:.1f}s".format(time.time() - script_start))
