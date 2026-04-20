#!/usr/bin/env python3
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


def handle_inputs():
    kwargs = {"comparison": True}
    parser = options.define_args(
        filename="lop_experiment",
        description="Run LOP protocol via main.run and cache metrics in dict-*.pkl.",
    )
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    parser.add_argument("--tasks", type=int, default=800, help="number of permutation tasks")
    parser.add_argument(
        "--steps-per-task",
        type=int,
        default=60000,
        help="optimization steps per task/context",
    )
    parser.add_argument(
        "--lop-metric-samples",
        type=int,
        default=256,
        help="max samples used for dead-unit/effective-rank metrics per context",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="optional results-dir override (writes acc/dict cache files here)",
    )
    args = parser.parse_args()

    # LOP fixed setup using pure main.run orchestration.
    args.experiment = "permMNIST"
    args.scenario = "task"
    args.singlehead = True
    args.contexts = args.tasks
    args.iters = args.steps_per_task
    args.batch = 1
    args.replay = "none"
    args.results_dict = True
    args.pdf = False
    args.lop_metrics = True

    if args.output_dir is not None:
        args.r_dir = args.output_dir

    set_default_values(args, also_hyper_params=True)
    check_for_errors(args, **kwargs)
    return args


def _dict_file_prefix(args):
    param_stamp = get_param_stamp_from_args(args)
    suffix = "--S{}".format(args.eval_s) if checkattr(args, "gen_classifier") else ""
    dict_prefix = "{}/dict-{}--n{}{}".format(
        args.r_dir, param_stamp, "All" if args.acc_n is None else args.acc_n, suffix
    )
    return dict_prefix, param_stamp


def _acc_file(args, param_stamp):
    suffix = "--S{}".format(args.eval_s) if checkattr(args, "gen_classifier") else ""
    return "{}/acc-{}{}.txt".format(args.r_dir, param_stamp, suffix)


def _get_unique_path(base_path):
    if not os.path.isfile(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    counter = 2
    while True:
        candidate = "{}-v{}{}".format(root, counter, ext)
        if not os.path.isfile(candidate):
            return candidate
        counter += 1


def run_and_collect(args):
    start_time = time.time()
    dict_prefix, param_stamp = _dict_file_prefix(args)
    acc_file = _acc_file(args, param_stamp)

    if os.path.isfile(dict_prefix + ".pkl"):
        print(" already run (dict): {}".format(param_stamp))
    elif os.path.isfile(acc_file):
        print(" already run (acc): {}".format(param_stamp))
        args.train = False
        main.run(args)
    elif os.path.isfile("{}/mM-{}".format(args.m_dir, param_stamp)):
        print(" ...testing: {}".format(param_stamp))
        args.train = False
        main.run(args)
    else:
        print(" ...running: {}".format(param_stamp))
        args.train = True
        main.run(args)

    print(" loading plotting dict: {}.pkl".format(dict_prefix))
    plotting_dict = utils.load_object(dict_prefix)
    print(" run_and_collect finished in {:.1f}s".format(time.time() - start_time))
    return plotting_dict, param_stamp


def _extract_series_by_context(plotting_dict, section, field, n_contexts):
    x_context = plotting_dict.get(section, {}).get("x_context", [])
    values = plotting_dict.get(section, {}).get(field, [])
    output = []
    for context_id in range(1, n_contexts + 1):
        indices = [i for i, x in enumerate(x_context) if x == context_id]
        output.append(values[indices[-1]] if len(indices) > 0 else np.nan)
    return output


def _extract_acc_series(plotting_dict, n_contexts):
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


def plot_lop_metrics(plotting_dict, args, param_stamp):
    os.makedirs(args.p_dir, exist_ok=True)
    contexts = list(range(1, args.contexts + 1))

    acc = _extract_acc_series(plotting_dict, args.contexts)
    dead_units = _extract_series_by_context(plotting_dict, "lop", "dead_unit_fraction", args.contexts)
    effective_rank = _extract_series_by_context(plotting_dict, "lop", "effective_rank", args.contexts)
    weight_mag = _extract_series_by_context(plotting_dict, "lop", "weight_magnitude", args.contexts)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_acc, ax_dead, ax_rank, ax_weight = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_acc.plot(contexts, acc, label="current_task_acc")
    ax_acc.set_title("Current task accuracy")
    ax_acc.set_xlabel("task")

    ax_dead.plot(contexts, dead_units, label="dead_unit_fraction", color="tab:red")
    ax_dead.set_title("Dead unit fraction")
    ax_dead.set_xlabel("task")

    ax_rank.plot(contexts, effective_rank, label="effective_rank", color="tab:green")
    ax_rank.set_title("Effective rank")
    ax_rank.set_xlabel("task")

    ax_weight.plot(contexts, weight_mag, label="weight_magnitude", color="tab:purple")
    ax_weight.set_title("Weight magnitude")
    ax_weight.set_xlabel("task")

    for ax in (ax_acc, ax_dead, ax_rank, ax_weight):
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("LOP summary ({})".format(param_stamp), fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    plot_path = _get_unique_path(os.path.join(args.p_dir, "lop-summary-{}.png".format(param_stamp)))
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print("Generated plot: {}".format(plot_path))


if __name__ == "__main__":
    script_start = time.time()
    args = handle_inputs()

    os.makedirs(args.r_dir, exist_ok=True)
    os.makedirs(args.p_dir, exist_ok=True)

    plotting_dict, param_stamp = run_and_collect(args)
    plot_lop_metrics(plotting_dict, args, param_stamp)

    print("\nDone in {:.1f}s".format(time.time() - script_start))
