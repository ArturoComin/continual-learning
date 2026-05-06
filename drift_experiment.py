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


def _log_progress(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[{}] {}".format(stamp, message), flush=True)


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
    parser.add_argument(
        "--lop-metric-samples",
        type=int,
        default=256,
        help="max samples used for dead-unit/effective-rank metrics per context",
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
    args.lop_metrics = True

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


def _has_valid_lop_data(plotting_dict, expected_contexts):
    if not isinstance(plotting_dict, dict):
        return False
    lop_dict = plotting_dict.get("lop")
    if not isinstance(lop_dict, dict):
        return False
    x_context = lop_dict.get("x_context", [])
    if len(x_context) == 0:
        return False
    for field in ("dead_unit_fraction", "effective_rank", "weight_magnitude"):
        values = lop_dict.get(field, [])
        if len(values) != len(x_context):
            return False
    if expected_contexts is not None and len(set(x_context)) < expected_contexts:
        return False
    return True


def run_and_collect(args):
    start_time = time.time()
    dict_prefix, param_stamp = _dict_file_prefix(args)
    acc_file = _acc_file(args, param_stamp)
    model_file = "{}/mM-{}".format(args.m_dir, param_stamp)
    _log_progress("run start: {}".format(param_stamp))

    if os.path.isfile(dict_prefix + ".pkl"):
        _log_progress("already run (dict): {}".format(param_stamp))
        plotting_dict = utils.load_object(dict_prefix)
        needs_rebuild = not _has_valid_plotting_data(
            plotting_dict, expected_contexts=args.contexts
        )
        if checkattr(args, "lop_metrics"):
            needs_rebuild = needs_rebuild or not _has_valid_lop_data(
                plotting_dict, expected_contexts=args.contexts
            )
        if needs_rebuild:
            _log_progress(
                "cached dict invalid/empty -> rebuilding: {}.pkl".format(dict_prefix)
            )
            os.remove(dict_prefix + ".pkl")
            args.train = True
            main.run(args)
    elif os.path.isfile(acc_file) or os.path.isfile(model_file):
        source = "acc" if os.path.isfile(acc_file) else "checkpoint"
        _log_progress(
            " found {} but missing dict -> rerunning training: {}".format(
                source, param_stamp
            )
        )
        args.train = True
        main.run(args)
    else:
        _log_progress("running: {}".format(param_stamp))
        args.train = True
        main.run(args)

    _log_progress("loading plotting dict: {}.pkl".format(dict_prefix))
    plotting_dict = utils.load_object(dict_prefix)
    _log_progress("run finished in {:.1f}s: {}".format(time.time() - start_time, param_stamp))
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


def _center_features(x):
    return x - x.mean(axis=0, keepdims=True)


def _linear_cka(x, y, eps=1e-12):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return np.nan
    x = _center_features(x[:n])
    y = _center_features(y[:n])
    hsic_xy = np.linalg.norm(np.matmul(x.T, y), ord="fro") ** 2
    hsic_xx = np.linalg.norm(np.matmul(x.T, x), ord="fro") ** 2
    hsic_yy = np.linalg.norm(np.matmul(y.T, y), ord="fro") ** 2
    denom = np.sqrt(max(hsic_xx * hsic_yy, eps))
    if denom <= 0:
        return np.nan
    return float(hsic_xy / denom)


def _orthogonal_procrustes_distance(x, y, eps=1e-12):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return np.nan
    x = _center_features(x[:n])
    y = _center_features(y[:n])
    x_norm = np.linalg.norm(x, ord="fro")
    y_norm = np.linalg.norm(y, ord="fro")
    if x_norm <= eps or y_norm <= eps:
        return np.nan
    x = x / x_norm
    y = y / y_norm
    u, _, vt = np.linalg.svd(np.matmul(x.T, y), full_matrices=False)
    rot = np.matmul(u, vt)
    return float(np.linalg.norm(np.matmul(x, rot) - y, ord="fro") / np.sqrt(max(n, 1)))


def _vector_to_matrix(vector, chunk_size=512):
    x = np.array(vector, dtype=float).reshape(-1)
    if x.size == 0:
        return None
    n_chunks = int(np.ceil(float(x.size) / float(chunk_size)))
    padded = n_chunks * chunk_size
    if padded > x.size:
        x = np.pad(x, (0, padded - x.size), mode="constant")
    return x.reshape(n_chunks, chunk_size)


def _pearson_corr(x, y, eps=1e-12):
    x = np.array(x, dtype=float).reshape(-1)
    y = np.array(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= eps:
        return np.nan
    return float(np.dot(x, y) / denom)


def _cross_validated_rdm(features, labels, seed=0):
    features = np.array(features, dtype=float)
    labels = np.array(labels)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        return None, None
    rng = np.random.RandomState(seed)
    means_a = {}
    means_b = {}
    valid = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if idx.size < 2:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        cut = idx.size // 2
        a_idx, b_idx = idx[:cut], idx[cut:]
        if a_idx.size == 0 or b_idx.size == 0:
            continue
        means_a[int(label)] = features[a_idx].mean(axis=0)
        means_b[int(label)] = features[b_idx].mean(axis=0)
        valid.append(int(label))
    if len(valid) < 2:
        return None, None
    valid = sorted(valid)
    rdm = np.zeros((len(valid), len(valid)), dtype=float)
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            la, lb = valid[i], valid[j]
            diff_a = means_a[la] - means_a[lb]
            diff_b = means_b[la] - means_b[lb]
            dist = float(np.dot(diff_a, diff_b))
            rdm[i, j] = dist
            rdm[j, i] = dist
    return rdm, valid


def _cv_rsa_similarity(ref_features, ref_labels, cur_features, cur_labels):
    ref_rdm, ref_valid = _cross_validated_rdm(ref_features, ref_labels, seed=0)
    cur_rdm, cur_valid = _cross_validated_rdm(cur_features, cur_labels, seed=1)
    if ref_rdm is None or cur_rdm is None:
        return np.nan
    common = sorted(set(ref_valid).intersection(set(cur_valid)))
    if len(common) < 2:
        return np.nan
    r_map = {label: i for i, label in enumerate(ref_valid)}
    c_map = {label: i for i, label in enumerate(cur_valid)}
    ref_sub = np.array([[ref_rdm[r_map[i], r_map[j]] for j in common] for i in common], dtype=float)
    cur_sub = np.array([[cur_rdm[c_map[i], c_map[j]] for j in common] for i in common], dtype=float)
    idx = np.triu_indices(len(common), k=1)
    return _pearson_corr(ref_sub[idx], cur_sub[idx])


def recompute_drift_from_snapshots(plotting_dict, reference_context, n_contexts):
    snapshots = plotting_dict.get("drift", {}).get("snapshots", {})
    ref_key = str(reference_context)
    if ref_key not in snapshots:
        return None
    ref_state = snapshots[ref_key]
    drift_values = {
        "param_cos_similarity": [],
        "representational_cos_similarity": [],
        "param_cka_similarity": [],
        "representational_cka_similarity": [],
        "param_procrustes_distance": [],
        "representational_procrustes_distance": [],
        "representational_cv_rsa_similarity": [],
    }
    for context_id in range(1, n_contexts + 1):
        key = str(context_id)
        if key not in snapshots:
            for metric_name in drift_values:
                drift_values[metric_name].append(np.nan)
            continue
        cur_state = snapshots[key]
        cur_param = cur_state.get("param_vector")
        ref_param = ref_state.get("param_vector")
        if cur_param is None or ref_param is None:
            drift_values["param_cos_similarity"].append(np.nan)
            drift_values["param_cka_similarity"].append(np.nan)
            drift_values["param_procrustes_distance"].append(np.nan)
        else:
            drift_values["param_cos_similarity"].append(1.0 - _cosine(cur_param, ref_param))
            cur_param_matrix = _vector_to_matrix(cur_param)
            ref_param_matrix = _vector_to_matrix(ref_param)
            drift_values["param_cka_similarity"].append(
                1.0 - _linear_cka(cur_param_matrix, ref_param_matrix)
            )
            drift_values["param_procrustes_distance"].append(
                _orthogonal_procrustes_distance(cur_param_matrix, ref_param_matrix)
            )
        repr_cos_values = []
        repr_cka_values = []
        repr_pro_values = []
        repr_rsa_values = []
        cur_repr_vectors = cur_state.get("repr_vectors", [])
        ref_repr_vectors = ref_state.get("repr_vectors", [])
        cur_repr_mats = cur_state.get("repr_matrices", [])
        ref_repr_mats = ref_state.get("repr_matrices", [])
        cur_repr_labels = cur_state.get("repr_labels", [])
        ref_repr_labels = ref_state.get("repr_labels", [])
        for dataset_id in range(context_id):
            if dataset_id >= len(cur_repr_vectors) or dataset_id >= len(ref_repr_vectors):
                continue
            cur_repr = cur_repr_vectors[dataset_id]
            ref_repr = ref_repr_vectors[dataset_id]
            if (cur_repr is None) or (ref_repr is None):
                continue
            repr_cos_values.append(1.0 - _cosine(cur_repr, ref_repr))
            if dataset_id < len(cur_repr_mats) and dataset_id < len(ref_repr_mats):
                cur_mat = cur_repr_mats[dataset_id]
                ref_mat = ref_repr_mats[dataset_id]
                if cur_mat is not None and ref_mat is not None:
                    repr_cka_values.append(1.0 - _linear_cka(cur_mat, ref_mat))
                    repr_pro_values.append(_orthogonal_procrustes_distance(cur_mat, ref_mat))
                    if dataset_id < len(cur_repr_labels) and dataset_id < len(ref_repr_labels):
                        cur_labels = cur_repr_labels[dataset_id]
                        ref_labels = ref_repr_labels[dataset_id]
                        if cur_labels is not None and ref_labels is not None:
                            repr_rsa_values.append(
                                1.0 - _cv_rsa_similarity(ref_mat, ref_labels, cur_mat, cur_labels)
                            )
        drift_values["representational_cos_similarity"].append(
            float(np.nanmean(repr_cos_values)) if len(repr_cos_values) > 0 else np.nan
        )
        drift_values["representational_cka_similarity"].append(
            float(np.nanmean(repr_cka_values)) if len(repr_cka_values) > 0 else np.nan
        )
        drift_values["representational_procrustes_distance"].append(
            float(np.nanmean(repr_pro_values)) if len(repr_pro_values) > 0 else np.nan
        )
        drift_values["representational_cv_rsa_similarity"].append(
            float(np.nanmean(repr_rsa_values)) if len(repr_rsa_values) > 0 else np.nan
        )
    return drift_values


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
    return args


if __name__ == "__main__":
    script_start = time.time()
    args = handle_inputs()

    os.makedirs(args.r_dir, exist_ok=True)
    os.makedirs(args.p_dir, exist_ok=True)

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
            "dead_unit_fraction": [],
            "effective_rank": [],
            "weight_magnitude": [],
        }
        for name in strategies
    }
    drift_metrics_by_ref = {
        ref_context: {
            name: {
                "param_cos_similarity": [],
                "representational_cos_similarity": [],
                "param_cka_similarity": [],
                "representational_cka_similarity": [],
                "param_procrustes_distance": [],
                "representational_procrustes_distance": [],
                "representational_cv_rsa_similarity": [],
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
        _log_progress("strategy start: {}".format(name))
        strategy_start = time.time()
        for seed in seed_list:
            seed_start = time.time()
            _log_progress("[{}] seed {} | preparing run args".format(name, seed))
            run_args = copy.deepcopy(args)
            for config_fn in config_fns:
                run_args = config_fn(run_args)
            run_args.seed = seed
            _log_progress("[{}] seed {} | run_and_collect start".format(name, seed))
            plotting_dict = run_and_collect(run_args)
            _log_progress("[{}] seed {} | extracting base metrics".format(name, seed))

            base_metrics[name]["current_task_acc"].append(
                extract_current_task_accuracy(
                    plotting_dict, n_contexts=run_args.contexts
                )
            )
            base_metrics[name]["average_acc_so_far"].append(
                extract_average_acc_so_far(plotting_dict, n_contexts=run_args.contexts)
            )
            lop_x_context = plotting_dict.get("lop", {}).get("x_context", [])
            dead_units = extract_series_by_context(
                {"x_context": lop_x_context},
                plotting_dict.get("lop", {}).get(
                    "dead_unit_fraction", [np.nan] * len(lop_x_context)
                ),
                n_contexts=run_args.contexts,
            )
            effective_rank = extract_series_by_context(
                {"x_context": lop_x_context},
                plotting_dict.get("lop", {}).get(
                    "effective_rank", [np.nan] * len(lop_x_context)
                ),
                n_contexts=run_args.contexts,
            )
            weight_magnitude = extract_series_by_context(
                {"x_context": lop_x_context},
                plotting_dict.get("lop", {}).get(
                    "weight_magnitude", [np.nan] * len(lop_x_context)
                ),
                n_contexts=run_args.contexts,
            )
            base_metrics[name]["dead_unit_fraction"].append(dead_units)
            base_metrics[name]["effective_rank"].append(effective_rank)
            base_metrics[name]["weight_magnitude"].append(weight_magnitude)
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
            param_cka = extract_series_by_context(
                {"x_context": plotting_dict["drift"]["x_context"]},
                plotting_dict["drift"].get("param_cka_similarity", [np.nan] * len(plotting_dict["drift"]["x_context"])),
                n_contexts=run_args.contexts,
            )
            repr_cka = extract_series_by_context(
                {"x_context": plotting_dict["drift"]["x_context"]},
                plotting_dict["drift"].get(
                    "representational_cka_similarity", [np.nan] * len(plotting_dict["drift"]["x_context"])
                ),
                n_contexts=run_args.contexts,
            )
            param_procrustes = extract_series_by_context(
                {"x_context": plotting_dict["drift"]["x_context"]},
                plotting_dict["drift"].get(
                    "param_procrustes_distance", [np.nan] * len(plotting_dict["drift"]["x_context"])
                ),
                n_contexts=run_args.contexts,
            )
            repr_procrustes = extract_series_by_context(
                {"x_context": plotting_dict["drift"]["x_context"]},
                plotting_dict["drift"].get(
                    "representational_procrustes_distance", [np.nan] * len(plotting_dict["drift"]["x_context"])
                ),
                n_contexts=run_args.contexts,
            )
            repr_cv_rsa = extract_series_by_context(
                {"x_context": plotting_dict["drift"]["x_context"]},
                plotting_dict["drift"].get(
                    "representational_cv_rsa_similarity", [np.nan] * len(plotting_dict["drift"]["x_context"])
                ),
                n_contexts=run_args.contexts,
            )
            for ref_context in args.drift_ref_contexts:
                _log_progress(
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
                recomputed = recompute_drift_from_snapshots(
                    plotting_dict,
                    reference_context=ref_context,
                    n_contexts=run_args.contexts,
                )
                fallback = {
                    "param_cos_similarity": [1.0 - val if not np.isnan(val) else np.nan for val in param_cos],
                    "representational_cos_similarity": [1.0 - val if not np.isnan(val) else np.nan for val in repr_cos],
                    "param_cka_similarity": [1.0 - val if not np.isnan(val) else np.nan for val in param_cka],
                    "representational_cka_similarity": [1.0 - val if not np.isnan(val) else np.nan for val in repr_cka],
                    "param_procrustes_distance": param_procrustes,
                    "representational_procrustes_distance": repr_procrustes,
                    "representational_cv_rsa_similarity": [
                        1.0 - val if not np.isnan(val) else np.nan for val in repr_cv_rsa
                    ],
                }
                drift_sources = fallback if recomputed is None else recomputed
                for metric_name in drift_metrics_by_ref[ref_context][name]:
                    drift_metrics_by_ref[ref_context][name][metric_name].append(
                        drift_sources.get(metric_name, fallback[metric_name])
                    )
            _log_progress(
                " [{}] seed {} | done in {:.1f}s".format(
                    name, seed, time.time() - seed_start
                )
            )
        _log_progress(
            "strategy done: {} | took {:.1f}s".format(name, time.time() - strategy_start)
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
        ("dead_unit_fraction", "Dead unit fraction", (0, 1)),
        ("effective_rank", "Effective rank", None),
        ("weight_magnitude", "Weight magnitude (mean abs)", None),
        ("param_cos_similarity", "Parameter drift (1 - cosine similarity)", (0, 1)),
        (
            "representational_cos_similarity",
            "Representational drift (1 - cosine similarity)",
            (0, 1),
        ),
        ("param_cka_similarity", "Parameter drift (1 - CKA similarity)", (0, 1)),
        ("representational_cka_similarity", "Representational drift (1 - CKA similarity)", (0, 1)),
        ("param_procrustes_distance", "Parameter drift (Procrustes distance)", None),
        ("representational_procrustes_distance", "Representational drift (Procrustes distance)", None),
        ("representational_cv_rsa_similarity", "Representational drift (1 - CV-RSA similarity)", None),
    ]

    for ref_context in args.drift_ref_contexts:
        print(" plotting page for reference context {}".format(ref_context))
        n_cols = 3
        n_rows = int(np.ceil(float(len(metric_specs)) / float(n_cols)))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        for idx, (metric_name, ylabel, ylim) in enumerate(metric_specs):
            ax = axes[idx]
            for strategy_idx, strategy_name in enumerate(strategy_names):
                if metric_name in ("current_task_acc", "average_acc_so_far"):
                    mean_vals = summary_base[strategy_name][metric_name]["mean"]
                    ci_vals = summary_base[strategy_name][metric_name]["ci"]
                elif metric_name in (
                    "dead_unit_fraction",
                    "effective_rank",
                    "weight_magnitude",
                ):
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

        if len(metric_specs) < len(axes):
            for hide_idx in range(len(metric_specs), len(axes)):
                axes[hide_idx].axis("off")

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
            elif metric_name in (
                "dead_unit_fraction",
                "effective_rank",
                "weight_magnitude",
            ):
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
        final_dead_unit_fraction = summary_base[strategy_name]["dead_unit_fraction"][
            "mean"
        ][-1]
        final_effective_rank = summary_base[strategy_name]["effective_rank"]["mean"][-1]
        final_weight_magnitude = summary_base[strategy_name]["weight_magnitude"]["mean"][
            -1
        ]
        ref_context = args.drift_ref_contexts[0]
        final_ref_task_acc = summary_task_n_by_ref[ref_context][strategy_name]["mean"][-1]
        final_param_drift = summary_drift_by_ref[ref_context][strategy_name][
            "param_cos_similarity"
        ]["mean"][-1]
        final_repr_drift = summary_drift_by_ref[ref_context][strategy_name][
            "representational_cos_similarity"
        ]["mean"][-1]
        final_param_cka_drift = summary_drift_by_ref[ref_context][strategy_name][
            "param_cka_similarity"
        ]["mean"][-1]
        final_repr_cka_drift = summary_drift_by_ref[ref_context][strategy_name][
            "representational_cka_similarity"
        ]["mean"][-1]
        final_param_procrustes = summary_drift_by_ref[ref_context][strategy_name][
            "param_procrustes_distance"
        ]["mean"][-1]
        final_repr_procrustes = summary_drift_by_ref[ref_context][strategy_name][
            "representational_procrustes_distance"
        ]["mean"][-1]
        final_repr_cv_rsa_drift = summary_drift_by_ref[ref_context][strategy_name][
            "representational_cv_rsa_similarity"
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
                "final_dead_unit_fraction": final_dead_unit_fraction,
                "final_effective_rank": final_effective_rank,
                "final_weight_magnitude": final_weight_magnitude,
                "final_ref_task_acc": final_ref_task_acc,
                "final_param_drift": final_param_drift,
                "final_repr_drift": final_repr_drift,
                "final_param_cka_drift": final_param_cka_drift,
                "final_repr_cka_drift": final_repr_cka_drift,
                "final_param_procrustes_distance": final_param_procrustes,
                "final_repr_procrustes_distance": final_repr_procrustes,
                "final_repr_cv_rsa_drift": final_repr_cv_rsa_drift,
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
