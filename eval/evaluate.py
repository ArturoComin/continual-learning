import numpy as np
import torch
from visual import visual_plt
from visual import visual_visdom
from utils import get_data_loader,checkattr


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####

def test_acc(model, dataset, batch_size=128, test_size=1024, verbose=True, context_id=None, allowed_classes=None,
             no_context_mask=False, **kwargs):
    '''Evaluate accuracy (= proportion of samples classified correctly) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model.device if hasattr(model, 'device') else model._device()
    cuda = model.cuda if hasattr(model, 'cuda') else model._is_on_cuda()

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Apply context-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_context_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(context=context_id+1)

    # Should output-labels be adjusted for allowed classes? (ASSUMPTION: [allowed_classes] has consecutive numbers)
    label_correction = 0 if checkattr(model, 'stream_classifier') or (allowed_classes is None) else allowed_classes[0]

    # If there is a separate network per context, select the correct subnetwork
    if model.label=="SeparateClassifiers":
        model = getattr(model, 'context{}'.format(context_id+1))
        allowed_classes = None

    # Loop over batches in [dataset]
    data_loader = get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for x, y in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -if the model is a "stream-classifier", add context
        if checkattr(model, 'stream_classifier'):
            context_tensor = torch.tensor([context_id]*x.shape[0]).to(device)
        # -evaluate model (if requested, only on [allowed_classes])
        with torch.no_grad():
            if checkattr(model, 'stream_classifier'):
                scores = model.classify(x.to(device), context=context_tensor)
            else:
                scores = model.classify(x.to(device), allowed_classes=allowed_classes)
        _, predicted = torch.max(scores.cpu(), 1)
        if model.prototypes and max(predicted).item() >= model.classes:
            # -in case of Domain-IL (or Task-IL + singlehead), collapse all corresponding domains to same class
            predicted = predicted % model.classes
        # -update statistics
        y = y-label_correction
        total_correct += (predicted == y).sum().item()
        total_tested += len(x)
    accuracy = total_correct / total_tested

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    if verbose:
        print('=> accuracy: {:.3f}'.format(accuracy))
    return accuracy


def test_all_so_far(model, datasets, current_context, iteration, test_size=None, no_context_mask=False,
                    visdom=None, summary_graph=True, plotting_dict=None, verbose=False):
    '''Evaluate accuracy of a classifier (=[model]) on all contexts so far (= up to [current_context]) using [datasets].

    [visdom]      None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    n_contexts = len(datasets)

    # Evaluate accuracy of model predictions
    # - in the academic CL setting:  for all contexts so far, reporting "0" for future contexts
    # - in task-free stream setting (current_context==None): always for all contexts
    precs = []
    for i in range(n_contexts):
        if (current_context is None) or (i+1 <= current_context):
            allowed_classes = None
            if model.scenario=='task' and not checkattr(model, 'singlehead'):
                allowed_classes = list(range(model.classes_per_context * i, model.classes_per_context * (i + 1)))
            precs.append(test_acc(model, datasets[i], test_size=test_size, verbose=verbose,
                                  allowed_classes=allowed_classes, no_context_mask=no_context_mask, context_id=i))
        else:
            precs.append(0)
    if current_context is None:
        current_context = i+1
    average_precs = sum([precs[context_id] for context_id in range(current_context)]) / current_context

    # Print results on screen
    if verbose:
        print(' => ave accuracy: {:.3f}'.format(average_precs))

    # Add results to [plotting_dict]
    if plotting_dict is not None:
        for i in range(n_contexts):
            plotting_dict['acc per context']['context {}'.format(i+1)].append(precs[i])
        plotting_dict['average'].append(average_precs)
        plotting_dict['x_iteration'].append(iteration)
        plotting_dict['x_context'].append(current_context)

    # Send results to visdom server
    names = ['context {}'.format(i + 1) for i in range(n_contexts)]
    if visdom is not None:
        visual_visdom.visualize_scalars(
            precs, names=names, title="accuracy ({})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylabel="test accuracy"
        )
        if n_contexts>1 and summary_graph:
            visual_visdom.visualize_scalars(
                [average_precs], names=["ave"], title="ave accuracy ({})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylabel="test accuracy"
            )


def test_all(model, datasets, current_context, iteration, test_size=None, no_context_mask=False,
             visdom=None, summary_graph=True, plotting_dict=None, verbose=False):
    '''Evaluate accuracy of a classifier (=[model]) on all contexts in [datasets], irrespective of [current_context].

    [visdom]      None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    n_contexts = len(datasets)

    precs = []
    for i in range(n_contexts):
        allowed_classes = None
        if model.scenario == 'task' and not checkattr(model, 'singlehead'):
            allowed_classes = list(range(model.classes_per_context * i, model.classes_per_context * (i + 1)))
        precs.append(test_acc(
            model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=allowed_classes,
            no_context_mask=no_context_mask, context_id=i
        ))

    if current_context is None:
        current_context = n_contexts
    average_precs = sum(precs) / n_contexts

    # Print results on screen
    if verbose:
        print(' => ave accuracy: {:.3f}'.format(average_precs))

    # Add results to [plotting_dict]
    if plotting_dict is not None:
        for i in range(n_contexts):
            plotting_dict['acc per context']['context {}'.format(i+1)].append(precs[i])
        plotting_dict['average'].append(average_precs)
        plotting_dict['x_iteration'].append(iteration)
        plotting_dict['x_context'].append(current_context)

    # Send results to visdom server
    names = ['context {}'.format(i + 1) for i in range(n_contexts)]
    if visdom is not None:
        visual_visdom.visualize_scalars(
            precs, names=names, title="accuracy ({})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylabel="test accuracy"
        )
        if n_contexts > 1 and summary_graph:
            visual_visdom.visualize_scalars(
                [average_precs], names=["ave"], title="ave accuracy ({})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylabel="test accuracy"
            )


def initiate_plotting_dict(n_contexts):
    '''Initiate <dict> with accuracy-measures to keep track of for plotting.'''
    plotting_dict = {}
    plotting_dict["acc per context"] = {}
    for i in range(n_contexts):
        plotting_dict["acc per context"]["context {}".format(i+1)] = []
    plotting_dict["average"] = []      # average accuracy over all contexts so far: Task-IL  -> only classes in context
                                       #                                            Class-IL -> all classes so far
    plotting_dict["x_iteration"] = []  # total number of iterations so far
    plotting_dict["x_context"] = []    # number of contexts so far (i.e., context on which training just finished)
    plotting_dict["online_acc"] = {
        "per_context": [],
        "x_context": [],
        "x_iteration": [],
    }
    plotting_dict["drift"] = {
        "param_cos_similarity": [],
        "representational_cos_similarity": [],
        "param_cka_similarity": [],
        "representational_cka_similarity": [],
        "param_procrustes_distance": [],
        "representational_procrustes_distance": [],
        "representational_cv_rsa_similarity": [],
        "x_iteration": [],
        "x_context": [],
        "snapshots": {},
    }
    plotting_dict["lop"] = {
        "dead_unit_fraction": [],
        "effective_rank": [],
        "weight_magnitude": [],
        "x_iteration": [],
        "x_context": [],
    }
    return plotting_dict


def _extract_model_for_context(model, context_id):
    if model.label == "SeparateClassifiers":
        return getattr(model, 'context{}'.format(context_id + 1))
    return model


def _get_parameter_vector(model):
    parameter_list = [param.detach().view(-1).float().cpu() for param in model.parameters() if param.requires_grad]
    if len(parameter_list) == 0:
        parameter_list = [param.detach().view(-1).float().cpu() for param in model.parameters()]
    return torch.cat(parameter_list, dim=0)


def _compute_representation_matrix(model, dataset, context_id, max_samples=256, batch_size=128):
    eval_model = _extract_model_for_context(model, context_id)
    device = eval_model.device if hasattr(eval_model, 'device') else eval_model._device()
    cuda = eval_model.cuda if hasattr(eval_model, 'cuda') else eval_model._is_on_cuda()

    mode = eval_model.training
    eval_model.eval()

    if hasattr(eval_model, "mask_dict") and eval_model.mask_dict is not None:
        eval_model.apply_XdGmask(context=context_id + 1)

    feature_batches = []
    label_batches = []
    n_collected = 0
    data_loader = get_data_loader(dataset, batch_size=batch_size, cuda=cuda)
    for x, y in data_loader:
        if n_collected >= max_samples:
            break
        x = x.to(device)
        with torch.no_grad():
            if checkattr(eval_model, 'stream_classifier'):
                context_tensor = torch.tensor([context_id] * x.shape[0]).to(device)
                features = eval_model.feature_extractor(x, context=context_tensor)
            else:
                features = eval_model.feature_extractor(x)
        features = features.view(features.shape[0], -1).float().cpu()
        feature_batches.append(features)
        label_batches.append(y.detach().long().cpu())
        n_collected += features.shape[0]

    eval_model.train(mode=mode)

    if len(feature_batches) == 0:
        return None, None
    all_features = torch.cat(feature_batches, dim=0)[:max_samples]
    all_labels = torch.cat(label_batches, dim=0)[:max_samples]
    return all_features, all_labels


def _compute_representation_vector(model, dataset, context_id, max_samples=256, batch_size=128):
    feature_matrix, _ = _compute_representation_matrix(
        model, dataset, context_id=context_id, max_samples=max_samples, batch_size=batch_size
    )
    if feature_matrix is None:
        return None
    return feature_matrix.mean(dim=0)


def _center_features(x):
    return x - x.mean(axis=0, keepdims=True)


def _linear_cka(x, y, eps=1e-12):
    if x is None or y is None:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return float("nan")
    x = _center_features(x[:n])
    y = _center_features(y[:n])
    hsic_xy = np.linalg.norm(np.matmul(x.T, y), ord="fro") ** 2
    hsic_xx = np.linalg.norm(np.matmul(x.T, x), ord="fro") ** 2
    hsic_yy = np.linalg.norm(np.matmul(y.T, y), ord="fro") ** 2
    denom = np.sqrt(max(hsic_xx * hsic_yy, eps))
    return float(hsic_xy / denom) if denom > 0 else float("nan")


def _orthogonal_procrustes_distance(x, y, eps=1e-12):
    if x is None or y is None:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return float("nan")
    x = _center_features(x[:n])
    y = _center_features(y[:n])
    x_norm = np.linalg.norm(x, ord="fro")
    y_norm = np.linalg.norm(y, ord="fro")
    if x_norm <= eps or y_norm <= eps:
        return float("nan")
    x = x / x_norm
    y = y / y_norm
    cross_cov = np.matmul(x.T, y)
    u, _, vt = np.linalg.svd(cross_cov, full_matrices=False)
    rotation = np.matmul(u, vt)
    residual = np.linalg.norm(np.matmul(x, rotation) - y, ord="fro")
    return float(residual / np.sqrt(max(n, 1)))


def _vector_to_matrix(vector, chunk_size=512):
    x = np.asarray(vector, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return None
    n_chunks = int(np.ceil(float(x.size) / float(chunk_size)))
    padded_size = n_chunks * chunk_size
    if padded_size > x.size:
        x = np.pad(x, (0, padded_size - x.size), mode="constant")
    return x.reshape(n_chunks, chunk_size)


def _pearson_corr(x, y, eps=1e-12):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= eps:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _cosine_similarity(x, y, eps=1e-12):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return float("nan")
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= eps:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _cross_validated_rdm(features, labels, rng_seed=0):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        return None, None
    unique_labels = np.unique(labels)
    rng = np.random.RandomState(rng_seed)
    split_a = {}
    split_b = {}
    valid_labels = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if idx.size < 2:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        split_point = idx.size // 2
        a_idx = idx[:split_point]
        b_idx = idx[split_point:]
        if a_idx.size == 0 or b_idx.size == 0:
            continue
        split_a[int(label)] = features[a_idx].mean(axis=0)
        split_b[int(label)] = features[b_idx].mean(axis=0)
        valid_labels.append(int(label))
    if len(valid_labels) < 2:
        return None, None
    valid_labels = sorted(valid_labels)
    n_labels = len(valid_labels)
    rdm = np.zeros((n_labels, n_labels), dtype=np.float64)
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            li = valid_labels[i]
            lj = valid_labels[j]
            diff_a = split_a[li] - split_a[lj]
            diff_b = split_b[li] - split_b[lj]
            dist = float(np.dot(diff_a, diff_b))
            rdm[i, j] = dist
            rdm[j, i] = dist
    return rdm, valid_labels


def _cv_rsa_similarity(ref_features, ref_labels, cur_features, cur_labels):
    ref_rdm, ref_valid_labels = _cross_validated_rdm(ref_features, ref_labels, rng_seed=0)
    cur_rdm, cur_valid_labels = _cross_validated_rdm(cur_features, cur_labels, rng_seed=1)
    if ref_rdm is None or cur_rdm is None:
        return float("nan")
    common_labels = sorted(set(ref_valid_labels).intersection(set(cur_valid_labels)))
    if len(common_labels) < 2:
        return float("nan")
    ref_map = {label: idx for idx, label in enumerate(ref_valid_labels)}
    cur_map = {label: idx for idx, label in enumerate(cur_valid_labels)}
    ref_sub = np.array(
        [[ref_rdm[ref_map[i], ref_map[j]] for j in common_labels] for i in common_labels],
        dtype=np.float64,
    )
    cur_sub = np.array(
        [[cur_rdm[cur_map[i], cur_map[j]] for j in common_labels] for i in common_labels],
        dtype=np.float64,
    )
    triu_idx = np.triu_indices(len(common_labels), k=1)
    return _pearson_corr(ref_sub[triu_idx], cur_sub[triu_idx])


def get_drift_reference_state(model, test_datasets, reference_context, repr_samples=256):
    reference_state = {
        "reference_context": reference_context,
        "param_vector": _get_parameter_vector(model).numpy(),
        "repr_vectors": {},
        "repr_matrices": {},
        "repr_labels": {},
    }
    for context_id, dataset in enumerate(test_datasets):
        matrix, labels = _compute_representation_matrix(
            model, dataset, context_id=context_id, max_samples=repr_samples
        )
        reference_state["repr_matrices"][context_id] = None if matrix is None else matrix.numpy()
        reference_state["repr_labels"][context_id] = None if labels is None else labels.numpy()
        reference_state["repr_vectors"][context_id] = None if matrix is None else matrix.mean(dim=0).numpy()
    return reference_state


def get_drift_state_serializable(model, test_datasets, repr_samples=256):
    state = {
        "param_vector": _get_parameter_vector(model).numpy().tolist(),
        "repr_vectors": [],
        "repr_matrices": [],
        "repr_labels": [],
    }
    for context_id, dataset in enumerate(test_datasets):
        matrix, labels = _compute_representation_matrix(
            model, dataset, context_id=context_id, max_samples=repr_samples
        )
        if matrix is None:
            state["repr_vectors"].append(None)
            state["repr_matrices"].append(None)
            state["repr_labels"].append(None)
        else:
            state["repr_vectors"].append(matrix.mean(dim=0).numpy().tolist())
            state["repr_matrices"].append(matrix.numpy().tolist())
            state["repr_labels"].append(labels.numpy().tolist())
    return state


def compute_drift_metrics(model, test_datasets, reference_state, current_context, repr_samples=256):
    current_param_vector = _get_parameter_vector(model).numpy()
    ref_param_vector = np.asarray(reference_state["param_vector"], dtype=np.float64)
    param_cos_similarity = _cosine_similarity(current_param_vector, ref_param_vector)
    param_matrix = _vector_to_matrix(current_param_vector)
    ref_param_matrix = _vector_to_matrix(ref_param_vector)
    param_cka_similarity = _linear_cka(param_matrix, ref_param_matrix)
    param_procrustes_distance = _orthogonal_procrustes_distance(param_matrix, ref_param_matrix)

    max_context = len(test_datasets) if current_context is None else current_context
    repr_cosines = []
    repr_cka_scores = []
    repr_procrustes_scores = []
    repr_cv_rsa_scores = []
    for context_id in range(max_context):
        ref_matrix = reference_state["repr_matrices"].get(context_id)
        ref_labels = reference_state["repr_labels"].get(context_id)
        if ref_matrix is None:
            continue
        cur_matrix, cur_labels = _compute_representation_matrix(
            model, test_datasets[context_id], context_id=context_id, max_samples=repr_samples
        )
        if cur_matrix is None:
            continue
        ref_matrix_np = np.asarray(ref_matrix, dtype=np.float64)
        cur_matrix_np = cur_matrix.numpy()
        n_common = min(ref_matrix_np.shape[0], cur_matrix_np.shape[0])
        if n_common < 2:
            continue
        ref_mean = ref_matrix_np[:n_common].mean(axis=0)
        cur_mean = cur_matrix_np[:n_common].mean(axis=0)
        repr_cosines.append(_cosine_similarity(cur_mean, ref_mean))
        repr_cka_scores.append(_linear_cka(cur_matrix_np[:n_common], ref_matrix_np[:n_common]))
        repr_procrustes_scores.append(
            _orthogonal_procrustes_distance(cur_matrix_np[:n_common], ref_matrix_np[:n_common])
        )
        if ref_labels is not None and cur_labels is not None:
            repr_cv_rsa_scores.append(
                _cv_rsa_similarity(
                    ref_matrix_np[:n_common],
                    np.asarray(ref_labels)[:n_common],
                    cur_matrix_np[:n_common],
                    cur_labels.numpy()[:n_common],
                )
            )

    representational_cos_similarity = float(np.nanmean(repr_cosines)) if len(repr_cosines) > 0 else float("nan")
    representational_cka_similarity = float(np.nanmean(repr_cka_scores)) if len(repr_cka_scores) > 0 else float("nan")
    representational_procrustes_distance = (
        float(np.nanmean(repr_procrustes_scores)) if len(repr_procrustes_scores) > 0 else float("nan")
    )
    representational_cv_rsa_similarity = (
        float(np.nanmean(repr_cv_rsa_scores)) if len(repr_cv_rsa_scores) > 0 else float("nan")
    )
    return {
        "param_cos_similarity": param_cos_similarity,
        "representational_cos_similarity": representational_cos_similarity,
        "param_cka_similarity": param_cka_similarity,
        "representational_cka_similarity": representational_cka_similarity,
        "param_procrustes_distance": param_procrustes_distance,
        "representational_procrustes_distance": representational_procrustes_distance,
        "representational_cv_rsa_similarity": representational_cv_rsa_similarity,
    }


def append_drift_to_plotting_dict(plotting_dict, drift_metrics, iteration, current_context):
    if (plotting_dict is None) or ("drift" not in plotting_dict):
        return
    for key in [
        "param_cos_similarity",
        "representational_cos_similarity",
        "param_cka_similarity",
        "representational_cka_similarity",
        "param_procrustes_distance",
        "representational_procrustes_distance",
        "representational_cv_rsa_similarity",
    ]:
        if key not in plotting_dict["drift"]:
            plotting_dict["drift"][key] = []
        plotting_dict["drift"][key].append(drift_metrics.get(key, float("nan")))
    plotting_dict["drift"]["x_iteration"].append(iteration)
    plotting_dict["drift"]["x_context"].append(current_context)


def append_drift_snapshot(plotting_dict, current_context, snapshot):
    if (plotting_dict is None) or ("drift" not in plotting_dict):
        return
    key = str(current_context)
    if key not in plotting_dict["drift"]["snapshots"]:
        plotting_dict["drift"]["snapshots"][key] = snapshot


def _mean_abs_weight_magnitude(model):
    eval_model = model
    if hasattr(model, "label") and model.label == "SeparateClassifiers":
        eval_model = getattr(model, "context1")
    total_abs = 0.0
    n_params = 0
    with torch.no_grad():
        for param in eval_model.parameters():
            if param.requires_grad:
                total_abs += param.detach().float().abs().sum().item()
                n_params += param.numel()
    return total_abs / max(n_params, 1)


@torch.no_grad()
def _dead_unit_fraction(model, dataset, context_id=0, max_samples=256, batch_size=128):
    eval_model = _extract_model_for_context(model, context_id)
    device = eval_model.device if hasattr(eval_model, 'device') else eval_model._device()
    cuda = eval_model.cuda if hasattr(eval_model, 'cuda') else eval_model._is_on_cuda()

    mode = eval_model.training
    eval_model.eval()

    if hasattr(eval_model, "mask_dict") and eval_model.mask_dict is not None:
        eval_model.apply_XdGmask(context=context_id + 1)

    activation_batches = []
    collected = 0
    data_loader = get_data_loader(dataset, batch_size=batch_size, cuda=cuda)
    for x, _ in data_loader:
        if collected >= max_samples:
            break
        x = x.to(device)
        h = eval_model.flatten(eval_model.convE(x))
        for lay_id in range(1, eval_model.fcE.layers + 1):
            layer = getattr(eval_model.fcE, "fcLayer{}".format(lay_id))
            h = layer(h)
        activation_batches.append(h.detach().float().cpu())
        collected += h.shape[0]

    eval_model.train(mode=mode)

    if len(activation_batches) == 0:
        return float("nan")
    activations = torch.cat(activation_batches, dim=0)[:max_samples]
    return (activations == 0).all(dim=0).float().mean().item()


@torch.no_grad()
def _effective_rank(model, dataset, context_id=0, max_samples=256, batch_size=128, eps=1e-9):
    eval_model = _extract_model_for_context(model, context_id)
    device = eval_model.device if hasattr(eval_model, 'device') else eval_model._device()
    cuda = eval_model.cuda if hasattr(eval_model, 'cuda') else eval_model._is_on_cuda()

    mode = eval_model.training
    eval_model.eval()

    if hasattr(eval_model, "mask_dict") and eval_model.mask_dict is not None:
        eval_model.apply_XdGmask(context=context_id + 1)

    feature_batches = []
    collected = 0
    data_loader = get_data_loader(dataset, batch_size=batch_size, cuda=cuda)
    for x, _ in data_loader:
        if collected >= max_samples:
            break
        x = x.to(device)
        features = eval_model.feature_extractor(x)
        feature_batches.append(features.detach().float().cpu().view(features.shape[0], -1))
        collected += features.shape[0]

    eval_model.train(mode=mode)

    if len(feature_batches) == 0:
        return float("nan")
    h = torch.cat(feature_batches, dim=0)[:max_samples]
    if h.size(0) < 2:
        return float("nan")
    h = h - h.mean(dim=0, keepdim=True)
    try:
        singular_values = torch.linalg.svdvals(h)
    except RuntimeError:
        singular_values = torch.svd(h)[1]
    singular_values = singular_values[singular_values > eps]
    if singular_values.numel() == 0:
        return float("nan")
    p = (singular_values ** 2) / (singular_values ** 2).sum()
    entropy = -(p * (p + eps).log()).sum()
    return torch.exp(entropy).item()


def compute_lop_metrics(model, dataset, context_id=0, metric_samples=256, batch_size=128):
    return {
        "dead_unit_fraction": _dead_unit_fraction(
            model, dataset, context_id=context_id, max_samples=metric_samples, batch_size=batch_size
        ),
        "effective_rank": _effective_rank(
            model, dataset, context_id=context_id, max_samples=metric_samples, batch_size=batch_size
        ),
        "weight_magnitude": _mean_abs_weight_magnitude(model),
    }


def append_lop_metrics_to_plotting_dict(plotting_dict, lop_metrics, iteration, current_context):
    if (plotting_dict is None) or ("lop" not in plotting_dict):
        return
    plotting_dict["lop"]["dead_unit_fraction"].append(lop_metrics["dead_unit_fraction"])
    plotting_dict["lop"]["effective_rank"].append(lop_metrics["effective_rank"])
    plotting_dict["lop"]["weight_magnitude"].append(lop_metrics["weight_magnitude"])
    plotting_dict["lop"]["x_iteration"].append(iteration)
    plotting_dict["lop"]["x_context"].append(current_context)


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----GENERATION EVALUATION----####
####-----------------------------####

def show_samples(model, config, pdf=None, visdom=None, size=32, pdf_title="Generated images", visdom_title="Samples"):
    '''Plot samples from a generative model in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    mode = model.training
    model.eval()

    # Generate samples from the model
    sample = model.sample(size)
    image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)

    # Plot generated images in [pdf] and/or [visdom]
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size)))
    # -make plots
    if pdf is not None:
        visual_plt.plot_images_from_tensor(image_tensor, pdf, title=pdf_title, nrow=nrow)
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, title='{} ({})'.format(visdom_title, visdom["graph"]), env=visdom["env"], nrow=nrow,
        )

    # Set model back to initial mode
    model.train(mode=mode)


####--------------------------------------------------------------------------------------------------------------####

####---------------------------------####
####----RECONSTRUCTION EVALUATION----####
####---------------------------------####

def show_reconstruction(model, dataset, config, pdf=None, visdom=None, size=32, context=None):
    '''Plot reconstructed examples by an auto-encoder [model] on [dataset], in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    mode = model.training
    model.eval()

    # Get data
    data_loader = get_data_loader(dataset, size, cuda=model._is_on_cuda())
    (data, labels) = next(iter(data_loader))
    data, labels = data.to(model._device()), labels.to(model._device())

    # Evaluate model
    with torch.no_grad():
        recon_batch = model(data, full=False)

    # Plot original and reconstructed images
    comparison = torch.cat(
        [data.view(-1, config['channels'], config['size'], config['size'])[:size],
         recon_batch.view(-1, config['channels'], config['size'], config['size'])[:size]]
    ).cpu()
    image_tensor = comparison.view(-1, config['channels'], config['size'], config['size'])
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size*2)))
    # -make plots
    if pdf is not None:
        context_stm = "" if context is None else " (context {})".format(context)
        visual_plt.plot_images_from_tensor(
            image_tensor, pdf, nrow=nrow, title="Reconstructions" + context_stm
        )
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, title='Reconstructions ({})'.format(visdom["graph"]), env=visdom["env"], nrow=nrow,
        )

    # Set model back to initial mode
    model.train(mode=mode)