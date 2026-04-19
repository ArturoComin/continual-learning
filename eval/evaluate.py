import numpy as np
import torch
import torch.nn.functional as F
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
    plotting_dict["drift"] = {
        "param_cos_similarity": [],
        "representational_cos_similarity": [],
        "x_iteration": [],
        "x_context": [],
        "snapshots": {},
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


def _compute_representation_vector(model, dataset, context_id, max_samples=256, batch_size=128):
    eval_model = _extract_model_for_context(model, context_id)
    device = eval_model.device if hasattr(eval_model, 'device') else eval_model._device()
    cuda = eval_model.cuda if hasattr(eval_model, 'cuda') else eval_model._is_on_cuda()

    mode = eval_model.training
    eval_model.eval()

    if hasattr(eval_model, "mask_dict") and eval_model.mask_dict is not None:
        eval_model.apply_XdGmask(context=context_id + 1)

    feature_batches = []
    n_collected = 0
    data_loader = get_data_loader(dataset, batch_size=batch_size, cuda=cuda)
    for x, _ in data_loader:
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
        n_collected += features.shape[0]

    eval_model.train(mode=mode)

    if len(feature_batches) == 0:
        return None
    all_features = torch.cat(feature_batches, dim=0)[:max_samples]
    return all_features.mean(dim=0)


def get_drift_reference_state(model, test_datasets, reference_context, repr_samples=256):
    reference_state = {
        "reference_context": reference_context,
        "param_vector": _get_parameter_vector(model),
        "repr_vectors": {},
    }
    for context_id, dataset in enumerate(test_datasets):
        reference_state["repr_vectors"][context_id] = _compute_representation_vector(
            model, dataset, context_id=context_id, max_samples=repr_samples
        )
    return reference_state


def get_drift_state_serializable(model, test_datasets, repr_samples=256):
    state = {
        "param_vector": _get_parameter_vector(model).numpy().tolist(),
        "repr_vectors": [],
    }
    for context_id, dataset in enumerate(test_datasets):
        vector = _compute_representation_vector(model, dataset, context_id=context_id, max_samples=repr_samples)
        state["repr_vectors"].append(None if vector is None else vector.numpy().tolist())
    return state


def compute_drift_metrics(model, test_datasets, reference_state, current_context, repr_samples=256):
    current_param_vector = _get_parameter_vector(model)
    param_cos_similarity = F.cosine_similarity(
        current_param_vector.unsqueeze(0), reference_state["param_vector"].unsqueeze(0)
    ).item()

    max_context = len(test_datasets) if current_context is None else current_context
    repr_cosines = []
    for context_id in range(max_context):
        ref_vector = reference_state["repr_vectors"].get(context_id)
        if ref_vector is None:
            continue
        cur_vector = _compute_representation_vector(
            model, test_datasets[context_id], context_id=context_id, max_samples=repr_samples
        )
        if cur_vector is None:
            continue
        repr_cosines.append(F.cosine_similarity(cur_vector.unsqueeze(0), ref_vector.unsqueeze(0)).item())

    representational_cos_similarity = float(np.mean(repr_cosines)) if len(repr_cosines) > 0 else 0.0
    return {
        "param_cos_similarity": param_cos_similarity,
        "representational_cos_similarity": representational_cos_similarity,
    }


def append_drift_to_plotting_dict(plotting_dict, drift_metrics, iteration, current_context):
    if (plotting_dict is None) or ("drift" not in plotting_dict):
        return
    plotting_dict["drift"]["param_cos_similarity"].append(drift_metrics["param_cos_similarity"])
    plotting_dict["drift"]["representational_cos_similarity"].append(drift_metrics["representational_cos_similarity"])
    plotting_dict["drift"]["x_iteration"].append(iteration)
    plotting_dict["drift"]["x_context"].append(current_context)


def append_drift_snapshot(plotting_dict, current_context, snapshot):
    if (plotting_dict is None) or ("drift" not in plotting_dict):
        return
    key = str(current_context)
    if key not in plotting_dict["drift"]["snapshots"]:
        plotting_dict["drift"]["snapshots"][key] = snapshot


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