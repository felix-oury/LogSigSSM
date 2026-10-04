"""
This module defines functions for creating datasets, building models, and training them using JAX
and Equinox. The main function, `create_dataset_model_and_train`, is designed to initialise the
dataset, construct the model, and execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:

- `seed`: A random seed for reproducibility.
- `data_dir`: The directory where the dataset is stored.
- `use_presplit`: A boolean indicating whether to use a pre-split dataset.
- `dataset_name`: The name of the dataset to load and use for training.
- `output_step`: For regression tasks, the number of steps to skip before outputting a prediction.
- `metric`: The metric to use for evaluation. Supported values are `'mse'` for regression and `'accuracy'` for
            classification.
- `include_time`: A boolean indicating whether to include time as a channel in the time series data.
- `T`: The maximum time value to scale time data to [0, T].
- `model_name`: The name of the model architecture to use.
- `stepsize`: The size of the intervals for the Log-ODE method.
- `logsig_depth`: The depth of the Log-ODE method. Currently implemented for depths 1 and 2.
- `model_args`: A dictionary of additional arguments to customise the model.
- `num_steps`: The number of steps to train the model.
- `print_steps`: How often to print the loss during training.
- `lr`: The learning rate for the optimiser.
- `lr_scheduler`: The learning rate scheduler function.
- `batch_size`: The number of samples per batch during training.
- `output_parent_dir`: The parent directory where the training outputs will be saved.

The module also includes the following key functions:

- `calc_output`: Computes the model output, handling stateful and nondeterministic models with JAX's `vmap` for
                 batching.
- `classification_loss`: Computes the loss for classification tasks, including optional regularisation.
- `regression_loss`: Computes the loss for regression tasks, including optional regularisation.
- `make_step`: Performs a single optimisation step, updating model parameters based on the computed gradients.
- `train_model`: Handles the training loop, managing metrics, early stopping, and saving progress at regular intervals.
"""

import os
import shutil
import time
import json
import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from data_dir.datasets import create_dataset
from models.generate_model import create_model


@eqx.filter_jit
def calc_output(model, X, state, key, stateful, nondeterministic):
    if stateful:
        if nondeterministic:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
            )(X, state, key)
        else:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
            )(X, state)
    elif nondeterministic:
        output = jax.vmap(model, in_axes=(0, None))(X, key)
    else:
        output = jax.vmap(model)(X)

    return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(diff_model, static_model, X, y, state, key):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
    )
    norm = 0
    if model.lip2:
        if hasattr(model, "vf"):
            for layer in model.vf.mlp.layers:
                norm += jnp.mean(
                    jnp.linalg.norm(layer.weight, axis=-1)
                    + jnp.linalg.norm(layer.bias, axis=-1)
                )
        else:
            norm += jnp.mean(jnp.linalg.norm(model.vf_A, axis=-1))
        norm *= model.lambd
    return (
        jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)) + norm,
        state,
    )


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(diff_model, static_model, X, y, state, key):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
    )
    pred_y = pred_y[:, :, 0]
    norm = 0
    if model.lip2:
        if hasattr(model, "vf"):
            for layer in model.vf.mlp.layers:
                norm += jnp.mean(
                    jnp.linalg.norm(layer.weight, axis=-1)
                    + jnp.linalg.norm(layer.bias, axis=-1)
                )
        else:
            norm += jnp.mean(jnp.linalg.norm(model.vf_A, axis=-1))
    return (
        jnp.mean(jnp.mean((pred_y - y) ** 2, axis=1)) + norm,
        state,
    )


@eqx.filter_jit
def make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key):
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value


def train_model(
    model_name,
    dataset_name,
    model,
    metric,
    filter_spec,
    state,
    dataloaders,
    num_steps,
    print_steps,
    early_stopping_steps,
    lr,
    lr_scheduler,
    batch_size,
    key,
    output_dir,
):

    if metric == "accuracy":
        best_val = max
        operator_improv = lambda x, y: x >= y
        operator_no_improv = lambda x, y: x <= y
    elif metric == "mse":
        best_val = min
        operator_improv = lambda x, y: x <= y
        operator_no_improv = lambda x, y: x >= y
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Directory {output_dir} has been created.")

    batchkey, key = jr.split(key, 2)
    opt = optax.adam(learning_rate=lr_scheduler(lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    if model.classification:
        loss_fn = classification_loss
    else:
        loss_fn = regression_loss

    running_loss = 0.0
    if metric == "accuracy":
        all_val_metric = [0.0]
        all_train_metric = [0.0]
        val_metric_for_best_model = [0.0]
    elif metric == "mse":
        all_val_metric = [100.0]
        all_train_metric = [100.0]
        val_metric_for_best_model = [100.0]
    no_val_improvement = 0
    all_time = []
    start = time.time()
    for step, data in zip(
        range(num_steps),
        dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        stepkey, key = jr.split(key, 2)
        X, y = data

        if (
            model_name == "bd_linear_ncde"
            or model_name == "diagonal_linear_ncde"
            or model_name == "dense_linear_ncde"
        ) and dataset_name == "Heartbeat":
            X = (X[0], X[1] / 10, X[2])
        model, state, opt_state, value = make_step(
            model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey
        )
        running_loss += value
        if (step + 1) % print_steps == 0:
            predictions = []
            labels = []
            for data in dataloaders["train"].loop_epoch(batch_size):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                if (
                    model_name == "bd_linear_ncde"
                    or model_name == "diagonal_linear_ncde"
                    or model_name == "dense_linear_ncde"
                ) and dataset_name == "Heartbeat":
                    X = (X[0], X[1] / 10, X[2])
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                )
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            if model.classification:
                train_metric = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
            else:
                prediction = prediction[:, :, 0]
                train_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
            predictions = []
            labels = []
            for data in dataloaders["val"].loop_epoch(batch_size):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                if (
                    model_name == "bd_linear_ncde"
                    or model_name == "diagonal_linear_ncde"
                    or model_name == "dense_linear_ncde"
                ) and dataset_name == "Heartbeat":
                    X = (X[0], X[1] / 10, X[2])
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                )
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            if model.classification:
                val_metric = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
            else:
                prediction = prediction[:, :, 0]
                val_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
            end = time.time()
            total_time = end - start
            print(
                f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
                f"Train metric: {train_metric}, "
                f"Validation metric: {val_metric}, Time: {total_time}"
            )
            start = time.time()
            if step > 0:
                if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
                    no_val_improvement += 1
                    if no_val_improvement > early_stopping_steps:
                        break
                else:
                    no_val_improvement = 0
                if operator_improv(val_metric, best_val(val_metric_for_best_model)):
                    val_metric_for_best_model.append(val_metric)
                    predictions = []
                    labels = []
                    for data in dataloaders["test"].loop_epoch(batch_size):
                        stepkey, key = jr.split(key, 2)
                        inference_model = eqx.tree_inference(model, value=True)
                        X, y = data
                        if (
                            model_name == "bd_linear_ncde"
                            or model_name == "diagonal_linear_ncde"
                            or model_name == "dense_linear_ncde"
                        ) and dataset_name == "Heartbeat":
                            X = (X[0], X[1] / 10, X[2])
                        prediction, _ = calc_output(
                            inference_model,
                            X,
                            state,
                            stepkey,
                            model.stateful,
                            model.nondeterministic,
                        )
                        predictions.append(prediction)
                        labels.append(y)
                    prediction = jnp.vstack(predictions)
                    y = jnp.vstack(labels)
                    if model.classification:
                        test_metric = jnp.mean(
                            jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                        )
                    else:
                        prediction = prediction[:, :, 0]
                        test_metric = jnp.mean(
                            jnp.mean((prediction - y) ** 2, axis=1), axis=0
                        )
                    print(f"Test metric: {test_metric}")
                running_loss = 0.0
                all_train_metric.append(train_metric)
                all_val_metric.append(val_metric)
                all_time.append(total_time)
                steps = jnp.arange(0, step + 1, print_steps)
                all_train_metric_save = jnp.array(all_train_metric)
                all_val_metric_save = jnp.array(all_val_metric)
                all_time_save = jnp.array(all_time)
                test_metric_save = jnp.array(test_metric)
                jnp.save(output_dir + "/steps.npy", steps)
                jnp.save(output_dir + "/all_train_metric.npy", all_train_metric_save)
                jnp.save(output_dir + "/all_val_metric.npy", all_val_metric_save)
                jnp.save(output_dir + "/all_time.npy", all_time_save)
                jnp.save(output_dir + "/test_metric.npy", test_metric_save)

    print(f"Test metric: {test_metric}")
    steps = jnp.arange(0, num_steps + 1, print_steps)
    all_train_metric = jnp.array(all_train_metric)
    all_val_metric = jnp.array(all_val_metric)
    all_time = jnp.array(all_time)
    test_metric = jnp.array(test_metric)
    jnp.save(output_dir + "/steps.npy", steps)
    jnp.save(output_dir + "/all_train_metric.npy", all_train_metric)
    jnp.save(output_dir + "/all_val_metric.npy", all_val_metric)
    jnp.save(output_dir + "/all_time.npy", all_time)
    jnp.save(output_dir + "/test_metric.npy", test_metric)

    return model


def create_dataset_model_and_train(
    seed,
    data_dir,
    use_presplit,
    dataset_name,
    output_step,
    metric,
    include_time,
    T,
    model_name,
    stepsize,
    logsig_depth,
    model_args,
    num_steps,
    print_steps,
    early_stopping_steps,
    lr,
    lr_scheduler,
    batch_size,
    output_parent_dir="",
):
    # New runs go to reruns/ so that the recorded runs in outputs/ are never overwritten
    output_parent_dir += "reruns/" + model_name + "/" + dataset_name

    # Build a compact, deterministic experiment name with a short hash of config
    prefix = f"T_{T:.2f}_t_{int(bool(include_time))}_n_{num_steps}_lr_{lr}"
    if model_name == "log_ncde" or model_name == "nrde":
        prefix += f"_s_{stepsize}_d_{logsig_depth}"

    # Normalise model_args for hashing (avoid non-deterministic object reprs)
    normalised_model_args = {}
    for k, v in model_args.items():
        if hasattr(v, "__class__") and not isinstance(v, (int, float, str, bool)) and v is not None:
            normalised_model_args[k] = v.__class__.__name__
        elif isinstance(v, float):
            normalised_model_args[k] = round(v, 8)
        else:
            normalised_model_args[k] = v

    scheduler_name = getattr(lr_scheduler, "__name__", lr_scheduler.__class__.__name__ if hasattr(lr_scheduler, "__class__") else str(lr_scheduler))

    hash_payload = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seed": seed,
        "T": round(T, 8),
        "include_time": bool(include_time),
        "num_steps": num_steps,
        "lr": lr,
        "stepsize": stepsize,
        "logsig_depth": logsig_depth,
        "batch_size": batch_size,
        "lr_scheduler": scheduler_name,
        "model_args": normalised_model_args,
    }
    hash_str = hashlib.blake2b(
        json.dumps(hash_payload, sort_keys=True).encode("utf-8"), digest_size=6
    ).hexdigest()

    output_dir = f"{prefix}_seed_{seed}_h_{hash_str}"

    key = jr.PRNGKey(seed)

    datasetkey, modelkey, trainkey, key = jr.split(key, 4)
    print(f"Creating dataset {dataset_name}")

    if (
        model_name == "bd_linear_ncde"
        or model_name == "diagonal_linear_ncde"
        or model_name == "dense_linear_ncde"
    ):
        scale = True
    else:
        scale = False

    dataset = create_dataset(
        data_dir,
        dataset_name,
        stepsize=stepsize,
        depth=logsig_depth,
        include_time=include_time,
        T=T,
        use_idxs=False,
        use_presplit=use_presplit,
        scale=scale,
        key=datasetkey,
    )

    print(f"Creating model {model_name}")
    classification = metric == "accuracy"
    model, state = create_model(
        model_name,
        dataset.data_dim,
        dataset.logsig_dim,
        logsig_depth,
        dataset.intervals,
        dataset.label_dim,
        classification=classification,
        output_step=output_step,
        **model_args,
        key=modelkey,
    )
    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    if (
        model_name == "nrde"
        or model_name == "log_ncde"
        or model_name == "bd_linear_ncde"
        or model_name == "diagonal_linear_ncde"
        or model_name == "dense_linear_ncde"
    ):
        dataloaders = dataset.path_dataloaders
        if model_name == "log_ncde":
            where = lambda model: (model.intervals, model.pairs)
            filter_spec = eqx.tree_at(
                where, filter_spec, replace=(False, False), is_leaf=lambda x: x is None
            )
        elif model_name == "nrde":
            where = lambda model: (model.intervals,)
            filter_spec = eqx.tree_at(where, filter_spec, replace=(False,))
    elif model_name == "ncde":
        dataloaders = dataset.coeff_dataloaders
    else:
        dataloaders = dataset.raw_dataloaders

    return train_model(
        model_name,
        dataset_name,
        model,
        metric,
        filter_spec,
        state,
        dataloaders,
        num_steps,
        print_steps,
        early_stopping_steps,
        lr,
        lr_scheduler,
        batch_size,
        trainkey,
        output_parent_dir + "/" + output_dir,
    )


def create_sig_s5_model_and_train(
    seed,
    data_dir,
    use_presplit,
    dataset_name,
    output_step,
    metric,
    include_time,
    T,
    model_name,
    stepsize,
    logsig_depth,
    model_args,
    num_steps,
    print_steps,
    early_stopping_steps,
    lr,
    lr_scheduler,
    batch_size,
    output_parent_dir="",
):
    """
    Train sig_s5 model with signature features using the official data loading pipeline.
    This ensures fair benchmarking by using the same JAX RNG-based data splits as other JAX models.
    """
    import numpy as np
    import sys
    import pickle

    key = jr.PRNGKey(seed)
    datasetkey, trainkey = jr.split(key)

    # Special handling for PPG to avoid OOM during coefficient computation
    if dataset_name == "ppg":
        # Load PPG data directly (pre-split)
        ppg_dir = os.path.join(data_dir, "processed", "PPG", "ppg")
        with open(os.path.join(ppg_dir, "X_train.pkl"), "rb") as f:
            Xtr_ntc = np.array(pickle.load(f)).astype(np.float32)
        with open(os.path.join(ppg_dir, "y_train.pkl"), "rb") as f:
            ytr = np.array(pickle.load(f)).astype(np.float32)
        with open(os.path.join(ppg_dir, "X_val.pkl"), "rb") as f:
            Xva_ntc = np.array(pickle.load(f)).astype(np.float32)
        with open(os.path.join(ppg_dir, "y_val.pkl"), "rb") as f:
            yva = np.array(pickle.load(f)).astype(np.float32)
        with open(os.path.join(ppg_dir, "X_test.pkl"), "rb") as f:
            Xte_ntc = np.array(pickle.load(f)).astype(np.float32)
        with open(os.path.join(ppg_dir, "y_test.pkl"), "rb") as f:
            yte = np.array(pickle.load(f)).astype(np.float32)
    else:
        # Use official dataset creation to get JAX RNG-based splits for UEA datasets
        dataset = create_dataset(
            data_dir,
            dataset_name,
            stepsize=stepsize,
            depth=logsig_depth,
            include_time=False,  # We'll add time after feature extraction if needed
            T=T,
            use_idxs=False,
            use_presplit=use_presplit,
            scale=False,
            key=datasetkey,
        )

        # Extract raw data from dataloaders (these are already split using JAX RNG)
        train_loader = dataset.raw_dataloaders["train"]
        val_loader = dataset.raw_dataloaders["val"]
        test_loader = dataset.raw_dataloaders["test"]

        # Convert to numpy arrays
        Xtr_ntc = np.array(train_loader.data)
        Xva_ntc = np.array(val_loader.data)
        Xte_ntc = np.array(test_loader.data)
        ytr = np.array(train_loader.labels)
        yva = np.array(val_loader.labels)
        yte = np.array(test_loader.labels)

    # Feature extraction configuration
    signature_cfg = model_args.get("signature", {})
    normalize = model_args.get("normalize", False)

    # Determine model variant for output directory
    model_name_out = "sig_s5"

    # Extract signature stride for target downsampling (needed for regression)
    feature_stride = int(signature_cfg.get("stride", 1)) if signature_cfg else 1

    # Feature computation helper - log signature features only
    def compute_features(x_ntc):
        x_nct = np.transpose(x_ntc, (0, 2, 1)).astype(np.float32)

        # Log signature features
        repo_root = os.path.dirname(os.path.abspath(__file__))
        sig_dir = os.path.join(repo_root, "running_mamba")
        if sig_dir not in sys.path:
            sys.path.append(sig_dir)
        try:
            from log_signature_fast import batch_compute_log_features
        except Exception as e:
            raise ImportError(f"Failed to import log signature computation from log_signature_fast: {e}")

        lengths = [int(x) for x in signature_cfg.get("lengths", [])]
        depths = [int(x) for x in signature_cfg.get("depths", [])]
        stride = int(signature_cfg.get("stride", 1))
        include_global = bool(signature_cfg.get("include_global", False))
        d_global = int(signature_cfg.get("d_global", 0)) if include_global else None

        feats, _, _ = batch_compute_log_features(x_nct, lengths, depths, stride, include_global, d_global)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return feats

    # Compute features for all splits
    Xtr = compute_features(Xtr_ntc)
    Xva = compute_features(Xva_ntc)
    Xte = compute_features(Xte_ntc)

    # Ensure all features are finite before normalization
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xva = np.nan_to_num(Xva, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Normalize if requested
    if normalize:
        mu = Xtr.mean(axis=(0, 1), keepdims=True)
        std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / std
        Xva = (Xva - mu) / std
        Xte = (Xte - mu) / std

        # Ensure normalized features are still finite
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        Xva = np.nan_to_num(Xva, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Add time channel if requested
    if include_time:
        N, Tp, D = Xtr.shape
        t = np.linspace(0.0, 1.0, Tp, dtype=np.float32)[None, :, None]
        Xtr = np.concatenate([Xtr.astype(np.float32), np.repeat(t, N, axis=0)], axis=2)

        N, Tp, D = Xva.shape
        t = np.linspace(0.0, 1.0, Tp, dtype=np.float32)[None, :, None]
        Xva = np.concatenate([Xva.astype(np.float32), np.repeat(t, N, axis=0)], axis=2)

        N, Tp, D = Xte.shape
        t = np.linspace(0.0, 1.0, Tp, dtype=np.float32)[None, :, None]
        Xte = np.concatenate([Xte.astype(np.float32), np.repeat(t, N, axis=0)], axis=2)

    # Determine if classification or regression
    classification = metric == "accuracy"

    # Convert labels to appropriate format
    if classification:
        # For classification, convert one-hot to integer labels
        if ytr.ndim > 1:
            ytr = np.argmax(ytr, axis=-1).astype(np.int64)
            yva = np.argmax(yva, axis=-1).astype(np.int64)
            yte = np.argmax(yte, axis=-1).astype(np.int64)
        num_classes = len(np.unique(ytr))
    else:
        # For regression, keep as-is
        num_classes = 1

    # Import S5 builder
    repo_root = os.path.dirname(os.path.abspath(__file__))
    s5_dir = os.path.join(repo_root, "running_s5")
    if s5_dir not in sys.path:
        sys.path.append(s5_dir)

    if classification:
        from s5_cv_pipeline import build_s5_classifier
        model_builder = build_s5_classifier
    else:
        from s5_cv_pipeline import build_s5_regressor
        model_builder = build_s5_regressor

    # Build model
    ssm_blocks_val = model_args.get("ssm_blocks", 1)
    if ssm_blocks_val is None:
        ssm_blocks_val = 1

    if classification:
        model = model_builder(
            input_dim=int(Xtr.shape[2]),
            num_classes=num_classes,
            hidden_dim=int(model_args.get("hidden_dim", 128)),
            ssm_size=int(model_args.get("ssm_dim", 64)),
            num_blocks=int(model_args.get("num_blocks", 4)),
            ssm_blocks=int(ssm_blocks_val),
            key=trainkey,
        )
    else:
        # For sig_s5 regression with signatures: adjust output_step for transformed space
        # Original: 49920 points, output every 128 → 390 predictions
        # Transformed: 12480 points (stride=4), output every 128/4=32 → 390 predictions
        output_step_transformed = output_step // feature_stride
        model = model_builder(
            input_dim=int(Xtr.shape[2]),
            hidden_dim=int(model_args.get("hidden_dim", 128)),
            ssm_size=int(model_args.get("ssm_dim", 64)),
            num_blocks=int(model_args.get("num_blocks", 4)),
            ssm_blocks=int(ssm_blocks_val),
            output_step=output_step_transformed,  # Output in transformed space
            key=trainkey,
        )

    state = eqx.nn.State(model)
    params, static_model = eqx.partition(model, eqx.is_inexact_array)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Training step
    @eqx.filter_jit
    def step_fn(params, state, opt_state, xb, yb, key):
        def loss_fn(p):
            m = eqx.combine(p, static_model)
            keys = jr.split(key, xb.shape[0])

            if classification:
                probs = jax.vmap(lambda xi, kk: m(xi, state, key=kk)[0])(xb, keys)
                probs = jnp.clip(probs, 1e-7, 1.0)
                onehot = jax.nn.one_hot(yb, num_classes)
                loss = -jnp.sum(onehot * jnp.log(probs), axis=-1)
            else:
                preds = jax.vmap(lambda xi, kk: m(xi, state, key=kk)[0])(xb, keys)
                loss = (preds[:, :, 0] - yb) ** 2

            return jnp.mean(loss)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, state, opt_state, loss

    # Setup output directory with model-info naming (aligned with sig_mamba)
    # New runs go to reruns/ so that the recorded runs in outputs/ are never overwritten
    output_parent_dir += "reruns/" + model_name_out + "/" + dataset_name + "/"

    # Build directory name with model architecture info (like sig_mamba)
    full_output_dir = output_parent_dir + f"lr_{lr}_time_{include_time}"
    for k, v in model_args.items():
        if k == "signature":
            # Skip feature config dicts (too verbose for directory name)
            continue
        full_output_dir += f"_{k}_{v}"
    full_output_dir += f"_seed_{seed}"

    os.makedirs(full_output_dir, exist_ok=True)

    # Get scheduler name for config
    scheduler_name = getattr(lr_scheduler, "__name__", lr_scheduler.__class__.__name__ if hasattr(lr_scheduler, "__class__") else str(lr_scheduler))

    # Save configuration for reproducibility
    config_dict = {
        "model_name": model_name_out,
        "dataset_name": dataset_name,
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "print_steps": print_steps,
        "early_stopping_steps": early_stopping_steps,
        "T": T,
        "include_time": include_time,
        "stepsize": stepsize,
        "logsig_depth": logsig_depth,
        "metric": metric,
        "classification": classification,
        "output_step": output_step,
        "normalize": normalize,
        "model_args": {
            "hidden_dim": int(model_args.get("hidden_dim", 128)),
            "ssm_dim": int(model_args.get("ssm_dim", 64)),
            "num_blocks": int(model_args.get("num_blocks", 4)),
            "ssm_blocks": int(model_args.get("ssm_blocks", 1)),
        },
        "signature_config": model_args.get("signature", {}),
        "lr_scheduler": scheduler_name,
    }
    with open(os.path.join(full_output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Training loop
    import time
    step = 0
    best_val = 1e9 if not classification else 0.0
    no_improvement = 0
    all_train_metrics = []
    all_val_metrics = []
    all_time = []
    steps_arr = []

    B = batch_size
    N = Xtr.shape[0]
    rng_np = np.random.default_rng(seed)

    start_time = time.time()
    should_stop = False

    while step < num_steps and not should_stop:
        # Shuffle for each epoch
        order = rng_np.permutation(N)

        for s in range(0, N, B):
            if step >= num_steps or should_stop:
                break

            idx = order[s:s+B]
            if len(idx) == 0:
                continue

            xb = jnp.array(Xtr[idx])
            yb = jnp.array(ytr[idx])

            subkey, trainkey = jr.split(trainkey)
            params, state, opt_state, loss = step_fn(params, state, opt_state, xb, yb, subkey)

            step += 1

            if step % print_steps == 0:
                # Evaluate
                m = eqx.combine(params, static_model)

                def predict(X):
                    b = X.shape[0]
                    keys = jr.split(jr.PRNGKey(0), b)
                    if classification:
                        probs = jax.vmap(lambda xi, kk: m(xi, state, key=kk)[0])(jnp.array(X), keys)
                        return np.array(jnp.argmax(probs, axis=-1))
                    else:
                        preds = jax.vmap(lambda xi, kk: m(xi, state, key=kk)[0])(jnp.array(X), keys)
                        return np.array(preds[:, :, 0])

                if classification:
                    train_metric = np.mean(predict(Xtr) == ytr)
                    val_metric = np.mean(predict(Xva) == yva)
                else:
                    pred_tr = predict(Xtr)
                    pred_va = predict(Xva)
                    train_metric = float(np.mean((pred_tr - ytr) ** 2))
                    val_metric = float(np.mean((pred_va - yva) ** 2))

                all_train_metrics.append(train_metric)
                all_val_metrics.append(val_metric)
                steps_arr.append(step)

                elapsed = time.time() - start_time
                all_time.append(elapsed)
                print(f"Step: {step}, Train: {train_metric:.6f}, Val: {val_metric:.6f}, Time: {elapsed:.2f}s")

                # Check for no improvement (for early stopping)
                if classification:
                    no_improv = val_metric <= best_val
                else:
                    no_improv = val_metric >= best_val

                if no_improv:
                    no_improvement += 1
                    if no_improvement > early_stopping_steps:
                        print(f"Early stopping at step {step}")
                        should_stop = True
                        break
                else:
                    no_improvement = 0

                # Check for improvement (for updating best model and test metric) - includes equality
                if classification:
                    is_improved = val_metric >= best_val
                else:
                    is_improved = val_metric <= best_val

                if is_improved:
                    best_val = val_metric
                    # Compute test metric
                    if classification:
                        best_test = np.mean(predict(Xte) == yte)
                    else:
                        pred_te = predict(Xte)
                        best_test = float(np.mean((pred_te - yte) ** 2))
                    print(f"Test Metric: {best_test:.6f}")

                start_time = time.time()

    # Save results
    np.save(os.path.join(full_output_dir, "steps.npy"), np.array(steps_arr, dtype=np.int32))
    np.save(os.path.join(full_output_dir, "all_train_metric.npy"), np.array(all_train_metrics, dtype=np.float32))
    np.save(os.path.join(full_output_dir, "all_val_metric.npy"), np.array(all_val_metrics, dtype=np.float32))
    np.save(os.path.join(full_output_dir, "all_time.npy"), np.array(all_time, dtype=np.float32))
    np.save(os.path.join(full_output_dir, "test_metric.npy"), np.float32(best_test))

    print(f"Training completed. Best val: {best_val:.6f}, Test: {best_test:.6f}")
