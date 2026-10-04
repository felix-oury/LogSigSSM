"""
This module defines classes and functions for creating and training sig_mamba using PyTorch.
The main function, `create_dataset_model_and_train`, is designed to initialise the dataset, construct the model, and
execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:

- `seed`: An integer representing the random seed for reproducibility.
- `data_dir`: The directory where the dataset is stored.
- `output_parent_dir`: The parent directory where the training outputs will be saved.
- `model_name`: A string specifying the model architecture to use ('sig_mamba').
- `metric`: The evaluation metric to use during training, either 'accuracy' for classification or 'mse' for regression.
- `batch_size`: The number of samples per batch during training.
- `dataset_name`: The name of the dataset to load and use for training.
- `n_samples`: The total number of samples in the dataset.
- `output_step`: For regression tasks, defines the interval for outputting predictions.
- `use_presplit`: A boolean indicating whether to use a pre-split dataset.
- `include_time`: A boolean that determines whether to include time as a feature in the dataset. sig_mamba
                  ignores it: its tokens are computed on the raw channels.
- `num_steps`: The total number of steps for training the model.
- `print_steps`: The interval of steps after which to print training progress and metrics.
- `lr`: The learning rate for the optimiser.
- `model_args`: A dictionary containing additional arguments and hyperparameters for model customisation.

The model trained here is `MambaClassification` from `running_mamba/mamba_classification.py`, applied to the
log-signature tokens from `running_mamba/log_signature_fast.py`. The `GLU`, `MambaBlock` and `Mamba` classes defined
below are not used by `create_dataset_model_and_train`.
"""

import os
import shutil
import time
import warnings
import json

import numpy as np
import torch

# Suppress JAX CUDA warning (JAX may be imported by dependencies but isn't used in PyTorch code)
warnings.filterwarnings('ignore', message='.*CUDA-enabled jaxlib is not installed.*')

# CRITICAL: Require real mamba_ssm for sig_mamba - no fallbacks
# This ensures we're using the actual CUDA kernels, not Python fallbacks
try:
    from mamba_ssm import Mamba as MambaLayer  # type: ignore
except Exception as e:
    raise ImportError(f"mamba_ssm is required for sig_mamba. Install it with: pip install mamba-ssm. Error: {e}")

from torch_experiments.jax_dataset import Dataset
import sys
import os


class GLU(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        out = self.linear(x)
        return out[:, :, : x.shape[2]] * torch.sigmoid(out[:, :, x.shape[2] :])


class MambaBlock(torch.nn.Module):
    def __init__(self, hidden_dim, state_dim, conv_dim, expansion):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.mamba = MambaLayer(
            d_model=hidden_dim, d_state=state_dim, d_conv=conv_dim, expand=expansion
        )
        self.glu = GLU(hidden_dim)
        self.activation = torch.nn.GELU()
        # Dropout set to 0.3 for sig_mamba
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(self.activation(x))
        x = self.glu(x)
        x = self.dropout(x)
        x = x + skip
        return x


class Mamba(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        input_dim,
        output_dim,
        hidden_dim,
        state_dim,
        conv_dim,
        expansion,
        classification,
        output_step=1,
    ):
        super().__init__()
        self.linear_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.Sequential(
            *[
                MambaBlock(hidden_dim, state_dim, conv_dim, expansion)
                for _ in range(num_blocks)
            ]
        )
        self.linear_decoder = torch.nn.Linear(hidden_dim, output_dim)
        self.classification = classification
        self.output_step = output_step

    def forward(self, x):
        x = self.linear_encoder(x)
        x = self.blocks(x)
        if self.classification:
            x = torch.mean(x, dim=1)
            x = torch.softmax(self.linear_decoder(x), dim=1)
        else:
            x = x[:, self.output_step - 1 :: self.output_step]
            x = torch.tanh(self.linear_decoder(x))
        return x


def create_dataset_model_and_train(
    seed,
    data_dir,
    output_parent_dir,
    model_name,
    metric,
    batch_size,
    dataset_name,
    n_samples,
    output_step,
    use_presplit,
    include_time,
    num_steps,
    print_steps,
    early_stopping_steps,
    lr,
    model_args,
    output_dir_override=None,
):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Enable deterministic behavior for CuDNN and PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Try to enable deterministic algorithms with warn_only mode
    # This makes operations deterministic where possible, but warns instead of erroring
    # when a deterministic implementation is not available (e.g., some CuBLAS operations)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions don't support warn_only parameter
        # In this case, skip strict determinism to avoid CUBLAS errors
        # The seeded generator and cudnn settings still provide good reproducibility
        pass
    
    # Create a generator for deterministic data loading
    generator = torch.Generator()
    generator.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handle model name with optional _catch/_wst suffix for catch24/WST features
    base_model_name = model_name.replace("_catch", "").replace("_wst", "")

    if base_model_name != "sig_mamba":
        raise ValueError(f"Only sig_mamba is supported, got: {model_name}")

    classification = metric == "accuracy"

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

    if output_dir_override is not None:
        # Explicit output directory (the default name below encodes the model
        # hyperparameters but not the tokeniser).
        output_dir = os.path.abspath(output_dir_override)
    else:
        # Use model_name directly (already includes _catch suffix if needed from run_experiment.py)
        # New runs go to reruns/ so that the recorded runs in outputs/ are never overwritten
        output_dir = output_parent_dir + f"reruns/{model_name}" + f"/{dataset_name}/"
        # For sig_mamba, signatures are a nested dict; avoid dumping it into the path suffix
        output_dir += f"lr_{lr}_time_{include_time}"
        for k, v in model_args.items():
            # Avoid dumping large dicts into the path; model_name already carries _catch/_wst
            if k in ("signature", "catch24", "wst"):
                continue
            output_dir += f"_{k}_{v}"
        output_dir += f"_seed_{seed}"

        # Convert to absolute path to avoid issues with working directory changes
        output_dir = os.path.abspath(output_dir)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Directory {output_dir} has been created.")

    # Save configuration for reproducibility (matching sig_s5 format)
    config_dict = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "print_steps": print_steps,
        "early_stopping_steps": early_stopping_steps,
        "include_time": include_time,
        "metric": metric,
        "classification": classification,
        "output_step": output_step,
        "use_presplit": use_presplit,
        "n_samples": n_samples,
    }
    
    # Add model architecture parameters
    if base_model_name == "sig_mamba":
        config_dict["normalize"] = model_args.get("normalize", False)
        config_dict["model_args"] = {
            "hidden_dim": model_args.get("hidden_dim", 128),
            "state_dim": model_args.get("state_dim", 64),
            "num_blocks": model_args.get("num_blocks", 4),
        }
        config_dict["model_args"]["conv_dim"] = model_args.get("conv_dim", 4)
        config_dict["model_args"]["expansion"] = model_args.get("expansion", 2)
        
        # Add feature configuration
        config_dict["signature_config"] = model_args.get("signature", {})
        config_dict["catch24_config"] = model_args.get("catch24", {})
        config_dict["wst_config"] = model_args.get("wst", {})
    else:
        # For standard mamba models
        config_dict["model_args"] = dict(model_args)
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    indexes = torch.randperm(n_samples)

    # For signature-based models we intentionally do not include the time channel
    # in raw inputs before feature extraction (time augmentation, if any, should
    # be handled at the feature level consistently with CV pipelines).
    include_time_effective = include_time and (base_model_name != "sig_mamba")

    train_dataset = Dataset(
        data_dir,
        dataset_name,
        True,
        False,
        False,
        indexes,
        presplit=use_presplit,
        include_time=include_time_effective,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_dataset = Dataset(
        data_dir,
        dataset_name,
        False,
        True,
        False,
        indexes,
        presplit=use_presplit,
        include_time=include_time_effective,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataset = Dataset(
        data_dir,
        dataset_name,
        False,
        False,
        True,
        indexes,
        presplit=use_presplit,
        include_time=include_time_effective,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    input_dim = train_dataset.input_dim
    output_dim = train_dataset.output_dim

    # If using signature + mamba, transform datasets into log signature feature sequences
    if base_model_name == "sig_mamba":
        # Import log signature feature computation from running_mamba
        try:
            curr_dir = os.path.dirname(__file__)
            repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
            running_mamba_dir = os.path.join(repo_root, "running_mamba")
            if running_mamba_dir not in sys.path:
                sys.path.append(running_mamba_dir)
            try:
                from log_signature_fast import batch_compute_log_features  # type: ignore
            except Exception as e:
                raise ImportError(f"Failed to import log signature computation from log_signature_fast: {e}")
        except Exception as e:
            raise ImportError(f"Failed to import log signature utilities: {e}")

        signature_cfg = model_args.get("signature", None)
        catch24_cfg = model_args.get("catch24", None)
        wst_cfg = model_args.get("wst", None)
        if signature_cfg is None and catch24_cfg is None and wst_cfg is None:
            raise ValueError("sig_mamba requires 'signature', 'catch24', or 'wst' dict in model_args")
        if signature_cfg is not None:
            lengths = [int(x) for x in signature_cfg.get("lengths", [])]
            depths = [int(x) for x in signature_cfg.get("depths", [])]
            stride = int(signature_cfg.get("stride", 1))
            include_global = bool(signature_cfg.get("include_global", False))
            d_global = int(signature_cfg.get("d_global", 0)) if include_global else None
        else:
            lengths = []
            depths = []
            stride = 1
            include_global = False
            d_global = None

        # Convert datasets to numpy (N, C, T)
        def to_nct(tensor):
            arr = tensor.detach().cpu().numpy()  # (N, T, C)
            return np.transpose(arr, (0, 2, 1))   # (N, C, T)

        X_tr_nct = to_nct(train_dataset.data)
        X_va_nct = to_nct(val_dataset.data)
        X_te_nct = to_nct(test_dataset.data)

        _logsig_t0 = time.time()
        if catch24_cfg is not None:
            try:
                curr_dir = os.path.dirname(__file__)
                repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
                running_mamba_dir = os.path.join(repo_root, "running_mamba")
                if running_mamba_dir not in sys.path:
                    sys.path.append(running_mamba_dir)
                from catch24_utils import compute_catch24_features  # type: ignore
            except Exception as e:
                raise ImportError(f"Failed to import catch24 utilities: {e}")
            feats_tr = compute_catch24_features(X_tr_nct)
            feats_va = compute_catch24_features(X_va_nct)
            feats_te = compute_catch24_features(X_te_nct)
            # For catch24, features are (N, 1, D). Keep API similarity by defining T' and D
            T_prime_tr = T_prime_va = T_prime_te = 1
            D_total = feats_tr.shape[-1]
            D_total_va = feats_va.shape[-1]
            D_total_te = feats_te.shape[-1]
        elif wst_cfg is not None:
            try:
                curr_dir = os.path.dirname(__file__)
                repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
                running_mamba_dir = os.path.join(repo_root, "running_mamba")
                if running_mamba_dir not in sys.path:
                    sys.path.append(running_mamba_dir)
                from wst_utils import compute_wst_features_windowed  # type: ignore
            except Exception as e:
                raise ImportError(f"Failed to import WST utilities: {e}")
            wst_lengths = [int(x) for x in wst_cfg.get("lengths", [])]
            wst_stride = int(wst_cfg.get("stride", 1))
            feats_tr = compute_wst_features_windowed(X_tr_nct, wst_lengths, wst_stride)
            feats_va = compute_wst_features_windowed(X_va_nct, wst_lengths, wst_stride)
            feats_te = compute_wst_features_windowed(X_te_nct, wst_lengths, wst_stride)
            # WST features are (N, T', D)
            T_prime_tr = feats_tr.shape[1]
            T_prime_va = feats_va.shape[1]
            T_prime_te = feats_te.shape[1]
            D_total = feats_tr.shape[-1]
            D_total_va = feats_va.shape[-1]
            D_total_te = feats_te.shape[-1]
        else:
            feats_tr, T_prime_tr, D_total = batch_compute_log_features(
                X_tr_nct, lengths, depths, stride, include_global, d_global
            )
            feats_va, T_prime_va, D_total_va = batch_compute_log_features(
                X_va_nct, lengths, depths, stride, include_global, d_global
            )
            feats_te, T_prime_te, D_total_te = batch_compute_log_features(
                X_te_nct, lengths, depths, stride, include_global, d_global
            )
        assert T_prime_tr == T_prime_va == T_prime_te and D_total == D_total_va == D_total_te

        # Wall-clock cost of tokenisation (train+val+test), reported separately
        # from training time.
        logsig_precompute_s = time.time() - _logsig_t0
        print(
            f"[TOKENISER] logsig_precompute_s={logsig_precompute_s:.4f} "
            f"token_dim_total={int(D_total)} n_tokens={int(T_prime_tr)}"
        )

        # Ensure all features are finite before normalization
        feats_tr = np.nan_to_num(feats_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        feats_va = np.nan_to_num(feats_va, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        feats_te = np.nan_to_num(feats_te, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Optional z-score normalization using training-set statistics, matching CV pipelines
        if bool(model_args.get("normalize", False)):
            try:
                mu = feats_tr.mean(axis=(0, 1), keepdims=True)
                std = feats_tr.std(axis=(0, 1), keepdims=True)
                std = std + 1e-6
                feats_tr = (feats_tr - mu) / std
                feats_va = (feats_va - mu) / std
                feats_te = (feats_te - mu) / std

                # Ensure normalized features are still finite
                feats_tr = np.nan_to_num(feats_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                feats_va = np.nan_to_num(feats_va, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                feats_te = np.nan_to_num(feats_te, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            except Exception:
                pass

        # Replace dataloaders with signature feature versions
        def make_loader(X_feats_np, labels_tensor):
            X_t = torch.from_numpy(X_feats_np).to(torch.float32)
            ds = torch.utils.data.TensorDataset(X_t, labels_tensor)
            return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)

        train_dataloader = make_loader(feats_tr, train_dataset.labels)
        val_dataloader = make_loader(feats_va, val_dataset.labels)
        test_dataloader = make_loader(feats_te, test_dataset.labels)

        input_dim = int(D_total)

        # Calculate output subsampling for regression tasks
        # Models output predictions for every feature timestep, but we need to subsample
        # to match the target length (output_step in original space becomes output_step_transformed in feature space)
        if not classification:
            if signature_cfg is not None:
                feature_stride = stride
            elif wst_cfg is not None:
                feature_stride = wst_stride
            else:
                feature_stride = 1  # catch24 has no stride
            # In feature space, we need to output every (output_step / feature_stride) timesteps
            output_step_transformed = max(1, output_step // feature_stride)
        else:
            output_step_transformed = 1  # Not used in classification

    else:
        # For non-signature models, no transformation needed
        output_step_transformed = 1

    # Build model - only sig_mamba supported
    # Use running_mamba's MambaClassification implementation
    try:
        curr_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
        running_mamba_dir = os.path.join(repo_root, "running_mamba")
        if running_mamba_dir not in sys.path:
            sys.path.append(running_mamba_dir)
        from mamba_classification import MambaClassification  # type: ignore
    except Exception as e:
        raise ImportError(f"Failed to import running_mamba MambaClassification: {e}")

    mk = dict(model_args)
    mk.pop("signature", None)

    # Verify mamba_ssm is available
    from mamba_classification import _HAS_MAMBA
    if not _HAS_MAMBA:
        raise ImportError("mamba_ssm not available! Cannot use sig_mamba without real mamba_ssm.")

    print(f"[DEBUG] Creating MambaClassification: input_dim={input_dim}, d_model={mk.get('hidden_dim', 128)}, n_layers={mk.get('num_blocks', 2)}, d_state={mk.get('state_dim', 16)}")

    model = MambaClassification(
        input_size=int(input_dim),
        num_classes=int(output_dim),
        d_model=int(mk.get("hidden_dim", 128)),
        n_layers=int(mk.get("num_blocks", 2)),
        d_state=int(mk.get("state_dim", 16)),
        d_conv=int(mk.get("conv_dim", 4)),
        expand=int(mk.get("expansion", 2)),
        dropout=0.3,  # Dropout set to 0.3 for sig_mamba
        regression=(not classification),  # Set regression mode for non-classification tasks
    ).to(device)

    print(f"[DEBUG] MambaClassification created. Using real mamba_ssm: {_HAS_MAMBA}, Total params: {sum(p.numel() for p in model.parameters())}")

    # Record parameter count, token shape and tokenisation time.
    # Write-only: does not touch training, evaluation, or early stopping.
    try:
        _meta = {
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
        }
        if base_model_name == "sig_mamba":
            _meta["token_dim_total"] = int(input_dim)
            for _k, _v in (
                ("n_tokens_train", "T_prime_tr"),
                ("n_tokens_val", "T_prime_va"),
                ("n_tokens_test", "T_prime_te"),
            ):
                if _v in locals():
                    _meta[_k] = int(locals()[_v])
            if "logsig_precompute_s" in locals():
                _meta["logsig_precompute_s"] = float(locals()["logsig_precompute_s"])
        with open(os.path.join(output_dir, "run_meta.json"), "w") as _f:
            json.dump(_meta, _f, indent=2)
    except Exception:
        pass

    # Use Adam optimizer (no weight decay) for all models to match JAX baseline
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    all_train_metrics = []
    all_val_metrics = []
    val_metric_for_best_model = []
    no_val_improvement = 0.0
    steps = []
    all_time = []
    step = 0
    start = time.time()
    # Reset peak memory stats before training so we capture only the training peak.
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    debug_losses = bool(os.environ.get("SIG_MAMBA_DEBUG_LOSS"))
    debug_interval = int(os.environ.get("SIG_MAMBA_DEBUG_INTERVAL", "50"))
    while step <= num_steps:
        for X, y in train_dataloader:
            optimizer.zero_grad()

            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            if classification:
                loss = torch.nn.functional.cross_entropy(y_hat, y.argmax(dim=1))
            else:
                # For regression: subsample model outputs to match target length
                y_hat_sub = y_hat[:, ::output_step_transformed, 0]
                loss = torch.nn.functional.mse_loss(y_hat_sub, y)
            loss.backward()
            optimizer.step()
            # Capture peak GPU memory after the first forward+backward (warm allocator).
            if step == 0 and torch.cuda.is_available():
                torch.cuda.synchronize(device)
                peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            running_loss += loss.item()
            if debug_losses and (step % max(1, debug_interval) == 0):
                print(f"[DEBUG] step={step} loss={loss.item():.6f}")

            # Periodic CUDA cleanup to prevent memory leaks during training
            if torch.cuda.is_available() and (step + 1) % 100 == 0:
                torch.cuda.empty_cache()

            if (step + 1) % print_steps == 0:

                model.eval()

                train_metric = 0.0
                for X, y in train_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    y_hat = model(X)
                    if classification:
                        metric = (
                            y_hat.argmax(dim=1) == y.argmax(dim=1)
                        ).float().cpu().sum() / len(y)
                    else:
                        y_hat_sub = y_hat[:, ::output_step_transformed, 0]
                        metric = torch.nn.functional.mse_loss(y_hat_sub, y).item()
                    train_metric += metric
                all_train_metrics.append((train_metric / len(train_dataloader)))

                val_metric = 0.0
                for X, y in val_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    y_hat = model(X)
                    if classification:
                        metric = (
                            y_hat.argmax(dim=1) == y.argmax(dim=1)
                        ).float().cpu().sum() / len(y)
                    else:
                        y_hat_sub = y_hat[:, ::output_step_transformed, 0]
                        metric = torch.nn.functional.mse_loss(y_hat_sub, y).item()
                    val_metric += metric
                end = time.time()
                total_time = end - start
                print(
                    f"Step: {step + 1}, "
                    f"Train Metric: {train_metric / len(train_dataloader)},"
                    f"Val Metric: {val_metric / len(val_dataloader)}, "
                    f"Time: {total_time}"
                )
                start = time.time()
                all_val_metrics.append((val_metric / len(val_dataloader)))
                all_time.append(total_time)
                steps.append(step + 1)
                running_loss = 0.0

                # CUDA cleanup after evaluation to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if len(val_metric_for_best_model) == 0 or operator_improv(
                    all_val_metrics[-1], best_val(val_metric_for_best_model)
                ):
                    no_val_improvement = 0.0
                    val_metric_for_best_model.append(all_val_metrics[-1])
                    test_metric = 0.0
                    for X, y in test_dataloader:
                        X = X.to(device)
                        y = y.to(device)
                        y_hat = model(X)
                        if classification:
                            metric = (
                                y_hat.argmax(dim=1) == y.argmax(dim=1)
                            ).float().cpu().sum() / len(y)
                        else:
                            y_hat_sub = y_hat[:, ::output_step_transformed, 0]
                            metric = torch.nn.functional.mse_loss(
                                y_hat_sub, y
                            ).item()
                        test_metric += metric
                    test_metric = test_metric / len(test_dataloader)

                    print(f"Test Metric: {test_metric}")
                if operator_no_improv(
                    all_val_metrics[-1], best_val(val_metric_for_best_model)
                ):
                    no_val_improvement += 1
                    if no_val_improvement > early_stopping_steps:
                        steps_save = np.array(steps)
                        all_train_metrics_save = np.array(all_train_metrics)
                        all_val_metrics_save = np.array(all_val_metrics)
                        all_time_save = np.array(all_time)
                        test_metric = np.array(test_metric)
                        os.makedirs(output_dir, exist_ok=True)
                        np.save(output_dir + "/steps.npy", steps_save)
                        np.save(
                            output_dir + "/all_train_metric.npy", all_train_metrics_save
                        )
                        np.save(
                            output_dir + "/all_val_metric.npy", all_val_metrics_save
                        )
                        np.save(output_dir + "/all_time.npy", all_time_save)
                        np.save(output_dir + "/test_metric.npy", test_metric)
                        np.save(output_dir + "/peak_memory_mb.npy", np.array(peak_memory_mb))

                        # CRITICAL: Cleanup before early stopping return
                        try:
                            del model
                            del optimizer
                            del train_dataloader
                            del val_dataloader
                            del test_dataloader
                            del train_dataset
                            del val_dataset
                            del test_dataset
                        except:
                            pass

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        import gc
                        gc.collect()
                        gc.collect()
                        gc.collect()
                        return

                steps_save = np.array(steps)
                all_train_metrics_save = np.array(all_train_metrics)
                all_val_metrics_save = np.array(all_val_metrics)
                all_time_save = np.array(all_time)
                test_metric = np.array(test_metric)
                os.makedirs(output_dir, exist_ok=True)
                np.save(output_dir + "/steps.npy", steps_save)
                np.save(output_dir + "/all_train_metric.npy", all_train_metrics_save)
                np.save(output_dir + "/all_val_metric.npy", all_val_metrics_save)
                np.save(output_dir + "/all_time.npy", all_time_save)
                np.save(output_dir + "/test_metric.npy", test_metric)
                np.save(output_dir + "/peak_memory_mb.npy", np.array(peak_memory_mb))
            model.train()
            step += 1

    # CRITICAL: Aggressive cleanup after training completes to prevent segfaults between seeds
    # Delete model and optimizer to free all CUDA memory before next seed starts
    try:
        del model
        del optimizer
        del train_dataloader
        del val_dataloader
        del test_dataloader
        del train_dataset
        del val_dataset
        del test_dataset
    except:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()
    gc.collect()
    gc.collect()

def prepare_sig_mamba_training_components(
    *,
    data_dir,
    dataset_name,
    n_samples,
    batch_size,
    include_time,
    output_step,
    use_presplit,
    model_args,
    lr,
    metric,
    seed: int = 0,
):
    """
    Build sig_mamba model, optimizer, and training DataLoader consistent with create_dataset_model_and_train.
    Returns: (model, optimizer, train_dataloader, device, classification, output_step_transformed)
    """
    # Reproducibility similar to main training entrypoint (lighter settings)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This helper is specialized for sig_mamba path
    base_model_name = "sig_mamba"
    classification = metric == "accuracy"

    # Index pool for Dataset wrapper
    indexes = torch.randperm(n_samples)

    # Signature-based models do not include raw time channel in Dataset
    include_time_effective = include_time and (base_model_name != "sig_mamba")

    train_dataset = Dataset(
        data_dir,
        dataset_name,
        True,
        False,
        False,
        indexes,
        presplit=use_presplit,
        include_time=include_time_effective,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    input_dim = train_dataset.input_dim
    output_dim = train_dataset.output_dim

    # Transform to signature/catch24/WST features (required for sig_mamba)
    try:
        curr_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
        running_mamba_dir = os.path.join(repo_root, "running_mamba")
        if running_mamba_dir not in sys.path:
            sys.path.append(running_mamba_dir)
        from log_signature_fast import batch_compute_log_features  # type: ignore
    except Exception as e:
        raise ImportError(f"Failed to import log signature computation from log_signature_fast: {e}")

    signature_cfg = model_args.get("signature", None)
    catch24_cfg = model_args.get("catch24", None)
    wst_cfg = model_args.get("wst", None)
    if signature_cfg is None and catch24_cfg is None and wst_cfg is None:
        raise ValueError("sig_mamba requires 'signature', 'catch24', or 'wst' dict in model_args")

    if signature_cfg is not None:
        lengths = [int(x) for x in signature_cfg.get("lengths", [])]
        depths = [int(x) for x in signature_cfg.get("depths", [])]
        stride = int(signature_cfg.get("stride", 1))
        include_global = bool(signature_cfg.get("include_global", False))
        d_global = int(signature_cfg.get("d_global", 0)) if include_global else None
    else:
        lengths = []
        depths = []
        stride = 1
        include_global = False
        d_global = None

    # Convert to (N, C, T) for feature computation
    def to_nct(tensor):
        arr = tensor.detach().cpu().numpy()
        return np.transpose(arr, (0, 2, 1))

    X_tr_nct = to_nct(train_dataset.data)

    if catch24_cfg is not None:
        try:
            curr_dir = os.path.dirname(__file__)
            repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
            running_mamba_dir = os.path.join(repo_root, "running_mamba")
            if running_mamba_dir not in sys.path:
                sys.path.append(running_mamba_dir)
            from catch24_utils import compute_catch24_features  # type: ignore
        except Exception as e:
            raise ImportError(f"Failed to import catch24 utilities: {e}")
        feats_tr = compute_catch24_features(X_tr_nct)
        T_prime_tr = 1
        D_total = feats_tr.shape[-1]
    elif wst_cfg is not None:
        try:
            curr_dir = os.path.dirname(__file__)
            repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
            running_mamba_dir = os.path.join(repo_root, "running_mamba")
            if running_mamba_dir not in sys.path:
                sys.path.append(running_mamba_dir)
            from wst_utils import compute_wst_features_windowed  # type: ignore
        except Exception as e:
            raise ImportError(f"Failed to import WST utilities: {e}")
        wst_lengths = [int(x) for x in wst_cfg.get("lengths", [])]
        wst_stride = int(wst_cfg.get("stride", 1))
        feats_tr = compute_wst_features_windowed(X_tr_nct, wst_lengths, wst_stride)
        T_prime_tr = feats_tr.shape[1]
        D_total = feats_tr.shape[-1]
    else:
        feats_tr, T_prime_tr, D_total = batch_compute_log_features(
            X_tr_nct, lengths, depths, stride, include_global, d_global
        )

    # Ensure numerical safety
    feats_tr = np.nan_to_num(feats_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Optional z-scoring following training path
    if bool(model_args.get("normalize", False)):
        try:
            mu = feats_tr.mean(axis=(0, 1), keepdims=True)
            std = feats_tr.std(axis=(0, 1), keepdims=True) + 1e-6
            feats_tr = (feats_tr - mu) / std
            feats_tr = np.nan_to_num(feats_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        except Exception:
            pass

    # Replace train loader with feature dataset
    X_t = torch.from_numpy(feats_tr).to(torch.float32)
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, train_dataset.labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    input_dim = int(D_total)

    # Compute output subsampling for regression in feature space
    if not classification:
        if signature_cfg is not None:
            feature_stride = stride
        elif wst_cfg is not None:
            feature_stride = wst_stride
        else:
            feature_stride = 1
        output_step_transformed = max(1, output_step // feature_stride)
    else:
        output_step_transformed = 1

    # Build running_mamba classification/regression model
    try:
        curr_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
        running_mamba_dir = os.path.join(repo_root, "running_mamba")
        if running_mamba_dir not in sys.path:
            sys.path.append(running_mamba_dir)
        from mamba_classification import MambaClassification  # type: ignore
        from mamba_classification import _HAS_MAMBA  # type: ignore
    except Exception as e:
        raise ImportError(f"Failed to import running_mamba MambaClassification: {e}")
    if not _HAS_MAMBA:
        raise ImportError("mamba_ssm not available! Cannot use sig_mamba without real mamba_ssm.")

    mk = dict(model_args)
    mk.pop("signature", None)

    model = MambaClassification(
        input_size=int(input_dim),
        num_classes=int(output_dim),
        d_model=int(mk.get("hidden_dim", 128)),
        n_layers=int(mk.get("num_blocks", 2)),
        d_state=int(mk.get("state_dim", 16)),
        d_conv=int(mk.get("conv_dim", 4)),
        expand=int(mk.get("expansion", 2)),
        dropout=0.3,
        regression=(not classification),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer, train_dataloader, device, classification, output_step_transformed
