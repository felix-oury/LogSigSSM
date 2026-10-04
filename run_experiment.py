"""
Experiment runner for training sequence models on time series datasets.

This script loads hyperparameters from JSON configuration files and trains models using
centralized training functions from `train.py` (JAX) or `torch_experiments/train.py` (PyTorch).

Training Flows:
- PyTorch models (sig_mamba, sig_transformers): Use `torch_experiments/train.py`
  - Handles signature/catch24/WST feature computation automatically
  - Supports both UEA classification and PPG regression
- JAX models (sig_s5): Use `train.py`
  - Handles signature feature computation only (no catch24/WST)
  - Supports both UEA classification and PPG regression

Early Stopping: All training paths use aligned two-check early stopping pattern for fair benchmarking.

Arguments for `run_experiments`:
- `model_names`: List of model architectures (sig_s5, sig_mamba, sig_transformers)
- `dataset_names`: List of datasets to train on
- `experiment_folder`: Directory containing JSON configuration files
- `pytorch_experiments`: Boolean for PyTorch (True) vs JAX (False)
- `prefer_catch24`, `prefer_wst`: Feature preference flags (only for sig_mamba / sig_transformers)
- `use_ppg`: Force PPG dataset usage

Usage:
python run_experiment.py --pytorch_experiments --models sig_mamba --datasets EigenWorms
python run_experiment.py --pytorch_experiments --models sig_mamba --ppg
python run_experiment.py --pytorch_experiments --models sig_mamba --datasets EigenWorms \
    --config experiment_configs/ablations/depth1/sig_mamba/EigenWorms.json \
    --output_dir reruns/sig_mamba_depth1/EigenWorms
python run_experiment.py --models sig_s5 --datasets EigenWorms
"""

import argparse
import json
import os
import sys


# Removed _seed_worker and _run_seed_isolated functions
# These were workarounds for CUDA initialization issues with multiprocessing spawn
# Now that MambaLayer initializes on CPU (multiprocessing-safe), they're no longer needed

# Models that use signature-feature preprocessing in the PyTorch branch
SIG_FEATURE_MODELS = ("sig_mamba", "sig_transformers")


def run_experiments(model_names, dataset_names, experiment_folder, pytorch_experiments, prefer_catch24=False, use_ppg=False, prefer_wst=False, use_toy=False, seed_override=None, config_override=None, output_dir_override=None):

    for model_name in model_names:
        for dataset_name in dataset_names:
            # If --ppg is passed, prefer routing to the PPG dataset and compatible model variants
            if use_ppg:
                dataset_name = "ppg"
            # Keep signature models on PPG to ensure fair benchmarking
            effective_model_name = model_name
            # Support standard, _wst and _catch24-suffixed JSON filenames; honor prefer flags
            base_dir = os.path.join(experiment_folder, f"{effective_model_name}")
            path_wst = os.path.join(base_dir, f"{dataset_name}_wst.json")
            path_catch24 = os.path.join(base_dir, f"{dataset_name}_catch24.json")
            path_std = os.path.join(base_dir, f"{dataset_name}.json")
            # Order of preference
            # For sig_s5, ignore prefer flags and only use the base .json config
            if (not pytorch_experiments) and (effective_model_name == "sig_s5"):
                candidate_paths = [path_std]
            else:
                if bool(prefer_wst):
                    candidate_paths = [path_wst, path_catch24, path_std]
                elif bool(prefer_catch24):
                    candidate_paths = [path_catch24, path_wst, path_std]
                else:
                    candidate_paths = [path_std, path_wst, path_catch24]
            # An explicit --config bypasses the lookup entirely.
            if config_override is not None:
                candidate_paths = [config_override]
            cfg_path = None
            for cp in candidate_paths:
                if os.path.exists(cp):
                    cfg_path = cp
                    break
            if cfg_path is None:
                raise FileNotFoundError(f"No config found for {dataset_name} in {base_dir} (tried: {candidate_paths})")
            with open(cfg_path, "r") as file:
                data = json.load(file)

            seeds = data["seeds"]
            if seed_override:
                seeds = seed_override
            data_dir = data["data_dir"]
            output_parent_dir = data["output_parent_dir"]
            lr_scheduler = eval(data["lr_scheduler"])
            num_steps = data["num_steps"]
            print_steps = data["print_steps"]
            early_stopping_steps = data["early_stopping_steps"]
            batch_size = int(data.get("batch_size", 4 if dataset_name == "ppg" else 32))
            metric = data["metric"]
            use_presplit = data["use_presplit"]
            T = data["T"]
            # Disable weight decay; not taken from hyperparameters
            #weight_decay = 0.01
            if effective_model_name in ("sig_mamba", "sig_s5", "sig_transformers"):
                dt0 = None
            else:
                dt0 = float(data["dt0"])
            scale = data["scale"]
            normalize = bool(data.get("normalize", False))
            lr = float(data["lr"])
            include_time = data["time"].lower() == "true"
            # Some feature-only configs (e.g., WST) omit model dims; provide safe defaults
            hidden_dim = int(data.get("hidden_dim", 128))
            ssm_dim = int(data.get("ssm_dim", 64))
            stepsize = 1
            logsig_depth = 1
            num_blocks = int(data.get("num_blocks", 4))
            ssm_blocks = int(data.get("ssm_blocks", 1)) if effective_model_name == "sig_s5" else None
            if dataset_name == "ppg":
                output_step = int(data["output_step"])
            else:
                output_step = 1
            if effective_model_name == "sig_mamba":
                conv_dim = int(data["convdim"]) if "convdim" in data else int(data.get("conv_dim", 3))
                expansion = int(data["expansion"]) if "expansion" in data else int(data.get("expand", 2))
            else:
                conv_dim = None
                expansion = None

            if pytorch_experiments:
                from torch_experiments.train import (
                    create_dataset_model_and_train as torch_create_dataset_model_and_train,
                )

                exps_n_samples = {
                    "EigenWorms": 236,
                    "EthanolConcentration": 524,
                    "Heartbeat": 409,
                    "MotorImagery": 378,
                    "SelfRegulationSCP1": 561,
                    "SelfRegulationSCP2": 380,
                    "ppg": 1232,
                    "signature1": 70000,  # 70% of 100000 for training
                    "signature2": 70000,
                    "signature3": 70000,
                    "signature4": 70000,
                }
                # Extract base dataset name for missingness studies (e.g., "EigenWorms_miss100_drop" → "EigenWorms")
                lookup_name = dataset_name.split("_miss")[0] if "_miss" in dataset_name else dataset_name
                n_samples = exps_n_samples[lookup_name]

                model_args = {
                    "num_blocks": num_blocks,
                    "hidden_dim": hidden_dim,
                    "state_dim": ssm_dim,
                    "conv_dim": conv_dim,
                    "expansion": expansion,
                }
                # If this is a signature-feature PyTorch model, pass signature config through
                if effective_model_name in SIG_FEATURE_MODELS:
                    # Pass normalization preference to PyTorch path so feature z-scoring matches CV pipelines
                    model_args["normalize"] = bool(normalize)
                    # Transformer-specific architecture args
                    if effective_model_name == "sig_transformers":
                        model_args["n_heads"] = int(data.get("n_heads", 4))
                        model_args["ffn_dim"] = int(data.get("ffn_dim", 256))
                    if "signature" in data:
                        model_args["signature"] = data["signature"]
                    if "catch24" in data:
                        model_args["catch24"] = data["catch24"]
                    if "wst" in data:
                        model_args["wst"] = data["wst"]
                run_args = {
                    "data_dir": data_dir,
                    "output_parent_dir": output_parent_dir,
                    "model_name": effective_model_name,
                    "metric": metric,
                    "batch_size": batch_size,
                    "dataset_name": dataset_name,
                    "n_samples": n_samples,
                    "output_step": output_step,
                    "use_presplit": use_presplit,
                    "include_time": include_time,
                    "num_steps": num_steps,
                    "print_steps": print_steps,
                    "early_stopping_steps": early_stopping_steps,
                    "lr": lr,
                    "model_args": model_args,
                }
                # Distinguish feature variants in model_name for output folder separation
                try:
                    if effective_model_name in SIG_FEATURE_MODELS:
                        if "wst" in data:
                            run_args["model_name"] = f"{effective_model_name}_wst"
                        elif "catch24" in data:
                            run_args["model_name"] = f"{effective_model_name}_catch"
                except Exception:
                    pass
                run_fn = torch_create_dataset_model_and_train
            else:
                # JAX branch: sig_s5 uses signature features + JAX S5 classifier
                if effective_model_name == "sig_s5":
                    from train import create_sig_s5_model_and_train

                    # Prepare model_args for sig_s5
                    model_args = {
                        "signature": data.get("signature", {}),
                        "normalize": normalize,
                        "hidden_dim": int(data.get("hidden_dim", 128)),
                        "ssm_dim": int(data.get("ssm_dim", 64)),
                        "num_blocks": int(data.get("num_blocks", 4)),
                        "ssm_blocks": int(data.get("ssm_blocks", 1)),
                    }

                    # Run training for each seed
                    for seed in seeds:
                        create_sig_s5_model_and_train(
                            seed=seed,
                            data_dir=data_dir,
                            use_presplit=use_presplit,
                            dataset_name=dataset_name,
                            output_step=output_step,
                            metric=metric,
                            include_time=include_time,
                            T=T,
                            model_name="sig_s5",
                            stepsize=stepsize,
                            logsig_depth=logsig_depth,
                            model_args=model_args,
                            num_steps=num_steps,
                            print_steps=print_steps,
                            early_stopping_steps=early_stopping_steps,
                            lr=float(data.get("lr", 1e-3)),
                            lr_scheduler=lr_scheduler,
                            batch_size=int(data.get("batch_size", 4)),
                            output_parent_dir=output_parent_dir,
                        )

                    continue


            # Run training for each seed
            # All models now use the same training loop (no special isolation needed)
            for seed in seeds:
                print(f"Running experiment with seed: {seed}")
                if output_dir_override is not None:
                    seed_dir = (
                        os.path.join(output_dir_override, f"seed_{seed}")
                        if len(seeds) > 1
                        else output_dir_override
                    )
                    run_fn(seed=seed, output_dir_override=seed_dir, **run_args)
                else:
                    run_fn(seed=seed, **run_args)


if __name__ == "__main__":

    args = argparse.ArgumentParser()

    args.add_argument("--pytorch_experiments", action="store_true")
    args.add_argument("--catch24", action="store_true", help="Prefer _catch24.json configs over signature configs")
    args.add_argument("--wst", action="store_true", help="Prefer _wst.json configs over signature/catch24 configs")
    args.add_argument("--ppg", action="store_true", help="Use PPG dataset defaults and map to regression variants where needed")
    args.add_argument("--models", type=str, default="", help="Comma-separated list of model names to run (overrides defaults)")
    args.add_argument("--datasets", type=str, default="", help="Comma-separated list of dataset names to run (overrides defaults)")
    args.add_argument("--seeds", type=str, default="", help="Comma-separated list of seeds to override config seeds")
    args.add_argument("--config", type=str, default=None,
                      help="explicit path to a single JSON config (bypasses config lookup)")
    args.add_argument("--output_dir", type=str, default=None,
                      help="explicit output directory for this run (one seed_<seed> subfolder per seed)")
    args = args.parse_args()
    pytorch_experiments = args.pytorch_experiments
    prefer_catch24 = args.catch24
    prefer_wst = args.wst
    use_ppg = args.ppg

    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        if pytorch_experiments:
            model_names = ["sig_mamba"]
        else:
            model_names = ["sig_s5"]
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        if use_ppg:
            dataset_names = ["ppg"]
        else:
            # Default to UEA datasets only (no PPG). Fix missing comma between Heartbeat and MotorImagery.
            dataset_names = [
                "SelfRegulationSCP1",
                "SelfRegulationSCP2",
                "EthanolConcentration",
                "Heartbeat",
                "MotorImagery",
                "EigenWorms",
            ]
    experiment_folder = "experiment_configs/repeats"

    if args.seeds:
        try:
            seed_override = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        except ValueError:
            raise SystemExit(f"Invalid --seeds value: {args.seeds}")
    else:
        seed_override = None

    # seed_override must be passed by keyword: the eighth positional slot is use_toy.
    run_experiments(model_names, dataset_names, experiment_folder, pytorch_experiments, prefer_catch24, use_ppg, prefer_wst, seed_override=seed_override, config_override=args.config, output_dir_override=args.output_dir)