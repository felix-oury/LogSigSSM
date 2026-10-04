"""
Training-step benchmark for LogSig-SSM (Table 5): time for 1000 training steps
and GPU memory, with the evaluation passes excluded.

For each dataset the configuration in experiment_configs/repeats/sig_mamba/
(or --experiment_folder) is built, 10 warm-up steps are run, and then 1000
timed optimisation steps (forward, backward, Adam step). Tokenisation happens
once, before the timed section; training runs log its cost in run_meta.json
(Table 6).

Reported per dataset: runtime of the 1000 steps, peak and average allocated
memory, and peak and average reserved memory (torch.cuda caching allocator),
sampled after every step.

At these model sizes a step takes a few milliseconds and its time is set by
the CPU side (data loading, kernel launches), so pin the process to a single
performance core; on a hybrid CPU an unpinned run can take several times as
long. Results go to reruns/sig_mamba_benchmark/ by default; the recorded runs
are in outputs/sig_mamba_benchmark/.

    taskset -c 2 python benchmarks/bench_sig_mamba.py
"""

import os
import json
import platform
import time
from datetime import datetime

import torch
import sys
import argparse

# Ensure sibling package 'torch_experiments' is importable when running this script directly
_SCRIPT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

try:
    import pynvml  # type: ignore
    _HAVE_NVML = True
except Exception:
    _HAVE_NVML = False

from torch_experiments.train import prepare_sig_mamba_training_components


def count_params_and_memory_mb(model: torch.nn.Module):
    num_params = 0
    bytes_total = 0
    for p in model.parameters():
        n = p.numel()
        num_params += n
        bytes_total += n * p.element_size()
    for b in model.buffers():
        bytes_total += b.numel() * b.element_size()
    return num_params, bytes_total / (1024 ** 2)


class GpuUtilSampler:
    def __init__(self, device_index=0):
        self.enabled = _HAVE_NVML and torch.cuda.is_available()
        if self.enabled:
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.samples = []

    def sample(self):
        if not self.enabled:
            return
        util = pynvml.nvmlDeviceGetUtilizationRates(self.h)
        self.samples.append(util.gpu)

    def stats(self):
        if not self.samples:
            return None, None
        mean_util = float(sum(self.samples)) / len(self.samples)
        return mean_util, max(self.samples)


def benchmark_steps(model, optimizer, train_loader, device, *, classification, output_step_transformed, warmup=10, steps=1000):
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    util = GpuUtilSampler(device_index=torch.cuda.current_device() if torch.cuda.is_available() else 0)
    alloc_samples_mb = []
    reserved_samples_mb = []

    # Warmup (not timed)
    warmup = max(0, int(warmup))
    it = iter(train_loader)
    for _ in range(warmup):
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            X, y = next(it)
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(X)
        if classification:
            loss = torch.nn.functional.cross_entropy(y_hat, y.argmax(dim=1))
        else:
            y_hat_sub = y_hat[:, ::output_step_transformed, 0]
            loss = torch.nn.functional.mse_loss(y_hat_sub, y)
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    # Timed section
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    steps_done = 0
    it = iter(train_loader)
    while steps_done < steps:
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            X, y = next(it)
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(X)
        if classification:
            loss = torch.nn.functional.cross_entropy(y_hat, y.argmax(dim=1))
        else:
            y_hat_sub = y_hat[:, ::output_step_transformed, 0]
            loss = torch.nn.functional.mse_loss(y_hat_sub, y)
        loss.backward()
        optimizer.step()
        util.sample()
        if torch.cuda.is_available():
            alloc_samples_mb.append(torch.cuda.memory_allocated(device) / (1024 ** 2))
            reserved_samples_mb.append(torch.cuda.memory_reserved(device) / (1024 ** 2))
        steps_done += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    runtime_s = end - start
    steps_per_s = steps_done / runtime_s if runtime_s > 0 else float("inf")
    peak_alloc_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else None
    peak_reserved_mb = (torch.cuda.max_memory_reserved(device) / (1024 ** 2)) if torch.cuda.is_available() else None
    avg_alloc_mb = (sum(alloc_samples_mb) / len(alloc_samples_mb)) if alloc_samples_mb else None
    avg_reserved_mb = (sum(reserved_samples_mb) / len(reserved_samples_mb)) if reserved_samples_mb else None
    util_mean, util_max = util.stats()
    return {
        "runtime_s_1000": runtime_s,
        "steps_per_s": steps_per_s,
        "peak_alloc_mb": peak_alloc_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "avg_alloc_mb": avg_alloc_mb,
        "avg_reserved_mb": avg_reserved_mb,
        "gpu_util_mean": util_mean,
        "gpu_util_max": util_max,
    }


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def environment():
    env = {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cpu": cpu_name(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    for name in ("mamba_ssm", "causal_conv1d"):
        try:
            env[name] = __import__(name).__version__
        except Exception:
            env[name] = None
    return env


def pick_cfg_path(experiment_folder, model_name, dataset_name, prefer_wst=False, prefer_catch24=False):
    base_dir = os.path.join(experiment_folder, f"{model_name}")
    path_wst = os.path.join(base_dir, f"{dataset_name}_wst.json")
    path_catch24 = os.path.join(base_dir, f"{dataset_name}_catch24.json")
    path_std = os.path.join(base_dir, f"{dataset_name}.json")
    order = [path_std, path_wst, path_catch24]
    if prefer_wst:
        order = [path_wst, path_catch24, path_std]
    elif prefer_catch24:
        order = [path_catch24, path_wst, path_std]
    for p in order:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No config found for {dataset_name} in {base_dir} (tried {order})")


def main(
    *,
    model_name,
    datasets,
    experiment_folder,
    output_dir,
    prefer_wst=False,
    prefer_catch24=False,
    double_dropout=True,
):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for dataset_name in datasets:
        cfg_path = pick_cfg_path(experiment_folder, model_name, dataset_name, prefer_wst, prefer_catch24)
        cfg = json.load(open(cfg_path, "r"))

        data_dir = cfg["data_dir"]
        metric = cfg["metric"]
        use_presplit = cfg["use_presplit"]
        batch_size = int(cfg.get("batch_size", 32))
        include_time = cfg["time"].lower() == "true"
        lr = float(cfg["lr"])
        normalize = bool(cfg.get("normalize", False))
        hidden_dim = int(cfg.get("hidden_dim", 128))
        ssm_dim = int(cfg.get("ssm_dim", 64))
        num_blocks = int(cfg.get("num_blocks", 4))
        conv_dim = int(cfg.get("convdim", cfg.get("conv_dim", 3)))
        expansion = int(cfg.get("expansion", cfg.get("expand", 2)))
        output_step = int(cfg["output_step"]) if dataset_name == "ppg" else 1

        exps_n_samples = {
            "EigenWorms": 236,
            "EthanolConcentration": 524,
            "Heartbeat": 409,
            "MotorImagery": 378,
            "SelfRegulationSCP1": 561,
            "SelfRegulationSCP2": 380,
            "ppg": 1232,
            "signature1": 70000,
            "signature2": 70000,
            "signature3": 70000,
            "signature4": 70000,
        }
        lookup_name = dataset_name.split("_miss")[0] if "_miss" in dataset_name else dataset_name
        n_samples = exps_n_samples[lookup_name]

        model_args = {
            "num_blocks": num_blocks,
            "hidden_dim": hidden_dim,
            "state_dim": ssm_dim,
            "conv_dim": conv_dim,
            "expansion": expansion,
            "double_dropout": bool(double_dropout),
            "normalize": normalize,
        }
        if model_name == "sig_mamba":
            if "signature" in cfg:
                model_args["signature"] = cfg["signature"]
            if "catch24" in cfg:
                model_args["catch24"] = cfg["catch24"]
            if "wst" in cfg:
                model_args["wst"] = cfg["wst"]
            model, optimizer, train_loader, device, classification, output_step_transformed = prepare_sig_mamba_training_components(
                data_dir=data_dir,
                dataset_name=dataset_name,
                n_samples=n_samples,
                batch_size=batch_size,
                include_time=include_time,
                output_step=output_step,
                use_presplit=use_presplit,
                model_args=model_args,
                lr=lr,
                metric=metric,
                seed=0,
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Use 'sig_mamba'.")

        num_params, param_mem_mb = count_params_and_memory_mb(model)
        bench = benchmark_steps(
            model, optimizer, train_loader, device,
            classification=classification,
            output_step_transformed=output_step_transformed,
            warmup=10,
            steps=1000,
        )

        result = {
            "dataset": dataset_name,
            "num_params": int(num_params),
            "param_mem_mb": round(float(param_mem_mb), 3),
            "runtime_s_1000": round(float(bench["runtime_s_1000"]), 3),
            "steps_per_s": round(float(bench["steps_per_s"]), 3),
            "peak_alloc_mb": None if bench["peak_alloc_mb"] is None else round(float(bench["peak_alloc_mb"]), 3),
            "peak_reserved_mb": None if bench["peak_reserved_mb"] is None else round(float(bench["peak_reserved_mb"]), 3),
            "avg_alloc_mb": None if bench["avg_alloc_mb"] is None else round(float(bench["avg_alloc_mb"]), 3),
            "avg_reserved_mb": None if bench["avg_reserved_mb"] is None else round(float(bench["avg_reserved_mb"]), 3),
            "gpu_util_mean": bench["gpu_util_mean"],
            "gpu_util_max": bench["gpu_util_max"],
        }
        rows.append(result)

        # Free ASAP
        try:
            del model
            del optimizer
            del train_loader
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(output_dir, f"{model_name}_bench_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({"environment": environment(), "results": rows}, f, indent=2)

    md_path = os.path.join(output_dir, f"{model_name}_bench_{ts}.md")
    with open(md_path, "w") as f:
        f.write("| Dataset | #Params | Param Mem (MB) | Runtime 1000 (s) | Steps/s | Peak Alloc (MB) | Peak Reserved (MB) | Avg Alloc (MB) | Avg Reserved (MB) | GPU Util Mean (%) | GPU Util Max (%) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['dataset']} | {r['num_params']:,} | {r['param_mem_mb']} | "
                f"{r['runtime_s_1000']} | {r['steps_per_s']} | "
                f"{'' if r['peak_alloc_mb'] is None else r['peak_alloc_mb']} | "
                f"{'' if r['peak_reserved_mb'] is None else r['peak_reserved_mb']} | "
                f"{'' if r['avg_alloc_mb'] is None else r['avg_alloc_mb']} | "
                f"{'' if r['avg_reserved_mb'] is None else r['avg_reserved_mb']} | "
                f"{'' if r['gpu_util_mean'] is None else round(float(r['gpu_util_mean']), 1)} | "
                f"{'' if r['gpu_util_max'] is None else round(float(r['gpu_util_max']), 1)} |\n"
            )
    print(f"Wrote: {json_path}\nWrote: {md_path}")


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sig_mamba", choices=["sig_mamba"], help="Model to benchmark")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated datasets; default to standard UEA list")
    parser.add_argument("--experiment_folder", type=str, default="experiment_configs/repeats", help="Path to config folder")
    parser.add_argument("--output_dir", type=str, default="reruns/sig_mamba_benchmark", help="Output directory")
    parser.add_argument("--prefer_wst", action="store_true")
    parser.add_argument("--prefer_catch24", action="store_true")
    parser.add_argument("--double_dropout", type=str, default="true")
    args = parser.parse_args()

    dd_str = (args.double_dropout or "true").strip().lower()
    if dd_str in ("1", "true", "yes", "y", "on"):
        dd = True
    elif dd_str in ("0", "false", "no", "n", "off"):
        dd = False
    else:
        raise SystemExit(f"Invalid --double_dropout value: {args.double_dropout}. Use true/false.")

    # Default: UEA classification datasets
    datasets = [
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "EthanolConcentration",
        "Heartbeat",
        "MotorImagery",
        "EigenWorms",
    ]
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    main(
        model_name=args.model,
        datasets=datasets,
        experiment_folder=args.experiment_folder,
        output_dir=args.output_dir,
        prefer_wst=args.prefer_wst,
        prefer_catch24=args.prefer_catch24,
        double_dropout=dd,
    )
