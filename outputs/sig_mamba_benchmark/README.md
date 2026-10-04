# Timing and GPU-memory benchmark (Tables 2 and 5)

[`benchmarks/bench_sig_mamba.py`](../../benchmarks/bench_sig_mamba.py) builds a LogSig-SSM
configuration, runs 10 warm-up steps and then times 1000 training steps (forward, backward, Adam
step). Tokenisation happens once beforehand and evaluation is not run, so neither is timed. After
every step it samples the allocated and reserved memory of the CUDA caching allocator and reports
their peak and average.

```bash
taskset -c 2 python benchmarks/bench_sig_mamba.py
```

A step takes a few milliseconds and its time is set by the CPU (data loading, kernel launches), so
pin the process to a single performance core. Memory does not depend on the core. Average reserved
memory can move by a few MB with the datasets run earlier in the same process: EigenWorms reserves
362 MB when it runs last and 368 MB when it runs first.

## Folders

| Folder | Contents |
|---|---|
| `2026-01-26_training_run/` | EigenWorms memory logged during a full training run (`avg_reserved_mb`: 364.4) |
| `2026-05-05_submission/` | The two benchmark runs made for the submission, on an RTX 4090, and the configurations they used (`configs/`) |
| `2026-10-03_rerun/current_configs/` | Four runs of the current configurations (`experiment_configs/repeats/sig_mamba/`) on one performance core |
| `2026-10-03_rerun/submission_configs/` | One run of the configurations in `2026-05-05_submission/configs/` on one performance core |
| `2026-10-03_rerun/core_pinning/` | The benchmark on a performance core (CPU 2), an efficiency core (CPU 24) and unpinned |

The submission configurations differ from the current ones for SCP1 (`conv_dim` 2) and for Ethanol,
Heartbeat and Motor (earlier tokenisers); EigenWorms and SCP2 are unchanged.

The 2026-10-03 runs were made on an RTX 4090 with an Intel i9-14900KF, Python 3.10, torch 2.4.1
(CUDA 12.1), mamba-ssm 2.2.6.post3 and causal-conv1d 1.5.4. Each file records the GPU and library
versions in its `environment` field, and from `sig_mamba_bench_20261003-191433` on also the CPU and
core affinity. The earlier files of that day ran on CPU 2 (`current_configs/*-191243`, `*-191311`
and `core_pinning/p_core/`), on CPU 24 (`core_pinning/e_core/`) or unpinned
(`core_pinning/unpinned/`).

## Results

Current configurations, one performance core, four runs (ranges over the runs):

| Dataset | Parameters | Time / 1000 steps (s) | Peak allocated (MB) | Average reserved (MB) |
|---|---:|---:|---:|---:|
| EigenWorms | 52,933 | 3.59–3.60 | 262 | 362 |
| SCP1 | 310,786 | 3.16–3.17 | 175 | 202 |
| SCP2 | 352,642 | 1.78–1.81 | 52 | 76 |
| Ethanol | 158,516 | 3.81–3.85 | 60 | 82 |
| Heartbeat | 396,546 | 2.12–2.20 | 37 | 54 |
| Motor | 70,034 | 5.24–5.34 | 52 | 78 |

Tables 2 and 5 report these runs for LogSig-SSM: parameter count, average reserved memory and time
per 1000 steps, rounded.

Submission configurations, two runs on 5 May 2026 and one on 3 October 2026. Parameter counts and
peak allocated memory are identical on both dates, and so is average reserved memory, except for
SCP1, which reserved 200 or 202 MB:

| Dataset | Parameters | Time, 5 May (s) | Time, 3 Oct (s) | Peak allocated (MB) | Average reserved (MB) |
|---|---:|---:|---:|---:|---:|
| EigenWorms | 52,933 | 3.60–3.61 | 3.61 | 262 | 362 |
| SCP1 | 310,274 | 3.16 | 3.17 | 175 | 200–202 |
| SCP2 | 352,642 | 1.79–1.89 | 1.79 | 52 | 76 |
| Ethanol | 158,468 | 3.86–4.02 | 3.84 | 59 | 62 |
| Heartbeat | 162,306 | 1.92–1.93 | 1.89 | 90 | 110 |
| Motor | 6,546 | 1.87–1.96 | 1.85 | 22 | 46 |

SCP1 has 310,274 parameters here and 310,786 in the current configuration because its convolution
width went from 2 to 3, which adds 256 parameters in each of its two blocks.

Time per 1000 steps with the current configurations, by CPU core (`core_pinning/`):

| Dataset | Performance core | Efficiency core | Unpinned |
|---|---:|---:|---:|
| EigenWorms | 3.60 | 4.81 | 6.19 |
| SCP1 | 3.16 | 4.53 | 6.51 |
| SCP2 | 1.79 | 4.50 | 1.73 |

Unpinned, Heartbeat took 9.37 s against 2.12–2.20 s on a performance core, and SCP1 with the
submission configuration 7.78 s against 3.17 s.
