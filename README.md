# LogSig-SSM

Code and results for **LogSig-SSM: Time-Series Modelling with Multi-Scale Log-Signature Compression
for State-Space Models**, by Felix Oury, Nicolas Calvo Peiro and Reiko J. Tanaka (NeurIPS 2026).

LogSig-SSM compresses a long multivariate time series into a short sequence of tokens, each built from
truncated log-signatures of windows of several lengths, and processes the tokens with a selective
state-space model (Mamba). The depth-two log-signature terms carry the cross-channel geometry of the
path (Lévy areas), which a single diagonal Mamba block cannot compute from the raw input, and the
shorter sequence reduces the cost of the scan. On the six long-sequence UEA datasets LogSig-SSM reaches
71.7% average accuracy, and on EigenWorms (17,984 time steps) it trains in 4 s per 1000 steps with
362 MB of GPU memory, against 122 s and 13,486 MB for Mamba on the raw input.

## Model

```
series x (T × d) ─► tokeniser ─► tokens U (T' × D) ─► linear ─► Mamba blocks ─► mean over tokens ─► linear ─► prediction
```

**Tokeniser** (`running_mamba/log_signature_fast.py`). Anchors are placed every `stride` samples, which
gives T' ≈ T / `stride` tokens. At each anchor and for each window length L_k, the last L_k samples are
joined into a piecewise-linear path and summarised by its log-signature truncated at depth m_k, computed
with [iisignature](https://github.com/bottler/iisignature) in the Lyndon basis. An optional global branch
adds the log-signature of the whole series up to the anchor. The pieces are concatenated into one token.
A depth-one log-signature of a window with d channels is its increment (d coordinates); depth two adds
the d(d−1)/2 Lévy areas, so EigenWorms (d = 6, one window of depth two) has 21-dimensional tokens. With
`normalize`, every token coordinate is standardised with training-set statistics. Tokenisation runs once,
on the CPU, before training.

**Backbone** (`MambaClassification` in `running_mamba/mamba_classification.py`). A linear layer embeds
each token into `hidden_dim` dimensions. It is followed by `num_blocks` residual blocks, each applying
layer normalisation, a Mamba layer (selective scan with state size `ssm_dim`, causal convolution of
width `convdim`, expansion factor `expansion`) and dropout 0.3, and by a final layer normalisation. For
classification the outputs are averaged over the tokens and a linear layer gives the class scores; for
regression (PPG-DaLiA) the linear layer is applied to every token.

**Training** (`torch_experiments/train.py`). Adam without weight decay, with cross-entropy for
classification and mean squared error for regression; up to 100,000 steps, evaluating every 1,000 steps
and stopping once the validation metric has not improved for more than 10 evaluations. The test metric
is recorded at the best validation point. Each UEA dataset is split 70:15:15 at random, the split being
fixed by the seed (2345, 3456, 4567, 5678 and 6789).

### Configurations

`experiment_configs/repeats/sig_mamba/<dataset>.json` holds the configuration of Table 11:

| Table 11 | Key | Meaning |
|---|---|---|
| LR | `lr` | learning rate |
| Hidden | `hidden_dim` | model dimension |
| State | `ssm_dim` | state size |
| Blocks | `num_blocks` | number of Mamba blocks |
| Conv | `convdim` | convolution width |
| Exp | `expansion` | expansion factor |
| Norm | `normalize` | standardise the token coordinates |
| Lengths / Depths | `signature.lengths`, `signature.depths` | window lengths L_k and truncation depths m_k |
| Stride | `signature.stride` | stride between anchors |
| Global | `signature.include_global`, `signature.d_global` | global branch and its depth |

`batch_size`, `num_steps`, `print_steps` and `early_stopping_steps` set the training loop and `seeds` the
five splits. `time` has no effect on LogSig-SSM, whose tokens are computed on the raw channels.

## Repository layout

| Path | Contents |
|---|---|
| `running_mamba/` | Log-signature, Catch24 and wavelet-scattering tokenisers; LogSig-SSM and Transformer models |
| `torch_experiments/` | LogSig-SSM training loop (PyTorch) and dataset loading |
| `running_s5/`, `models/`, `train.py` | LogSig-S5 (JAX): classifier, S5 layer and training loop |
| `run_experiment.py` | Runs the configurations in `experiment_configs/` |
| `experiment_configs/` | Reported configurations in `repeats/<model>/<dataset>.json`; depth-1 tokeniser in `ablations/depth1/` |
| `data_dir/` | Download and preprocessing of the UEA and PPG-DaLiA datasets |
| `benchmarks/` | Time and GPU-memory benchmark |
| `outputs/` | The reported runs, see [Results](#results) |
| `environments/` | Exact package versions of the two environments |

## Installation

There are two environments: PyTorch for LogSig-SSM, and JAX for LogSig-S5 and the data processing.
`environments/` lists their exact package versions on our RTX 4090 machine.

```bash
# PyTorch: LogSig-SSM
conda create -n sig_pytorch_stable python=3.10
conda activate sig_pytorch_stable
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 einops==0.8.0 packaging ninja
pip install --no-build-isolation causal-conv1d==1.5.4 mamba-ssm==2.2.6.post3 iisignature==0.24
pip install pycatch22==0.4.5 kymatio==0.3.0 jax
```

```bash
# JAX: LogSig-S5 and data processing
conda create -n sig_s5_stable python=3.10
conda activate sig_s5_stable
pip install "jax[cuda12]==0.4.38" equinox==0.12.2 optax==0.2.4 diffrax==0.7.0 signax==0.1.1 roughpy==0.2.0
pip install numpy==1.26.4 sktime==0.30.1 aeon==1.3.0 scikit-learn pandas tqdm
pip install --no-build-isolation iisignature==0.24
```

`iisignature` has no wheels and needs NumPy to build, hence `--no-build-isolation` once NumPy is
installed. The PyTorch environment needs `jax` only to read the processed datasets.

## Data

In the JAX environment:

```bash
python data_dir/download_uea.py
python data_dir/process_uea.py
python data_dir/process_ppg.py   # expects PPG_FieldStudy in data_dir/raw/PPG_FieldStudy/
```

## Running

Each command trains the five seeds of a configuration. Runs are written to
`reruns/<model>/<dataset>/<run>/` (ignored by git), or to `<output_dir>/seed_<seed>/` with `--output_dir`;
the recorded runs in `outputs/` are never overwritten. Without `--datasets`, the six UEA datasets are run.

```bash
# LogSig-SSM on a UEA dataset and on PPG-DaLiA (PyTorch environment)
python run_experiment.py --pytorch_experiments --models sig_mamba --datasets EigenWorms
python run_experiment.py --pytorch_experiments --models sig_mamba --ppg

# Catch24 / wavelet-scattering tokenisers
python run_experiment.py --pytorch_experiments --models sig_mamba --datasets EigenWorms --catch24
python run_experiment.py --pytorch_experiments --models sig_mamba --datasets EigenWorms --wst

# Depth-1 tokeniser
python run_experiment.py --pytorch_experiments --models sig_mamba --datasets EigenWorms \
    --config experiment_configs/ablations/depth1/sig_mamba/EigenWorms.json \
    --output_dir reruns/sig_mamba_depth1/EigenWorms

# LogSig-S5 (JAX environment)
python run_experiment.py --models sig_s5 --datasets EigenWorms

# Time per 1000 training steps and GPU memory (pin to one performance core)
taskset -c 2 python benchmarks/bench_sig_mamba.py
```

A run directory holds the configuration (`config.json`), the test metric at the best validation point
(`test_metric.npy`), the training and validation metrics at every evaluation (`all_train_metric.npy`,
`all_val_metric.npy`, `steps.npy`), the wall-clock time between consecutive evaluations, which covers
`print_steps` training steps together with the evaluation passes (`all_time.npy`), the peak allocated
GPU memory after the first training step (`peak_memory_mb.npy`), and the parameter count, token width
and tokenisation time (`run_meta.json`).

## Results

`python outputs/summarise_results.py` recomputes every table below from the stored files into
[`outputs/summary.md`](outputs/summary.md).

| Paper | Folder |
|---|---|
| Table 1: LogSig-SSM on UEA; Table 11: its configurations | `outputs/sig_mamba/<dataset>/` |
| Tables 2 and 5: time per 1000 training steps, GPU memory, parameters | `outputs/sig_mamba_benchmark/`, see its [README](outputs/sig_mamba_benchmark/README.md) |
| Table 3: PPG-DaLiA, Weather, PhysioNet | `outputs/sig_mamba/{ppg, Weather, physionet, physionet_no_oi}/` |
| Table 6: tokenisation time | `run_meta.json` in `outputs/sig_mamba_efficiency/` (EigenWorms, SCP1, SCP2) and `outputs/sig_mamba/` (Ethanol, Heartbeat, Motor) |
| Tables 4 and 7: ablations; Tables 12–15: their configurations | `outputs/sig_mamba_catch/`, `outputs/sig_mamba_wst/`, `outputs/sig_s5/`, `outputs/sig_transformers/` |
| Table 9: depth one against depth two | `outputs/sig_mamba_depth1/` and `outputs/sig_mamba/` |
| Table 10: search-free tokeniser rule | `outputs/search_free_rule/` |

## Citation

```bibtex
@inproceedings{oury2026logsigssm,
  title     = {{LogSig-SSM}: Time-Series Modelling with Multi-Scale Log-Signature Compression for State-Space Models},
  author    = {Oury, Felix and Calvo Peiro, Nicolas and Tanaka, Reiko J.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

The data pipeline (`data_dir/`), the JAX models (`models/`) and the training and launch scripts
(`train.py`, `run_experiment.py`) build on [log-neural-cdes](https://github.com/Benjamin-Walker/log-neural-cdes)
by Benjamin Walker, released under the MIT licence.

## Licence

MIT, see [LICENSE](LICENSE).

