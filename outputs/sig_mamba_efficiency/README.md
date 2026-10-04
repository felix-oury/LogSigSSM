# Logging runs for EigenWorms, SCP1 and SCP2

Re-runs of the reported LogSig-SSM configurations of EigenWorms, SCP1 and SCP2 (same configuration and
seeds as `outputs/sig_mamba/`), made because the reported runs of these three datasets predate the logging
of `run_meta.json` (tokenisation time, token width, parameter count).

- Table 6 takes the tokenisation times of these three datasets from here.
- The reported accuracies are those of `outputs/sig_mamba/`.
- The time and GPU memory of Tables 2 and 5 come from `outputs/sig_mamba_benchmark/`; `all_time.npy` and
  `peak_memory_mb.npy` here follow the training-run conventions described in the main README.
