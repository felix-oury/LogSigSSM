"""
This module provides a function to generate an S5 model.

Function:
- `create_model`: Generates and returns an S5 model instance along with its Equinox state.

Parameters for `create_model`:
- `data_dim`: The input data dimension.
- `label_dim`: The output label dimension.
- `hidden_dim`: The hidden state dimension for the model.
- `num_blocks`: The number of blocks (layers) in the S5 model.
- `classification`: A boolean indicating whether the task is classification (True) or regression (False).
- `output_step`: The step interval for outputting predictions in sequence models.
- `ssm_dim`: The state-space model dimension for S5.
- `ssm_blocks`: The number of SSM blocks in S5.
- `s5_init`, `s5_bidirectional`, `s5_conj_sym`, `s5_discretization`, `s5_dt_min`, `s5_dt_max`,
  `s5_step_rescale`: S5-specific hyperparameters.
- `key`: A JAX PRNG key for random number generation.

Returns:
- A tuple `(model, state)` where `state` is an `eqx.nn.State` for the model.

Raises:
- `ValueError`: If required hyperparameters are not provided.
"""

import equinox as eqx

from models.S5 import S5


def create_model(
    data_dim,
    label_dim,
    hidden_dim,
    num_blocks,
    classification=True,
    output_step=1,
    ssm_dim=None,
    ssm_blocks=None,
    s5_init="lecun_normal",
    s5_bidirectional=True,
    s5_conj_sym=True,
    s5_discretization="zoh",
    s5_dt_min=0.001,
    s5_dt_max=0.1,
    s5_step_rescale=1.0,
    *,
    key,
):
    if ssm_dim is None:
        raise ValueError("Must specify ssm_dim for S5.")
    if ssm_blocks is None:
        raise ValueError("Must specify ssm_blocks for S5.")

    ssm = S5(
        num_blocks,
        data_dim,
        ssm_dim,
        ssm_blocks,
        hidden_dim,
        label_dim,
        classification,
        output_step,
        s5_init,
        s5_bidirectional,
        s5_conj_sym,
        s5_discretization,
        s5_dt_min,
        s5_dt_max,
        s5_step_rescale,
        key=key,
    )
    state = eqx.nn.State(ssm)
    return ssm, state