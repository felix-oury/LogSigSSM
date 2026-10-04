#!/usr/bin/env python3
"""
Minimal S5 model builders for sig_s5 pipeline.
Only contains build_s5_classifier and build_s5_regressor functions.
"""
import os
import sys
from typing import Optional

# JAX / Equinox
import jax
import jax.random as jr
import equinox as eqx

# S5 model (JAX/Equinox)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from models.S5 import S5 as JaxS5

try:
    from aeon.datasets import load_classification
except Exception:
    raise ImportError("aeon is required. Please `pip install aeon`.")


def build_s5_classifier(input_dim: int, num_classes: int, *,
                        hidden_dim: int, ssm_size: int, num_blocks: int,
                        ssm_blocks: int = 1, output_step: int = 1,
                        key: Optional[jax.Array] = None) -> JaxS5:
    """Build S5 classifier model for sig_s5."""
    if key is None:
        key = jr.PRNGKey(0)
    model = JaxS5(
        num_blocks=int(num_blocks),
        N=int(input_dim),
        ssm_size=int(ssm_size),
        ssm_blocks=int(ssm_blocks),
        H=int(hidden_dim),
        output_dim=int(num_classes),
        classification=True,
        output_step=int(output_step),
        C_init="lecun_normal",
        conj_sym=False,
        clip_eigs=False,
        discretisation="bilinear",
        dt_min=1e-3,
        dt_max=1e-1,
        step_rescale=1.0,
        key=key,
    )
    # Replace BatchNorm with no-op norm for single-device
    try:
        class _NoOpNorm(eqx.Module):
            def __call__(self, x, state):
                return x, state
        new_blocks = []
        for b in model.blocks:
            b = eqx.tree_at(lambda bb: bb.norm, b, _NoOpNorm()) 
            b = eqx.tree_at(lambda bb: bb.drop, b, eqx.nn.Dropout(p=0))  # Dropout set to 0 for sig_s5
            new_blocks.append(b)
        model = eqx.tree_at(lambda m: m.blocks, model, tuple(new_blocks))
    except Exception:
        pass
    return model


def build_s5_regressor(input_dim: int, *, hidden_dim: int, ssm_size: int, num_blocks: int, 
                       ssm_blocks: int = 1, output_step: int = 1, key: Optional[jax.Array] = None) -> JaxS5:
    """Build S5 regressor model for sig_s5."""
    if key is None:
        key = jr.PRNGKey(0)
    model = JaxS5(
        num_blocks=int(num_blocks),
        N=int(input_dim),
        ssm_size=int(ssm_size),
        ssm_blocks=int(ssm_blocks),
        H=int(hidden_dim),
        output_dim=1,
        classification=False,
        output_step=int(output_step),
        C_init="lecun_normal",
        conj_sym=False,
        clip_eigs=False,
        discretisation="bilinear",
        dt_min=1e-3,
        dt_max=1e-1,
        step_rescale=1.0,
        key=key,
    )
    try:
        class _NoOpNorm(eqx.Module):
            def __call__(self, x, state):
                return x, state
        new_blocks = []
        for b in model.blocks:
            b = eqx.tree_at(lambda bb: bb.norm, b, _NoOpNorm())
            b = eqx.tree_at(lambda bb: bb.drop, b, eqx.nn.Dropout(p=0))
            new_blocks.append(b)
        model = eqx.tree_at(lambda m: m.blocks, model, tuple(new_blocks))
    except Exception:
        pass
    return model
