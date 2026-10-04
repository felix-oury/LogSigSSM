"""
This module contains a function for generating the coefficients for a Hermite cubic spline with backwards differences.
"""

import jax
import jax.numpy as jnp


def calc_coeffs(data, include_time, T):
    if include_time:
        ts = data[:, :, 0]
    else:
        ts = (T / data.shape[1]) * jnp.repeat(
            jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
        )
    # Lazy import to avoid requiring diffrax unless coefficient generation is needed
    import diffrax  # type: ignore
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, data)
    return coeffs
