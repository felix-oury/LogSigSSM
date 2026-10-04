"""
log_signature_fast.py
Log signature computation with iisignature.
Based on signature_fast.py but uses log signatures instead of signatures.
Keeps the same public API: compute_windowed_log_signatures, batch_compute_log_features.
"""

from __future__ import annotations
import functools
from typing import List, Tuple, Dict, Optional
import numpy as np
import iisignature


def n_logsig_dim(m: int, D: int) -> int:
    """
    Calculate the dimension of the log signature up to depth D for m channels.
    For log signatures, the dimension is different from the signature.
    """
    if D <= 0:
        return 0
    # Log signature dimension: use iisignature's built-in function
    return iisignature.logsiglength(m, D)


def stride_grid(T: int, s: int) -> np.ndarray:
    return np.arange(0, T, s, dtype=int)


@functools.lru_cache(maxsize=None)
def _prepared(C: int, depth: int) -> object:
    """Cached iisignature.prepare.

    The prepared basis depends only on (C, depth), but iisignature does not
    memoise it: repeated prepare(3, 1) calls cost full price every time. It is
    also surprisingly expensive at low channel counts -- a flat ~90 ms for
    C <= 8 regardless of depth, dropping to ~0 ms for C >= 12, presumably
    because an optimised basis is only built while the free Lie algebra is
    small enough to be worth constructing.

    Since compute_windowed_log_signatures is called once per time series, an
    uncached prepare is paid once per sample rather than once per configuration.
    On low-channel datasets that overhead dominated everything else: 524 samples
    x 85 ms = 44.8 s of the 44.8 s measured for EthanolConcentration, whose real
    tokenisation cost is ~0.65 s.
    """
    return iisignature.prepare(C, depth)


def prepare_keys_by_depth(C: int, depths: List[int]) -> Dict[int, object]:
    """iisignature.prepare for each unique depth, cached across calls."""
    return {d: _prepared(C, d) for d in sorted(set([d for d in depths if d > 0]))}


def _logsig_vec(path_CT: np.ndarray, depth: int, keys: Dict[int, object]) -> np.ndarray:
    """
    Compute log signature for a single path using prepared keys.
    Uses iisignature.logsig instead of iisignature.sig.
    """
    if depth <= 0:
        return np.zeros((0,), dtype=np.float32)
    # iisignature expects shape (L, C)
    v = iisignature.logsig(path_CT.T, keys[depth]).astype(np.float32)
    if not np.all(np.isfinite(v)):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return v


def compute_windowed_log_signatures(
    P: np.ndarray,                  # (C, T)
    lengths: List[int],             # [L_k]
    depths: List[int],              # [d_k]
    stride: int,
    include_global: bool = False,
    global_depth: Optional[int] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute windowed log signatures for a single time series.

    Args:
        P: Time series of shape (C, T) where C is channels and T is time steps
        lengths: List of window lengths for each branch
        depths: List of log signature depths for each branch
        stride: Stride for sampling time points
        include_global: Whether to include global log signature
        global_depth: Depth for global log signature

    Returns:
        feats: Log signature features of shape (T_prime, D_total)
        T_prime: Number of sampled time points
        D_total: Total feature dimension
    """
    C, T = P.shape
    assert len(lengths) == len(depths)
    t_grid = stride_grid(T, int(stride))
    T_prime = len(t_grid)

    # Prepare keys (depths + optional global)
    all_depths = list(depths)
    if include_global and (global_depth is not None):
        all_depths.append(global_depth)
    keys = prepare_keys_by_depth(C, all_depths)

    # Calculate dimensions for log signature branches
    dims = [n_logsig_dim(C, d) for d in depths]
    D_total = int(sum(dims))
    d_dim_global = 0
    if include_global and global_depth and global_depth > 0:
        d_dim_global = n_logsig_dim(C, int(global_depth))
        D_total += d_dim_global

    feats = np.zeros((T_prime, D_total), dtype=np.float32)

    for idx_tau, t_tau in enumerate(t_grid):
        cursor = 0
        # Branches
        for Lk, dk, d_dim in zip(lengths, depths, dims):
            if dk <= 0 or d_dim == 0:
                continue
            start = max(0, int(t_tau) - int(Lk) + 1)
            seg = P[:, start : int(t_tau) + 1]  # (C, L_eff)
            v = _logsig_vec(seg, dk, keys)
            feats[idx_tau, cursor : cursor + d_dim] = v
            cursor += d_dim

        # Global
        if include_global and global_depth and global_depth > 0:
            seg_g = P[:, : int(t_tau) + 1]
            v_g = _logsig_vec(seg_g, global_depth, keys)
            feats[idx_tau, cursor : cursor + d_dim_global] = v_g
            cursor += d_dim_global

    return feats, T_prime, D_total


def batch_compute_log_features(
    X: np.ndarray,                  # (N, C, T)
    lengths: List[int],
    depths: List[int],
    stride: int,
    include_global: bool = False,
    global_depth: Optional[int] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute log signature features for a batch of time series.

    Args:
        X: Batch of time series of shape (N, C, T)
        lengths: List of window lengths for each branch
        depths: List of log signature depths for each branch
        stride: Stride for sampling time points
        include_global: Whether to include global log signature
        global_depth: Depth for global log signature

    Returns:
        out: Log signature features of shape (N, T_prime, D_total)
        T_prime: Number of sampled time points
        D_total: Total feature dimension
    """
    N, C, T = X.shape
    feats0, T_prime, D_total = compute_windowed_log_signatures(
        X[0], lengths, depths, stride, include_global, global_depth
    )
    out = np.zeros((N, T_prime, D_total), dtype=np.float32)
    out[0] = feats0
    for n in range(1, N):
        feats_n, Tp_n, Dn = compute_windowed_log_signatures(
            X[n], lengths, depths, stride, include_global, global_depth
        )
        assert Tp_n == T_prime and Dn == D_total, "Inconsistent T' or D across samples."
        out[n] = feats_n
    return out, T_prime, D_total
