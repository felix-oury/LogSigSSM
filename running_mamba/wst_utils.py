#!/usr/bin/env python3
import os
import numpy as np
from typing import List, Optional


def _get_scattering1d(L_max: int):
    try:
        # Lazy import to avoid mandatory dependency when not using --wst
        from kymatio.torch import Scattering1D  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        raise ImportError(
            "kymatio and torch are required for --wst. Please `pip install kymatio torch`."
        ) from e

    # Choose J relative to L_max; keep small to limit feature size
    # Ensure at least a minimal scale; allow Q to be configured via env
    J = max(2, int(np.floor(np.log2(max(8, int(L_max))))) - 2)
    try:
        q_env = int(os.environ.get("WST_Q", "4"))
        Q = int(max(1, min(8, q_env)))
    except Exception:
        Q = 4

    # Force CPU for WST when JAX is present to avoid CUDA conflicts
    import sys
    use_cpu = 'jax' in sys.modules
    device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        Sx = Scattering1D(J=J, shape=int(L_max), Q=int(Q)).to(device)
    except TypeError:
        # Older kymatio may not accept Q; fall back to default
        Sx = Scattering1D(J=J, shape=int(L_max)).to(device)
    return Sx, device


def _scat_mean_coeffs(Sx, device, x_1d_np: np.ndarray, pad_to: int) -> np.ndarray:
    import torch  # type: ignore
    # Pad/truncate to pad_to
    x = np.asarray(x_1d_np, dtype=np.float32)
    if x.shape[0] < int(pad_to):
        x = np.pad(x, (0, int(pad_to) - x.shape[0]), mode="constant", constant_values=0.0)
    elif x.shape[0] > int(pad_to):
        x = x[: int(pad_to)]
    xt = torch.from_numpy(x[None, :]).to(device)  # (1, T)
    with torch.no_grad():
        S = Sx(xt)  # (1, C_scat, T_out)
    # Average over time dimension to get fixed-length vector
    if S.dim() == 3:
        vec = S.mean(dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    elif S.dim() == 2:
        vec = S.squeeze(0).detach().cpu().numpy().astype(np.float32)
    else:
        vec = S.detach().cpu().numpy().astype(np.float32).reshape(-1)
    # Replace non-finite
    if not np.all(np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return vec


def compute_wst_features_windowed(
    X_ct: np.ndarray,
    lengths: List[int],
    stride: int,
    *,
    pad_mode: str = "zero",
) -> np.ndarray:
    """
    Compute windowed Wavelet Scattering Transform (WST) features using Kymatio,
    mirroring the catch24 windowing API.

    Inputs:
      - X_ct: array (N, C, T)
      - lengths: list of window sizes (K)
      - stride: hop size (>=1)
      - pad_mode: kept for API symmetry; segments are zero-padded before WST

    Output:
      - features (N, T', D) where T' = ceil(T/stride) and
        D = C * C_scat * K, with C_scat determined by (J, shape=max(lengths)).
    """
    if X_ct.ndim != 3:
        raise ValueError(f"Expected X_ct with shape (N, C, T); got {X_ct.shape}")
    if not lengths:
        raise ValueError("lengths must be non-empty")
    if stride is None or int(stride) < 1:
        raise ValueError("stride must be >= 1")

    N, C, T = X_ct.shape
    K = int(len(lengths))
    lengths_arr = [int(max(1, L)) for L in lengths]
    positions = list(range(0, T, int(stride)))
    Tp = int(len(positions))

    L_max = int(max(lengths_arr))
    Sx, device = _get_scattering1d(L_max)

    # Probe coeff dimension once
    try:
        probe = _scat_mean_coeffs(Sx, device, np.zeros((L_max,), dtype=np.float32), L_max)
        D_per_channel = int(probe.shape[0])
    except Exception:
        # Fallback safe dim
        D_per_channel = 64

    D_total = int(C * D_per_channel * K)
    out = np.zeros((N, Tp, D_total), dtype=np.float32)

    # Batch across samples, positions, lengths and channels for speed
    try:
        bs_env = int(os.environ.get("WST_BATCH", "512"))
        BATCH = int(max(1, bs_env))
    except Exception:
        BATCH = 512

    batch_segments: List[np.ndarray] = []
    batch_i: List[int] = []
    batch_pi: List[int] = []
    batch_off: List[int] = []

    def _flush_batch():
        import torch  # type: ignore
        if not batch_segments:
            return
        xnp = np.stack(batch_segments).astype(np.float32)
        xt = torch.from_numpy(xnp).to(device)
        with torch.no_grad():
            S = Sx(xt)
        if S.dim() == 3:
            V = S.mean(dim=-1).detach().cpu().numpy().astype(np.float32)
        elif S.dim() == 2:
            V = S.detach().cpu().numpy().astype(np.float32)
        else:
            V = S.detach().cpu().numpy().astype(np.float32).reshape(S.shape[0], -1)
        for j in range(V.shape[0]):
            ii = batch_i[j]
            pp = batch_pi[j]
            off = batch_off[j]
            vj = V[j]
            out[ii, pp, off: off + D_per_channel] = vj[:D_per_channel]
        batch_segments.clear()
        batch_i.clear()
        batch_pi.clear()
        batch_off.clear()

    for i in range(N):
        for pi, s in enumerate(positions):
            for l_idx, L in enumerate(lengths_arr):
                e = int(s + L)
                for c in range(C):
                    ts_full = np.asarray(X_ct[i, c, :], dtype=np.float32)
                    if e <= T:
                        seg = ts_full[s:e]
                    else:
                        seg = ts_full[s:T]
                        if seg.size < L:
                            pad_len = int(L - seg.size)
                            if pad_mode == "edge":
                                if seg.size == 0:
                                    fill_val = float(np.nanmean(ts_full)) if np.isfinite(ts_full).any() else 0.0
                                    seg = np.full((L,), fill_val, dtype=np.float32)
                                else:
                                    seg = np.pad(seg, (0, pad_len), mode="edge")
                            elif pad_mode == "wrap":
                                base = seg if seg.size > 0 else (ts_full if T > 0 else np.zeros((1,), dtype=np.float32))
                                seg = np.pad(base, (0, pad_len), mode="wrap")
                            else:
                                seg = np.pad(seg, (0, pad_len), mode="constant", constant_values=0.0)
                    if not np.all(np.isfinite(seg)):
                        finite_mask = np.isfinite(seg)
                        if finite_mask.any():
                            mean_val = float(np.nanmean(seg[finite_mask]))
                            seg = np.where(finite_mask, seg, mean_val).astype(np.float32)
                        else:
                            seg = np.zeros_like(seg, dtype=np.float32)
                    # pad/truncate to L_max for Sx
                    if seg.shape[0] < L_max:
                        seg = np.pad(seg, (0, int(L_max - seg.shape[0])), mode="constant", constant_values=0.0)
                    elif seg.shape[0] > L_max:
                        seg = seg[:L_max]
                    batch_segments.append(seg)
                    batch_i.append(i)
                    batch_pi.append(pi)
                    d_off = int((l_idx * C + c) * D_per_channel)
                    batch_off.append(d_off)
                    if len(batch_segments) >= BATCH:
                        _flush_batch()
        # flush at end of positions loop to reduce memory
        _flush_batch()
    # final flush
    _flush_batch()

    # CRITICAL: Explicitly delete Scattering1D object and clear CUDA cache
    # This prevents memory accumulation between seeds that can cause segmentation faults
    try:
        import torch
        import gc
        del Sx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception:
        pass

    return out


