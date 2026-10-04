#!/usr/bin/env python3
import numpy as np
from typing import Tuple


def _catch22_1d(ts: np.ndarray) -> np.ndarray:
    """
    Compute 22 canonical time-series features using the catch22 package for a 1D series.
    Falls back gracefully if the package API differs slightly.
    Returns: (22,) float32 array.
    """
    try:
        from catch22 import catch22_all  # type: ignore
    except Exception:
        try:
            from pycatch22 import catch22_all  # type: ignore
        except Exception as e:
            raise ImportError(
                "catch22/pycatch22 is required for --catch24. Install with `pip install pycatch22` or conda-forge."
            ) from e

    res = catch22_all(np.asarray(ts, dtype=float))
    # Typical API returns a dict with 'values' (list of 22 floats)
    if isinstance(res, dict):
        if 'values' in res and isinstance(res['values'], (list, tuple, np.ndarray)):
            vals = np.asarray(res['values'], dtype=np.float32).reshape(-1)
            if vals.shape[0] != 22:
                # Some versions return an array of wrong length if features fail; coerce length
                vals = vals.astype(np.float32)
            return vals
        # Some versions may return separate keys per feature; try to preserve order by 'names'
        names = res.get('names', None)
        if names and isinstance(names, (list, tuple)):
            vals = [float(res.get(name, np.nan)) for name in names]
            return np.asarray(vals, dtype=np.float32).reshape(-1)
    # Fallback: if function returned tuple (values, names)
    if isinstance(res, (list, tuple)) and len(res) >= 1:
        vals = np.asarray(res[0], dtype=np.float32).reshape(-1)
        return vals
    # Last resort
    raise RuntimeError("Unexpected return type from catch22_all; cannot parse features")


def compute_catch24_features(X_ct: np.ndarray) -> np.ndarray:
    """
    Compute catch24 features for a multivariate time series batch.
    Input: X_ct of shape (N, C, T), float.
    Output: features of shape (N, 1, D) where D = C * 24 (22 catch22 + mean + variance per channel).
    """
    if X_ct.ndim != 3:
        raise ValueError(f"Expected X_ct with shape (N, C, T); got {X_ct.shape}")
    N, C, T = X_ct.shape
    # Preallocate output (N, 1, C * 24)
    D_per_channel = 24
    D_total = int(C * D_per_channel)
    out = np.zeros((N, 1, D_total), dtype=np.float32)

    for i in range(N):
        cursor = 0
        for c in range(C):
            ts = np.asarray(X_ct[i, c, :], dtype=np.float32)
            # Replace non-finite with local mean (then zeros if still invalid)
            if not np.all(np.isfinite(ts)):
                finite_mask = np.isfinite(ts)
                if finite_mask.any():
                    mean_val = float(np.nanmean(ts[finite_mask]))
                    ts = np.where(finite_mask, ts, mean_val).astype(np.float32)
                else:
                    ts = np.zeros_like(ts, dtype=np.float32)
            try:
                f22 = _catch22_1d(ts)
            except Exception:
                # On failure, fill with zeros to keep pipeline robust
                f22 = np.zeros((22,), dtype=np.float32)
            mean_ch = float(np.mean(ts)) if ts.size > 0 else 0.0
            var_ch = float(np.var(ts)) if ts.size > 0 else 0.0
            vec = np.concatenate([f22.astype(np.float32), np.array([mean_ch, var_ch], dtype=np.float32)], axis=0)
            out[i, 0, cursor:cursor + D_per_channel] = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            cursor += D_per_channel
    return out 


def compute_catch24_features_windowed(
    X_ct: np.ndarray,
    lengths: list,
    stride: int,
    *,
    pad_mode: str = "edge",
) -> np.ndarray:
    """
    Compute windowed catch24 features on a sliding grid, matching the signatures pipeline logic.

    Inputs:
      - X_ct: array of shape (N, C, T)
      - lengths: list of window sizes (K integers). Each window size is applied per grid position and concatenated.
      - stride: hop size between grid positions (>=1)
      - pad_mode: how to handle tail windows near the end. One of {'edge','zero','wrap'}.

    Output:
      - features of shape (N, T', D) where T' = ceil(T / stride) and D = C * 24 * K.

    Notes:
      - Each grid position s in {0, stride, 2*stride, ...} builds a feature vector by concatenating
        catch24 (22) + mean + variance for each channel and for each L in lengths.
      - For boundaries, if s+L exceeds T, we pad the segment to length L according to pad_mode.
    """
    if X_ct.ndim != 3:
        raise ValueError(f"Expected X_ct with shape (N, C, T); got {X_ct.shape}")
    if not lengths:
        raise ValueError("lengths must be a non-empty list of window sizes")
    if stride is None or int(stride) < 1:
        raise ValueError("stride must be >= 1")

    N, C, T = X_ct.shape
    K = int(len(lengths))
    lengths_arr = [int(max(1, L)) for L in lengths]
    # Grid positions (same convention as signature pipeline: ceil(T / stride) steps)
    positions = list(range(0, T, int(stride)))
    Tp = int(len(positions))

    D_per_channel = 24
    D_total = int(C * D_per_channel * K)
    out = np.zeros((N, Tp, D_total), dtype=np.float32)

    for i in range(N):
        for pi, s in enumerate(positions):
            cursor = 0
            for L in lengths_arr:
                e = int(s + L)
                for c in range(C):
                    ts_full = np.asarray(X_ct[i, c, :], dtype=np.float32)
                    # Extract segment with boundary handling
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
                                if seg.size == 0:
                                    base = ts_full if T > 0 else np.zeros((1,), dtype=np.float32)
                                else:
                                    base = seg
                                seg = np.pad(seg, (0, pad_len), mode="wrap")
                            else:  # zero
                                seg = np.pad(seg, (0, pad_len), mode="constant", constant_values=0.0)
                    # Replace non-finite
                    if not np.all(np.isfinite(seg)):
                        finite_mask = np.isfinite(seg)
                        if finite_mask.any():
                            mean_val = float(np.nanmean(seg[finite_mask]))
                            seg = np.where(finite_mask, seg, mean_val).astype(np.float32)
                        else:
                            seg = np.zeros_like(seg, dtype=np.float32)
                    try:
                        f22 = _catch22_1d(seg)
                    except Exception:
                        f22 = np.zeros((22,), dtype=np.float32)
                    mean_ch = float(np.mean(seg)) if seg.size > 0 else 0.0
                    var_ch = float(np.var(seg)) if seg.size > 0 else 0.0
                    vec = np.concatenate([f22.astype(np.float32), np.array([mean_ch, var_ch], dtype=np.float32)], axis=0)
                    out[i, pi, cursor:cursor + D_per_channel] = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    cursor += D_per_channel
    return out