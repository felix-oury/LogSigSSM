#!/usr/bin/env python3
"""
Transformer encoder backbone with the same I/O contract as `MambaClassification`:
(B, L, input_size) -> classification: (B, num_classes); regression: (B, L, num_classes).

Uses causal self-attention to mirror autoregressive SSM stacking (signals are observed in full;
mask still biases the representation toward causal structure used in Mamba).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerClassification(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        d_model: int = 128,
        n_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        regression: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()
        d_model = int(d_model)
        nhead = int(nhead)
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_proj = nn.Linear(int(input_size), d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation=str(activation),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=float(dropout))
        self.out = nn.Linear(d_model, int(num_classes))
        self.regression = bool(regression)
        self.d_model = d_model

        # Cached causal attention bias per length (-inf upper triangle).
        self._mask_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _causal_attn_bias(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (L, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = torch.triu(
                torch.full((L, L), float("-inf"), dtype=torch.float32), diagonal=1
            )
        return self._mask_cache[key].to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, input_size)
        Returns:
            classification: (B, num_classes)
            regression: (B, L, num_classes)
        """
        x = x.float()
        x = self.input_proj(x)
        b, seq_len, _ = x.shape
        attn_mask = self._causal_attn_bias(seq_len, x.device, dtype=x.dtype)
        x = self.encoder(x, mask=attn_mask)
        x = self.norm(x)

        if self.regression:
            x = self.dropout(x)
            return self.out(x)
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.out(pooled)
