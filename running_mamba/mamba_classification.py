#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import math
from einops import rearrange, repeat


# Robust import of selective_scan from mamba_ssm for CUDA acceleration
_CURR_DIR = os.path.dirname(__file__)
_MAMBA_DIR = os.path.abspath(os.path.join(_CURR_DIR, "..", ".github_imports", "mamba"))
if _MAMBA_DIR not in sys.path:
    sys.path.insert(0, _MAMBA_DIR)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _sel_scan
    _HAS_SEL_SCAN = True
except Exception:
    _HAS_SEL_SCAN = False
    try:
        import warnings as _warnings
        _warnings.warn("Mamba selective_scan CUDA op not found; falling back to slow Python scan.")
    except Exception:
        pass

# For backward compatibility: check if old Mamba import exists
_HAS_MAMBA = _HAS_SEL_SCAN


class MambaLayer(nn.Module):
    """
    Mamba SSM Layer with proper A, B, C, D initialization.
    This is a custom implementation that properly initializes the SSM matrices.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # Always initialize on CPU, then move to device later with .to(device)
        # This avoids CUDA context issues with multiprocessing spawn
        factory_kwargs = {"device": None, "dtype": dtype}
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.d_model * self.expand
        
        self.dt_rank = (self.d_model + 15) // 16 if dt_rank == "auto" else int(dt_rank)
        
        # Input projection: projects to expanded dimension
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False, **factory_kwargs)
        
        # Convolution over sequence (causal)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            **factory_kwargs
        )
        
        # SSM projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        
        # Initialize dt projection
        dt_init_std = (self.dt_rank ** -0.5) * float(dt_scale)
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"dt_init {dt_init} not supported")
        
        # Initialize dt bias so softplus(dt_bias) is in [dt_min, dt_max]
        # Use CPU initialization to avoid CUDA multiprocessing issues
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # S4D-style real initialization for A (same as S6)
        # Initialize on CPU to avoid CUDA multiprocessing issues
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))  # keep in fp32
        self.A_log._no_weight_decay = True
        
        # D "skip" parameter
        # Initialize on CPU to avoid CUDA multiprocessing issues
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, L, D)
        Returns: (B, L, D)
        """
        batch, seqlen, dim = hidden_states.shape
        
        # Input projection and split for gating
        xz = self.in_proj(hidden_states)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)
        
        # Causal convolution
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seqlen]  # truncate to original length for causality
        x = rearrange(x, "b d l -> b l d")
        
        # SiLU activation
        x = torch.nn.functional.silu(x)
        
        # SSM computation
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Compute dt, B, C from input
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # dt projection
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        
        # Rearrange for selective_scan
        x_ssm = rearrange(x, "b l d -> b d l")
        dt = rearrange(dt, "b l d -> b d l")
        B = rearrange(B, "b l n -> b n l").contiguous()
        C = rearrange(C, "b l n -> b n l").contiguous()
        
        # Use CUDA selective_scan only if available AND tensors are on CUDA
        if _HAS_SEL_SCAN and hidden_states.is_cuda:
            y = _sel_scan(
                x_ssm,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            y = rearrange(y, "b d l -> b l d")
        else:
            # Fallback: naive PyTorch scan
            device = hidden_states.device
            N = self.d_state
            D_skip = self.D.float()
            bias = self.dt_proj.bias.float().to(device)
            
            s = torch.zeros(batch, self.d_inner, N, device=device, dtype=torch.float32)
            ys = []
            for t in range(seqlen):
                x_t = x_ssm[:, :, t].float()
                B_t = B[:, :, t].float()
                C_t = C[:, :, t].float()
                delta_t = torch.nn.functional.softplus(dt[:, :, t] + bias.view(1, -1))
                
                # Discretize: A_bar = exp(delta * A)
                A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
                
                # State update: s = A_bar * s + B * x
                b_term = torch.einsum('bd,bn->bdn', x_t, B_t)
                s = A_bar * s + b_term
                
                # Output: y = C * s + D * x
                y_t = torch.einsum('bdn,bn->bd', s, C_t) + D_skip.view(1, -1) * x_t
                ys.append(y_t.unsqueeze(-1))
            
            y = torch.cat(ys, dim=-1)
            y = rearrange(y, "b d l -> b l d")
        
        # Gating with z
        y = y * torch.nn.functional.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


class MambaBlock(nn.Module):
    """Mamba block with normalization, SSM, and residual connection."""
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaLayer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(p=float(dropout))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual block
        y = self.norm(x)
        y = self.mamba(y)
        y = self.dropout(y)
        return x + y


class MambaClassification(nn.Module):
    """
    Mamba-based classification/regression model.
    Uses proper SSM initialization with A, B, C, D matrices.
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        regression: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(int(input_size), int(d_model))
        
        # Use our custom MambaBlock with proper initialization
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=int(d_model),
                d_state=int(d_state),
                d_conv=int(d_conv),
                expand=int(expand),
                dropout=float(dropout),
            )
            for _ in range(int(n_layers))
        ])
        
        self.norm = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(p=float(dropout))
        self.out = nn.Linear(int(d_model), int(num_classes))
        self.regression = bool(regression)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, L, input_size)
        Returns:
            For classification: (B, num_classes)
            For regression: (B, L, num_classes)
        """
        x = x.float()
        x = self.input_proj(x)
        
        # Pass through Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        if self.regression:
            # For regression, keep sequence dimension
            x = self.dropout(x)
            return self.out(x)
        else:
            # For classification, pool over sequence
            x = x.mean(dim=1)
            x = self.dropout(x)
            return self.out(x)


