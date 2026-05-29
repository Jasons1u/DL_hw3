"""Rotary Position Embeddings — §6.

You implement: RoPE1D, RoPE2D.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class RoPE1D(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        if cos.dim() == 2: # (T, D)
            cos = cos.unsqueeze(0).unsqueeze(0) # (1, 1, T, D)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3: # (B, T, D)
            cos = cos.unsqueeze(1) # (B, 1, T, D)
            sin = sin.unsqueeze(1)
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x1 * cos - x2 * sin
        x_out[..., 1::2] = x2 * cos + x1 * sin
        return x_out

class RoPE2D(nn.Module):
    def __init__(self, head_dim: int, grid_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.base = base

        inv_freq = base ** (-torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))
        t = torch.arange(grid_size).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, x_coords: torch.Tensor, y_coords: torch.Tensor) -> torch.Tensor:
        half_dim = self.head_dim // 2
        x_part = x[..., :half_dim]
        y_part = x[..., half_dim:]
        
        cos_x = self.cos_cached[x_coords]
        sin_x = self.sin_cached[x_coords]
        cos_y = self.cos_cached[y_coords]
        sin_y = self.sin_cached[y_coords]
        if cos_x.dim() == 2: # (T, D)
            cos_x = cos_x.unsqueeze(0).unsqueeze(0)
            sin_x = sin_x.unsqueeze(0).unsqueeze(0)
            cos_y = cos_y.unsqueeze(0).unsqueeze(0)
            sin_y = sin_y.unsqueeze(0).unsqueeze(0)
        elif cos_x.dim() == 3: # (B, T, D)
            cos_x = cos_x.unsqueeze(1)
            sin_x = sin_x.unsqueeze(1)
            cos_y = cos_y.unsqueeze(1)
            sin_y = sin_y.unsqueeze(1)
        
        x1_x = x_part[..., 0::2]
        x2_x = x_part[..., 1::2]
        x_out_part = torch.empty_like(x_part)
        x_out_part[..., 0::2] = x1_x * cos_x - x2_x * sin_x
        x_out_part[..., 1::2] = x2_x * cos_x + x1_x * sin_x
        
        x1_y = y_part[..., 0::2]
        x2_y = y_part[..., 1::2]
        y_out_part = torch.empty_like(y_part)
        y_out_part[..., 0::2] = x1_y * cos_y - x2_y * sin_y
        y_out_part[..., 1::2] = x2_y * cos_y + x1_y * sin_y
        
        return torch.cat([x_out_part, y_out_part], dim=-1)
