"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from basics.model import Block


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)   # (B, d_model, H/P, W/P)
        x = x.flatten(2)   # (B, d_model, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, d_model)
        return x


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add positional encoding (learned, rope1d, or rope2d).
      4. Pass through `num_blocks` Transformer Blocks (is_decoder=False).
      5. Apply a final LayerNorm.
      6. Return the [CLS] embedding (B, d_model) by default.
         If return_all_tokens=True, return (B, N+1, d_model).

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
        pos_encoding: "learned" | "rope1d" | "rope2d"
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 768,
        num_heads: int = 12,
        num_blocks: int = 12,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = pos_encoding

        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        self.num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        elif pos_encoding == "rope1d":
            from basics.rope import RoPE1D
            head_dim = d_model // num_heads
            # Support up to 4× the training sequence length for extrapolation
            self.rope = RoPE1D(head_dim, max_seq_len=self.num_patches * 4 + 1)
            self.pos_embed = None
        elif pos_encoding == "rope2d":
            from basics.rope import RoPE2D
            head_dim = d_model // num_heads
            self.rope = RoPE2D(head_dim, grid_size=self.grid_size * 4)
            self.pos_embed = None
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            Block(d_model=d_model, num_heads=num_heads, block_size=self.num_patches + 1, is_decoder=False)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _apply_rope1d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D RoPE to x of shape (B, T, d_model) treating it as Q/K."""
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        head_dim = self.rope.head_dim
        num_heads = D // head_dim
        # Reshape to (B, num_heads, T, head_dim)
        x_h = x.view(B, T, num_heads, head_dim).transpose(1, 2)
        x_rot = self.rope(x_h, positions)
        return x_rot.transpose(1, 2).reshape(B, T, D)

    def _apply_rope2d(self, x: torch.Tensor, N: int) -> torch.Tensor:
        """Apply 2D RoPE to x: CLS token is position (0,0), patches get grid coords."""
        B, T, D = x.shape
        head_dim = self.rope.head_dim
        num_heads = D // head_dim

        grid = int(math.isqrt(N))
        x_coords = torch.arange(N, device=x.device) % grid
        y_coords = torch.arange(N, device=x.device) // grid

        # CLS token gets position (0, 0); patches get their grid coords
        x_c = torch.zeros(1, device=x.device, dtype=torch.long)
        y_c = torch.zeros(1, device=x.device, dtype=torch.long)

        x_coords = torch.cat([x_c, x_coords])
        y_coords = torch.cat([y_c, y_coords])

        x_h = x.view(B, T, num_heads, head_dim).transpose(1, 2)
        x_rot = self.rope(x_h, x_coords, y_coords)
        return x_rot.transpose(1, 2).reshape(B, T, D)

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)  # (B, N, d_model)
        N = x.shape[1]  # actual number of patches (may differ from self.num_patches at extrapolation)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, d_model)

        if self.pos_encoding == "learned":
            # Optionally interpolate pos_embed if sequence length differs
            if x.shape[1] != self.pos_embed.shape[1]:
                pos_embed = self._interpolate_pos_embed(x.shape[1] - 1)
            else:
                pos_embed = self.pos_embed
            x = x + pos_embed
            x = self.pos_drop(x)
        elif self.pos_encoding == "rope1d":
            x = self._apply_rope1d(x)
        elif self.pos_encoding == "rope2d":
            x = self._apply_rope2d(x, N)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_all_tokens:
            return x  # (B, N+1, d_model)
        return x[:, 0]  # (B, d_model)

    def _interpolate_pos_embed(self, new_num_patches: int) -> torch.Tensor:
        """Bilinearly interpolate patch position embeddings to a new grid size."""
        pos_embed = self.pos_embed  # (1, N+1, d_model)
        cls_pe = pos_embed[:, :1, :]  # (1, 1, d_model)
        patch_pe = pos_embed[:, 1:, :]  # (1, N, d_model)

        old_grid = int(math.isqrt(patch_pe.shape[1]))
        new_grid = int(math.isqrt(new_num_patches))

        patch_pe = patch_pe.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pe = torch.nn.functional.interpolate(
            patch_pe, size=(new_grid, new_grid), mode="bilinear", align_corners=False
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
        return torch.cat([cls_pe, patch_pe], dim=1)
