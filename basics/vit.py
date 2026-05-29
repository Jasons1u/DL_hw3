"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

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
        
        # Number of patches: (H / P) * (W / P)
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use a Conv2d layer to extract patches and project them to d_model simultaneously
        # in_channels is 3 assuming standard RGB images
        self.proj = nn.Conv2d(
            in_channels=3, 
            out_channels=d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (Batch_Size, Channels, Height, Width) -> (B, 3, H, W)
        
        # Apply convolution
        x = self.proj(x)  # Shape becomes: (B, d_model, H/patch_size, W/patch_size)
        
        # Flatten the spatial dimensions (Height and Width) into a single sequence dimension
        x = x.flatten(2)  # Shape becomes: (B, d_model, num_patches)
        
        # Transpose to match transformer expectations: (Batch_Size, Sequence_Length, d_model)
        x = x.transpose(1, 2)  # Shape becomes: (B, num_patches, d_model)
        
        return x



class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add a learnable positional embedding of shape (1, num_patches+1, d_model).
      4. Pass the sequence through `num_blocks` Transformer Blocks
         (with is_decoder=False).
      5. Apply a final LayerNorm.
      6. Return only the [CLS] slice — shape (B, d_model).

    For §5 (VLM), you may want a `return_all_tokens=True` flag that returns the
    full (B, num_patches+1, d_model) sequence instead. Add it when you get there.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 768,
        num_heads: int = 12,
        num_blocks: int = 12,
        num_classes: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()

        # 1. Patch Embeddings
        # Ensure PatchEmbeddings is also imported or defined above this in your vit.py
        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        num_patches = self.patch_embed.num_patches

        # 2. CLS Token and Positional Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Using zeros as hinted in your traceback
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder (using basics.model.Block)
        self.blocks = nn.ModuleList([
            Block(d_model=d_model, num_heads=num_heads, block_size=num_patches + 1, is_decoder=False)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 4. Classification Head
        self.head = nn.Linear(d_model, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Create patch embeddings
        x = self.patch_embed(x)  # (B, num_patches, d_model)

        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, d_model)

        # 3. Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 5. Extract the CLS token's representation and apply the final LayerNorm
        x = self.norm(x)
        
        # If no classification head (CLIP pretraining), return the full sequence
        if isinstance(self.head, nn.Identity):
            return x
            
        cls_out = x[:, 0]  # Take the 0th token (CLS) across all batches: (B, d_model)

        # 6. Pass through the classification head
        out = self.head(cls_out)  # (B, num_classes)

        return out


