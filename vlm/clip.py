"""CLIP-style contrastive learning — §3.

You implement: clip_loss, ProjectionHeads.

The frozen text encoder is provided in `basics.text_encoder.FrozenTextEncoder`.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHeads(nn.Module):
    def __init__(self, d_image: int, d_text: int, d_proj: int = 256) -> None:
        super().__init__()
        self.image_proj = nn.Linear(d_image, d_proj, bias=False)
        self.text_proj = nn.Linear(d_text, d_proj, bias=False)

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_proj = F.normalize(self.image_proj(image_embeds), p=2, dim=-1)
        text_proj = F.normalize(self.text_proj(text_embeds), p=2, dim=-1)
        return image_proj, text_proj

def init_logit_scale() -> nn.Parameter:
    return nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

def clip_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    scale = torch.exp(logit_scale)
    logits = (image_embeds @ text_embeds.T) * scale
    B = image_embeds.size(0)
    labels = torch.arange(B, device=image_embeds.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0
