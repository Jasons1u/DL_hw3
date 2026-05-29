"""Vision-Language Projector — §5.

You implement: VisionLanguageProjector.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class VisionLanguageProjector(nn.Module):
    def __init__(self, d_image: int, d_decoder: int, expansion: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_image, expansion * d_image),
            nn.GELU(),
            nn.Linear(expansion * d_image, d_decoder)
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim == 2:
            image_features = image_features.unsqueeze(1)
        return self.net(image_features)
