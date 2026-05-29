"""LoRA adapters — §4.

You implement: LoRALinear, apply_lora_to_attention.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.base_layer = base_layer

        for param in self.base_layer.parameters():
            param.requires_grad = False

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.A = nn.Parameter(torch.empty(rank, base_layer.in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base_layer.out_features, rank, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.scaling * (x @ self.A.T @ self.B.T)

def apply_lora_to_attention(model: nn.Module, rank: int, alpha: float) -> nn.Module:
    from basics.model import Head
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, Head):
            if hasattr(module, 'q_proj') and isinstance(module.q_proj, nn.Linear):
                module.q_proj = LoRALinear(module.q_proj, rank, alpha)
            if hasattr(module, 'v_proj') and isinstance(module.v_proj, nn.Linear):
                module.v_proj = LoRALinear(module.v_proj, rank, alpha)
    return model
