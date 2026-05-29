"""Vision-Language Model — §5.

You implement: VisionLanguageModel.
"""
from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
import vlm.masking

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]

class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        if injection == "cls":
            visual_feats = self.vit(images) 
            if visual_feats.ndim == 2: visual_feats = visual_feats.unsqueeze(1)
        else:
            visual_feats = self.vit(images, return_all_tokens=True)
            
        visual_embeds = self.projector(visual_feats)
        text_embeds = self.decoder.get_input_embeddings()(input_ids)

        if injection in ["cls", "all_patches"]:
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            B, V, _ = visual_embeds.shape
            visual_mask = torch.ones(B, V, dtype=attention_mask.dtype, device=attention_mask.device)
            full_attention_mask = torch.cat([visual_mask, attention_mask], dim=1)
            if labels is not None:
                visual_labels = torch.full((B, V), -100, dtype=labels.dtype, device=labels.device)
                full_labels = torch.cat([visual_labels, labels], dim=1)
            else:
                full_labels = None
        else:
            # Interleaved
            inputs_embeds = text_embeds.clone()
            for b in range(inputs_embeds.size(0)):
                img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
                if len(img_pos) > 0:
                    inputs_embeds[b, img_pos[0]:img_pos[0]+visual_embeds.size(1)] = visual_embeds[b]
            full_attention_mask = attention_mask
            full_labels = labels

        if mask_mode == "image_bidir" and injection != "interleaved":
            full_attention_mask = vlm.masking.build_image_bidir_mask(
                visual_embeds.size(1), text_embeds.size(1), dtype=inputs_embeds.dtype, device=inputs_embeds.device
            ).unsqueeze(0).expand(inputs_embeds.size(0), -1, -1, -1)

        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask if mask_mode != "image_bidir" else None,
            labels=full_labels
        )
        return {"loss": out.loss, "logits": out.logits}

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        # Simplified inference generation method
        self.eval()
        pass
