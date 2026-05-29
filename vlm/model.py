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
        vit_dtype = next(self.vit.parameters()).dtype
        vit_images = images.to(vit_dtype)
        if injection == "cls":
            visual_feats = self.vit(vit_images)
            if visual_feats.ndim == 2: visual_feats = visual_feats.unsqueeze(1)
        else:
            visual_feats = self.vit(vit_images, return_all_tokens=True)
            
        dec_dtype = next(self.decoder.parameters()).dtype
        visual_embeds = self.projector(visual_feats).to(dec_dtype)
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
            # Interleaved: expand <image> placeholder into the full visual token sequence.
            # Split text at the <image> position; concatenate before + visual + after.
            B = text_embeds.size(0)
            V = visual_embeds.size(1)
            all_embeds, all_masks, all_labels = [], [], []
            for b in range(B):
                if self.image_token_id is not None:
                    img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
                else:
                    img_pos = torch.tensor([], dtype=torch.long)
                if len(img_pos) > 0:
                    pos = img_pos[0].item()
                    before = text_embeds[b, :pos]           # tokens before <image>
                    after  = text_embeds[b, pos+1:]         # tokens after <image>
                    seq    = torch.cat([before, visual_embeds[b], after], dim=0)
                    am_bef = attention_mask[b, :pos]
                    am_aft = attention_mask[b, pos+1:]
                    vis_am = torch.ones(V, dtype=attention_mask.dtype, device=attention_mask.device)
                    am     = torch.cat([am_bef, vis_am, am_aft], dim=0)
                    if labels is not None:
                        lb_bef = labels[b, :pos]
                        lb_aft = labels[b, pos+1:]
                        vis_lb = torch.full((V,), -100, dtype=labels.dtype, device=labels.device)
                        lb     = torch.cat([lb_bef, vis_lb, lb_aft], dim=0)
                    else:
                        lb = None
                else:
                    # Fallback: prepend visual tokens
                    seq = torch.cat([visual_embeds[b], text_embeds[b]], dim=0)
                    vis_am = torch.ones(V, dtype=attention_mask.dtype, device=attention_mask.device)
                    am  = torch.cat([vis_am, attention_mask[b]], dim=0)
                    if labels is not None:
                        vis_lb = torch.full((V,), -100, dtype=labels.dtype, device=labels.device)
                        lb = torch.cat([vis_lb, labels[b]], dim=0)
                    else:
                        lb = None
                all_embeds.append(seq)
                all_masks.append(am)
                if lb is not None:
                    all_labels.append(lb)
            # Pad to same length
            max_len = max(e.size(0) for e in all_embeds)
            d = text_embeds.size(-1)
            inputs_embeds = torch.zeros(B, max_len, d, dtype=dec_dtype, device=text_embeds.device)
            full_attention_mask = torch.zeros(B, max_len, dtype=attention_mask.dtype, device=attention_mask.device)
            for b in range(B):
                L = all_embeds[b].size(0)
                inputs_embeds[b, :L] = all_embeds[b]
                full_attention_mask[b, :L] = all_masks[b]
            if all_labels:
                full_labels = torch.full((B, max_len), -100, dtype=labels.dtype, device=labels.device)
                for b in range(B):
                    L = all_labels[b].size(0)
                    full_labels[b, :L] = all_labels[b]
            else:
                full_labels = None

        if mask_mode == "image_bidir" and injection != "interleaved":
            full_attention_mask = vlm.masking.build_image_bidir_mask(
                visual_embeds.size(1), text_embeds.size(1), dtype=inputs_embeds.dtype, device=inputs_embeds.device
            ).expand(inputs_embeds.size(0), -1, -1, -1)

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
