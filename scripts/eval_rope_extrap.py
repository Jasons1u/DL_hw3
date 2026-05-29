"""§6 — RoPE length-extrapolation evaluation.

Evaluates zero-shot CLIP accuracy at:
  (a) Training size: 64×64 images → 64 patches (8×8 grid)
  (b) Upsampled size: 96×96 images → 144 patches (12×12 grid)

Usage:
    uv run python scripts/eval_rope_extrap.py \
        --checkpoint runs/clip_learned_pe/best.pt \
        --pos-encoding learned
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from basics.vit import ViT
from basics.text_encoder import FrozenTextEncoder
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders


class MockProjectionHeads(torch.nn.Module):
    def __init__(self, image_proj, text_proj):
        super().__init__()
        self.image_proj = image_proj
        self.text_proj = text_proj

    def forward(self, img, txt):
        return (F.normalize(self.image_proj(img), dim=-1),
                F.normalize(self.text_proj(txt), dim=-1))


class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pos-encoding", default="learned",
                   choices=["learned", "rope1d", "rope2d"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def eval_zeroshot(vit, image_proj, text_proj, text_encoder, loader, device):
    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    text_embeds = text_encoder(class_prompts)
    _, class_proj = MockProjectionHeads(image_proj, text_proj)(
        torch.zeros(len(class_prompts), vit.d_model, device=text_embeds.device),
        text_embeds,
    )
    class_proj = F.normalize(class_proj, dim=-1)

    correct = total = 0
    vit.eval()
    image_proj.eval()
    text_proj.eval()
    for images, captions in loader:
        images = images.to(device)
        labels = torch.tensor([class_prompts.index(c) for c in captions], device=device)
        feats = vit(images)
        img_proj = F.normalize(image_proj(feats), dim=-1)
        preds = (img_proj @ class_proj.T).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Build ViT with specified pos encoding
    vit = ViT(img_size=64, patch_size=8, d_model=384, num_heads=6, num_blocks=6,
              pos_encoding=args.pos_encoding)
    vit.load_state_dict(ckpt["image_encoder"])
    vit = vit.to(device)

    image_proj = ProjectionHead(384, 256).to(device)
    image_proj.proj.load_state_dict({"weight": ckpt["image_proj"]["proj.weight"]})

    text_proj = ProjectionHead(384, 256).to(device)
    text_proj.proj.load_state_dict({"weight": ckpt["text_proj"]["proj.weight"]})

    text_encoder = FrozenTextEncoder().to(device)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # 1. Standard (training) size: 64×64
    _, val_loader, _ = build_eurosat_loaders(img_size=64, batch_size=256, num_workers=4)
    acc_train = eval_zeroshot(vit, image_proj, text_proj, text_encoder, val_loader, device)
    print(f"Train-size (64×64, 64 patches): {acc_train:.4f}")

    # 2. Extrapolation: 96×96 images → 144 patches (12×12 grid)
    from vlm.data import EuroSATCLIPDataset
    from torch.utils.data import DataLoader
    from datasets import load_dataset

    full_ds = load_dataset("blanchon/EuroSAT_RGB", split="train")
    from vlm.data import _stratified_split_indices
    _, val_indices, _ = _stratified_split_indices(full_ds["label"])
    val_ds_96 = EuroSATCLIPDataset(img_size=96, ds=full_ds.select(val_indices))

    def _collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        caps = [b[1] for b in batch]
        return imgs, caps

    val_loader_96 = DataLoader(val_ds_96, batch_size=256, shuffle=False, num_workers=4,
                               collate_fn=_collate, pin_memory=True)

    acc_extrap = eval_zeroshot(vit, image_proj, text_proj, text_encoder, val_loader_96, device)
    print(f"Extrap-size  (96×96, 144 patches): {acc_extrap:.4f}")
    print(f"Degradation: {(acc_train - acc_extrap)*100:.1f} pp")


if __name__ == "__main__":
    main()
