"""§3 — CLIP-style pretraining on EuroSAT."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from vlm.data import build_eurosat_loaders, EUROSAT_CLASSES
from basics.vit import ViT
from basics.text_encoder import FrozenTextEncoder
from vlm.clip import clip_loss
from vlm.eval import zeroshot_classification_accuracy


class MockProjectionHeads(nn.Module):
    def __init__(self, image_proj, text_proj):
        super().__init__()
        self.image_proj = image_proj
        self.text_proj = text_proj
    def forward(self, img, txt):
        if img.dim() == 3:
            img = img[:, 0, :]
        return F.normalize(self.image_proj(img), dim=-1), F.normalize(self.text_proj(txt), dim=-1)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()

def main() -> None:
    print("Starting pretraining script...", flush=True)
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=str(args.output_dir))

    if args.wandb:
        import wandb
        wandb.init(project="hw3_clip_eurosat", config=cfg)

    device = torch.device(args.device)

    print("Building train/val/test loaders...", flush=True)
    train_loader, val_loader, test_loader = build_eurosat_loaders(
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 2)
    )
    print("Loaders built successfully!", flush=True)

    print("Initializing Models...", flush=True)
    vit_cfg = cfg.get("vit", {})
    image_encoder = ViT(
        img_size=vit_cfg.get("img_size", 224),
        patch_size=vit_cfg.get("patch_size", 16),
        d_model=vit_cfg.get("d_model", 384),
        num_heads=vit_cfg.get("num_heads", 6),
        num_blocks=vit_cfg.get("num_blocks", 6),
        num_classes=0,
    ).to(device)

    text_encoder = FrozenTextEncoder().to(device)

    embed_dim = cfg.get("embed_dim", 256)
    image_proj = ProjectionHead(vit_cfg.get("d_model", 384), embed_dim).to(device)
    text_dim = 384
    text_proj = ProjectionHead(text_dim, embed_dim).to(device)

    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    epochs = cfg.get("epochs", 10)
    lr = cfg.get("lr", 1e-3)

    trainable_params = list(image_encoder.parameters()) + list(image_proj.parameters()) + list(text_proj.parameters()) + [logit_scale]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=cfg.get("weight_decay", 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    print(f"Starting training loop for {epochs} epochs on {device}...", flush=True)
    for epoch in range(epochs):
        image_encoder.train()
        image_proj.train()
        text_proj.train()

        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            images, texts = batch
            images = images.to(device)

            optimizer.zero_grad()

            img_embeds = image_encoder(images)
            img_embeds = image_proj(img_embeds[:, 0, :])

            with torch.no_grad():
                txt_features = text_encoder(texts)
            txt_embeds = text_proj(txt_features.clone())

            img_embeds = torch.nn.functional.normalize(img_embeds, p=2, dim=-1)
            txt_embeds = torch.nn.functional.normalize(txt_embeds, p=2, dim=-1)

            temperature = 1.0 / torch.exp(logit_scale)
            loss = clip_loss(img_embeds, txt_embeds, temperature=temperature)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                logit_scale.clamp_(max=math.log(100.0))

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        print(f"Evaluating Zero-shot Val Acc...", flush=True)
        image_encoder.d_model = 384
        image_encoder.eval()
        image_proj.eval()
        class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
        class_indices = list(range(len(EUROSAT_CLASSES)))
        mock_heads = MockProjectionHeads(image_proj, text_proj)
        val_acc = zeroshot_classification_accuracy(
            image_encoder, mock_heads, text_encoder, val_loader, 
            class_prompts, class_indices, device
        )

        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Zero-shot Val Acc: {val_acc:.4f}", flush=True)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if args.wandb:
            wandb.log({"train_loss": avg_loss, "val_acc": val_acc, "epoch": epoch})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "image_encoder": image_encoder.state_dict(),
                "image_proj": image_proj.state_dict(),
                "text_proj": text_proj.state_dict(),
                "logit_scale": logit_scale
            }, args.output_dir / "best.pt")
            print(f"Saved new best checkpoint with Val Acc: {best_val_acc:.4f}", flush=True)

    writer.close()
    print("Pretraining complete!", flush=True)

if __name__ == "__main__":
    main()
