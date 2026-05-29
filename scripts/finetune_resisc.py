"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from basics.vit import ViT
from basics.lora import apply_lora_to_attention
from vlm.data import build_resisc45_loaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=None, help="LoRA alpha; defaults to 2*rank")
    p.add_argument("--pretrained", type=Path, help="Path to pretrained ViT checkpoint")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vit(pretrained: Path | None, device: torch.device) -> ViT:
    vit = ViT(img_size=64, patch_size=8, d_model=384, num_heads=6, num_blocks=6)
    if pretrained is not None:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("image_encoder", ckpt)
        vit.load_state_dict(state)
        print(f"Loaded ViT from {pretrained}")
    return vit.to(device)


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    num_classes = cfg["num_classes"]
    train_cfg = cfg["train"]
    opt_cfg = cfg["optim"]

    # Per-method LR override
    method_overrides = cfg.get("methods", {}).get(args.method, {})
    lr = float(method_overrides.get("lr", opt_cfg["lr"]))

    # --- Build model ---
    vit = load_vit(args.pretrained, device)

    if args.method == "linear_probe":
        for p in vit.parameters():
            p.requires_grad = False
    elif args.method == "lora":
        alpha = args.alpha if args.alpha is not None else 2 * args.rank
        vit = apply_lora_to_attention(vit, rank=args.rank, alpha=alpha)
    # full_ft: all params remain trainable

    head = nn.Linear(384, num_classes).to(device)

    trainable = list(filter(lambda p: p.requires_grad, vit.parameters())) + list(head.parameters())
    total = sum(p.numel() for p in vit.parameters()) + sum(p.numel() for p in head.parameters())
    trainable_n = sum(p.numel() for p in trainable)
    print(f"Method: {args.method} | Total params: {total:,} | Trainable: {trainable_n:,} ({100*trainable_n/total:.1f}%)")

    # --- Data ---
    train_loader, test_loader = build_resisc45_loaders(
        batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"]
    )

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=float(opt_cfg["weight_decay"]))
    num_epochs = train_cfg["num_epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    # --- Training ---
    best_acc = 0.0
    results = {}
    t0 = time.time()

    for epoch in range(num_epochs):
        vit.train()
        head.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            feats = vit(images)
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval
        vit.eval()
        head.eval()
        correct = total_ex = 0
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = head(vit(images))
                correct += (logits.argmax(1) == labels).sum().item()
                total_ex += labels.size(0)
        acc = correct / total_ex
        if acc > best_acc:
            best_acc = acc
        print(f"Epoch {epoch+1}/{num_epochs} | Test acc: {acc:.4f} | Best: {best_acc:.4f}")

    wall_time = time.time() - t0
    results = {
        "method": args.method,
        "rank": args.rank,
        "final_test_acc": best_acc,
        "trainable_params": trainable_n,
        "total_params": total,
        "peak_gpu_gb": peak_mem,
        "wall_time_s": wall_time,
    }
    print(f"\nResults: {results}")

    import json
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
