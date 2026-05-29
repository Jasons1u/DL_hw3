"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \
        --injection all_patches --mask-mode image_bidir \
        --freeze-config A
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from basics.vit import ViT
from vlm.data import build_clevr_dataloader
from vlm.projector import VisionLanguageProjector
from vlm.model import VisionLanguageModel

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, "
             "C=+full decoder, D=all three.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Train VLM script initialized. Saving to:", args.output_dir)

    device = torch.device(args.device)

    # 1. Load Tokenizer & LM Decoder
    print("Loading tokenizer and decoder...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['decoder']['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    decoder = AutoModelForCausalLM.from_pretrained(
        cfg['decoder']['model_name'],
        torch_dtype=getattr(torch, cfg['decoder']['torch_dtype']),
        attn_implementation=cfg['decoder'].get('attn_implementation', 'eager')
    ).to(device)

    # 2. Load Pretrained ViT
    print("Loading Pretrained ViT...")
    vit = ViT(img_size=64, patch_size=8, d_model=384, num_heads=6, num_blocks=6, num_classes=0)
    state_dict = torch.load(args.pretrained_vit, map_location='cpu')
    if 'image_encoder' in state_dict:
        vit.load_state_dict(state_dict['image_encoder'])
    else:
        vit.load_state_dict(state_dict)
    vit = vit.to(device)

    # 3. Create Projector & VLM
    print("Initializing Projector and VLM...")
    d_image = 384
    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(d_image, d_decoder, expansion=cfg['projector']['expansion']).to(device)
    
    vlm = VisionLanguageModel(
        vit=vit, 
        projector=projector, 
        decoder=decoder, 
        tokenizer=tokenizer, 
        image_token_id=getattr(tokenizer, 'image_token_id', None)
    ).to(device)

    # 4. Apply Freezing Logic (Setup A: Projector Only)
    # Default to A for now to get it running
    print(f"Applying freeze config {args.freeze_config}...")
    for param in vlm.vit.parameters():
        param.requires_grad = False
    if args.freeze_config == 'A':
        for param in vlm.decoder.parameters():
            param.requires_grad = False
        for param in vlm.projector.parameters():
            param.requires_grad = True

    # 5. Dataloader
    print("Building Dataloader...")
    train_loader = build_clevr_dataloader(
        split="train", 
        batch_size=cfg['train']['batch_size'], 
        num_workers=cfg['train']['num_workers'],
        tokenizer=tokenizer
    )

    # 6. Optimizer & Scheduler
    trainable_params = [p for p in vlm.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=float(cfg['optim']['lr']), weight_decay=cfg['optim']['weight_decay'], betas=tuple(cfg['optim']['betas']))
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg['optim']['warmup_steps'], cfg['train']['num_steps'])

    # 7. Training Loop
    print(f"Starting training for {cfg['train']['num_steps']} steps...")
    vlm.train()
    step = 0
    train_iter = iter(train_loader)
    
    start_time = time.time()
    
    while step < cfg['train']['num_steps']:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch['images'].to(device, dtype=torch.bfloat16 if cfg['decoder']['torch_dtype'] == 'bfloat16' else torch.float32)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        # Bfloat16 autocast for mixed precision if needed, but tensors are already casted
        out = vlm(
            images=images, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            injection=args.injection,
            mask_mode=args.mask_mode
        )
        
        loss = out['loss']
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        step += 1
        
        if step % cfg['train']['log_every'] == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{cfg['train']['num_steps']} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")
            start_time = time.time()

    print("Training complete!")
    torch.save(vlm.state_dict(), args.output_dir / "vlm_final.pt")
    print(f"Saved final model to {args.output_dir / 'vlm_final.pt'}")

if __name__ == "__main__":
    main()
