"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \
        --pretrained-vit runs/clip_eurosat/best.pt \
        --injection all_patches --mask-mode image_bidir \
        --freeze-config A
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from basics.vit import ViT
from basics.lora import LoRALinear
from vlm.data import build_clevr_dataloader
from vlm.projector import VisionLanguageProjector
from vlm.model import VisionLanguageModel
from vlm.eval import batch_clevr_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True)
    p.add_argument("--injection", choices=["cls", "all_patches", "interleaved"], default="all_patches")
    p.add_argument("--mask-mode", choices=["causal", "image_bidir"], default="causal")
    p.add_argument("--freeze-config", choices=["A", "B", "C", "D"], default="A")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-steps", type=int, default=None,
                   help="Override num_steps from config")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def apply_decoder_lora(decoder: nn.Module, rank: int = 8, alpha: float = 16.0) -> None:
    """Wrap q_proj / v_proj inside SmolLM2 attention layers with LoRALinear."""
    for name, module in decoder.named_modules():
        if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
            module.q_proj = LoRALinear(module.q_proj, rank, alpha)
        if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
            module.v_proj = LoRALinear(module.v_proj, rank, alpha)


def configure_freezing(vlm: VisionLanguageModel, freeze_config: str) -> None:
    # Freeze everything first
    for p in vlm.parameters():
        p.requires_grad = False

    # Projector is always trained
    for p in vlm.projector.parameters():
        p.requires_grad = True

    if freeze_config == "A":
        pass  # encoder + decoder frozen
    elif freeze_config == "B":
        apply_decoder_lora(vlm.decoder, rank=8, alpha=16.0)
        for name, p in vlm.decoder.named_parameters():
            if p.requires_grad:  # LoRA A/B set by LoRALinear
                pass
            # LoRALinear sets requires_grad on A and B already
    elif freeze_config == "C":
        for p in vlm.decoder.parameters():
            p.requires_grad = True
    elif freeze_config == "D":
        for p in vlm.vit.parameters():
            p.requires_grad = True
        for p in vlm.decoder.parameters():
            p.requires_grad = True


@torch.no_grad()
def evaluate(vlm, val_loader, device, injection, mask_mode, num_examples=500):
    vlm.eval()
    tokenizer = vlm.tokenizer
    all_preds, all_golds, all_qtypes = [], [], []
    count = 0
    for batch in val_loader:
        if count >= num_examples:
            break
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Encode visual tokens
        if injection == "cls":
            visual_feats = vlm.vit(images)
            if visual_feats.ndim == 2:
                visual_feats = visual_feats.unsqueeze(1)
        else:
            visual_feats = vlm.vit(images, return_all_tokens=True)
        visual_embeds = vlm.projector(visual_feats).to(torch.bfloat16)
        B, V, D = visual_embeds.shape

        # Per-example generation: slice prompt at first answer token
        for b_idx in range(B):
            if count >= num_examples:
                break
            ans_positions = (labels[b_idx] != -100).nonzero(as_tuple=True)[0]
            if len(ans_positions) == 0:
                prompt_len = attention_mask[b_idx].sum().item()
            else:
                prompt_len = int(ans_positions[0].item())
            prompt_ids_b = input_ids[b_idx, :prompt_len].unsqueeze(0)
            prompt_mask_b = attention_mask[b_idx, :prompt_len].unsqueeze(0)
            txt_emb = vlm.decoder.get_input_embeddings()(prompt_ids_b)
            inp = torch.cat([visual_embeds[b_idx:b_idx+1], txt_emb], dim=1)
            vis_am = torch.ones(1, V, dtype=prompt_mask_b.dtype, device=device)
            attn = torch.cat([vis_am, prompt_mask_b], dim=1)
            gen = vlm.decoder.generate(
                inputs_embeds=inp, attention_mask=attn,
                max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            # Take only first word (CLEVR answers are single words/numbers)
            pred = pred.split()[0] if pred.split() else ""
            all_preds.append(pred)
            all_golds.append(batch["answer"][b_idx])
            all_qtypes.append(batch["q_type"][b_idx])
            count += 1

    vlm.train()
    return batch_clevr_accuracy(all_preds[:num_examples], all_golds[:num_examples], all_qtypes[:num_examples])


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output_dir}")

    device = torch.device(args.device)

    # 1. Load tokenizer and decoder
    print("Loading decoder...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["decoder"]["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Add <image> special token for interleaved injection
    if not hasattr(tokenizer, "image_token") or "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    try:
        import flash_attn  # noqa: F401
        attn_impl = cfg["decoder"].get("attn_implementation", "eager")
    except ImportError:
        attn_impl = "eager"
    decoder = AutoModelForCausalLM.from_pretrained(
        cfg["decoder"]["model_name"],
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(device)

    # 2. Load ViT
    print("Loading ViT...")
    vit = ViT(img_size=64, patch_size=8, d_model=384, num_heads=6, num_blocks=6)
    ckpt = torch.load(args.pretrained_vit, map_location="cpu", weights_only=False)
    vit.load_state_dict(ckpt.get("image_encoder", ckpt))
    vit = vit.to(device)

    # 3. Build VLM
    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(384, d_decoder, expansion=cfg["projector"]["expansion"]).to(device)
    # Resize decoder embedding table to accommodate the new <image> token
    decoder.resize_token_embeddings(len(tokenizer))
    vlm = VisionLanguageModel(vit=vit, projector=projector, decoder=decoder,
                              tokenizer=tokenizer, image_token_id=image_token_id)

    # 4. Freeze config
    print(f"Freeze config: {args.freeze_config}")
    configure_freezing(vlm, args.freeze_config)
    trainable_n = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in vlm.parameters())
    print(f"Trainable: {trainable_n:,} / {total_n:,} ({100*trainable_n/total_n:.1f}%)")

    # 5. Data
    print("Building data loaders...")
    use_img_tok = (args.injection == "interleaved")
    train_loader = build_clevr_dataloader(
        split="train", batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"], tokenizer=tokenizer,
        use_image_token=use_img_tok,
    )
    val_loader = build_clevr_dataloader(
        split="val", batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"], tokenizer=tokenizer,
        use_image_token=use_img_tok,
    )

    # 6. Optimizer & scheduler
    trainable_params = [p for p in vlm.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=float(cfg["optim"]["lr"]),
                      weight_decay=float(cfg["optim"]["weight_decay"]),
                      betas=tuple(cfg["optim"]["betas"]))
    num_steps = args.num_steps if args.num_steps is not None else cfg["train"]["num_steps"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["optim"]["warmup_steps"], num_steps)

    # 7. Training
    print(f"Training for {num_steps} steps...")
    vlm.train()
    step = 0
    train_iter = iter(train_loader)
    best_acc = 0.0
    step_times = []
    peak_mem = 0.0

    while step < num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        t0 = time.time()
        images = batch["images"].to(device)  # ViT runs in float32
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out = vlm(images=images, input_ids=input_ids, attention_mask=attention_mask,
                  labels=labels, injection=args.injection, mask_mode=args.mask_mode)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        step_times.append(time.time() - t0)

        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device) / 1e9)

        if step % cfg["train"]["log_every"] == 0:
            avg_t = sum(step_times[-25:]) / len(step_times[-25:])
            print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f} | {avg_t*1000:.0f}ms/step")

        if step % cfg["train"]["eval_every_steps"] == 0 or step == num_steps:
            print(f"Evaluating at step {step}...")
            acc_dict = evaluate(vlm, val_loader, device, args.injection, args.mask_mode,
                                num_examples=cfg["train"]["eval_max_examples"])
            acc = acc_dict["overall"]
            print(f"Step {step} | Val acc: {acc:.4f} | {acc_dict}")
            if acc > best_acc:
                best_acc = acc
                torch.save({"vlm": vlm.state_dict(), "step": step, "acc": acc},
                           args.output_dir / "best.pt")

    # Always save final checkpoint regardless of eval accuracy
    torch.save({"vlm": vlm.state_dict(), "step": num_steps, "acc": best_acc},
               args.output_dir / "vlm_final.pt")
    print(f"Saved final checkpoint to {args.output_dir / 'vlm_final.pt'}")

    avg_step_time = sum(step_times) / len(step_times)
    results = {
        "injection": args.injection,
        "mask_mode": args.mask_mode,
        "freeze_config": args.freeze_config,
        "best_val_acc": best_acc,
        "trainable_params": trainable_n,
        "peak_gpu_gb": peak_mem,
        "avg_step_time_s": avg_step_time,
    }
    print(f"\nResults: {results}")
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
