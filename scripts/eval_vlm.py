"""§5 — Qualitative evaluation of a trained VLM.

Usage:
    uv run python scripts/eval_vlm.py \
        --checkpoint runs/vlm_cls_causal_A/vlm_final.pt \
        --injection cls --num-examples 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from basics.vit import ViT
from vlm.data import build_clevr_dataloader
from vlm.projector import VisionLanguageProjector
from vlm.model import VisionLanguageModel
from vlm.eval import batch_clevr_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, default=Path("runs/clip_eurosat/best.pt"))
    p.add_argument("--injection", choices=["cls", "all_patches", "interleaved"], default="cls")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_vlm(pretrained_vit: Path, decoder_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Add <image> token
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_name, dtype=torch.bfloat16, attn_implementation=attn_impl
    ).to(device)
    decoder.resize_token_embeddings(len(tokenizer))

    vit = ViT(img_size=64, patch_size=8, d_model=384, num_heads=6, num_blocks=6)
    ckpt = torch.load(pretrained_vit, map_location="cpu", weights_only=False)
    vit.load_state_dict(ckpt.get("image_encoder", ckpt))
    vit = vit.to(device)

    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(384, d_decoder, expansion=4).to(device)

    vlm = VisionLanguageModel(vit=vit, projector=projector, decoder=decoder,
                              tokenizer=tokenizer, image_token_id=image_token_id)
    return vlm, tokenizer, image_token_id


@torch.no_grad()
def evaluate(vlm, tokenizer, val_loader, device, injection, num_examples=500):
    vlm.eval()
    all_preds, all_golds, all_qtypes = [], [], []
    dec_dtype = next(vlm.decoder.parameters()).dtype
    vit_dtype = next(vlm.vit.parameters()).dtype

    for batch in val_loader:
        if len(all_preds) >= num_examples:
            break
        images = batch["images"].to(device)
        questions = batch["question"]
        answers_gold = batch["answer"]
        qtypes_batch = batch["q_type"]
        B = images.shape[0]

        # Encode visual tokens
        if injection == "cls":
            vis_feats = vlm.vit(images.to(vit_dtype))
            if vis_feats.ndim == 2:
                vis_feats = vis_feats.unsqueeze(1)
        else:
            vis_feats = vlm.vit(images.to(vit_dtype), return_all_tokens=True)
        vis_embeds = vlm.projector(vis_feats).to(dec_dtype)
        V = vis_embeds.shape[1]

        # Tokenize prompts directly (no answer tokens)
        prompts = [f"Question: {q} Answer:" for q in questions]
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=64).to(device)
        txt_emb = vlm.decoder.get_input_embeddings()(enc["input_ids"])

        # Per-example generation (batch_size=1 for simplicity)
        for b_idx in range(min(B, num_examples - len(all_preds))):
            txt_b = txt_emb[b_idx:b_idx+1]
            attn_b = enc["attention_mask"][b_idx:b_idx+1]
            inp = torch.cat([vis_embeds[b_idx:b_idx+1], txt_b], dim=1)
            vis_am = torch.ones(1, V, dtype=attn_b.dtype, device=device)
            attn = torch.cat([vis_am, attn_b], dim=1)
            gen = vlm.decoder.generate(
                inputs_embeds=inp, attention_mask=attn,
                max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            pred = pred.split()[0] if pred.split() else ""
            all_preds.append(pred)
            all_golds.append(answers_gold[b_idx])
            all_qtypes.append(qtypes_batch[b_idx])

    return batch_clevr_accuracy(all_preds, all_golds, all_qtypes), \
           list(zip(all_preds, all_golds, all_qtypes))


def main() -> None:
    args = parse_args()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    decoder_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

    print("Building VLM...")
    vlm, tokenizer, image_token_id = build_vlm(args.pretrained_vit, decoder_name, device)

    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("vlm", ckpt)
    # Load only the projector weights (the parts that were trained in config A)
    proj_state = {k.replace("projector.", ""): v for k, v in state.items() if k.startswith("projector.")}
    vlm.projector.load_state_dict(proj_state, strict=True)
    print("Loaded projector weights.")

    use_img_tok = (args.injection == "interleaved")
    val_loader = build_clevr_dataloader(
        split=args.split, batch_size=32, num_workers=4,
        tokenizer=tokenizer, use_image_token=use_img_tok,
    )

    print(f"Evaluating {args.num_examples} examples...")
    acc_dict, examples = evaluate(vlm, tokenizer, val_loader, device, args.injection, args.num_examples)

    print(f"Accuracy: {acc_dict}")
    out_dir = args.checkpoint.parent
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump({"accuracy": acc_dict, "injection": args.injection}, f, indent=2)

    # Print 10 examples
    print("\n=== Sample Predictions ===")
    for pred, gold, qtype in examples[:10]:
        match = "✓" if pred.lower() == gold.lower() else "✗"
        print(f"[{match}] Q-type={qtype} | pred={repr(pred)} | gold={repr(gold)}")


if __name__ == "__main__":
    main()
