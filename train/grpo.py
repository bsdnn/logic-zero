"""GRPO trainer. Handles both the main pipeline (ref = DPO) and ablation (ref = SFT)
via the --ref-adapter flag (spec §4 ablation branch).

Colab-friendly: auto-picks bf16/fp16 based on GPU, moves model to CUDA after load,
optional Drive backup of the final 'best/' checkpoint, wandb only if WANDB_API_KEY set.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

from train.common import load_base_model, to_chat
from train.reward import grpo_reward_funcs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-adapter", required=True,
                        help="LoRA to begin training from (DPO ckpt for main, SFT ckpt for ablation)")
    parser.add_argument("--ref-adapter", required=True,
                        help="Frozen LoRA used as reference for KL (same as --start-adapter typically)")
    parser.add_argument("--grpo-data", default="data/grpo_data.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of training prompts (for cheap runs).")
    parser.add_argument("--run-name", default="logic-zero-grpo-main")
    parser.add_argument("--drive-backup-dir", default=None,
                        help="If set, copy the final GRPO 'best/' checkpoint to "
                             "this Drive directory after training completes.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-pick precision: bf16 on Ampere+ (A100/L4), fp16 on T4.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp16'} "
          f"(CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})",
          flush=True)

    print(f"[load] base model + start adapter ({args.start_adapter}, trainable)...", flush=True)
    model, tok = load_base_model()
    if torch.cuda.is_available():
        model = model.to("cuda")
    model = PeftModel.from_pretrained(model, args.start_adapter, is_trainable=True)

    print(f"[load] reference model (frozen) from {args.ref_adapter}...", flush=True)
    ref_model, _ = load_base_model()
    if torch.cuda.is_available():
        ref_model = ref_model.to("cuda")
    ref_model = PeftModel.from_pretrained(ref_model, args.ref_adapter)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    records = [json.loads(l) for l in open(args.grpo_data)]
    if args.limit is not None:
        records = records[: args.limit]
    print(f"[data] {len(records)} prompts", flush=True)

    dataset = Dataset.from_list([
        {"prompt": to_chat(tok, r["prompt"]), "ground_truth": r["ground_truth"]}
        for r in records
    ])

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"

    cfg = GRPOConfig(
        output_dir=str(out_dir / "raw"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=1.0,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=report_to,
        run_name=args.run_name,
    )

    # trl renamed tokenizer→processing_class around 0.13; support both.
    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=grpo_reward_funcs,
            args=cfg,
            train_dataset=dataset,
            processing_class=tok,
        )
    except TypeError:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=grpo_reward_funcs,
            args=cfg,
            train_dataset=dataset,
            tokenizer=tok,
        )

    trainer.train()
    final_dir = out_dir / "best"
    model.save_pretrained(final_dir)
    print(f"GRPO saved to {final_dir}")

    if args.drive_backup_dir:
        drive_dst = Path(args.drive_backup_dir) / "best"
        if drive_dst.exists():
            shutil.rmtree(drive_dst)
        drive_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(final_dir, drive_dst)
        print(f"GRPO checkpoint backed up to {drive_dst}")


if __name__ == "__main__":
    main()
