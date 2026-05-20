"""GRPO trainer. Handles both the main pipeline (ref = DPO) and ablation (ref = SFT)
via the --ref-adapter flag (spec §4 ablation branch)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

from train.common import load_base_model, to_chat
from train.reward import grpo_reward_funcs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-adapter", required=True, help="LoRA to begin training from (DPO ckpt for main, SFT ckpt for ablation)")
    parser.add_argument("--ref-adapter", required=True, help="Frozen LoRA used as reference for KL (same as --start-adapter typically)")
    parser.add_argument("--grpo-data", default="data/grpo_data.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--run-name", default="logic-zero-grpo-main")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_base_model()
    model = PeftModel.from_pretrained(model, args.start_adapter, is_trainable=True)
    ref_model, _ = load_base_model()
    ref_model = PeftModel.from_pretrained(ref_model, args.ref_adapter)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    records = [json.loads(l) for l in open(args.grpo_data)]
    dataset = Dataset.from_list([
        {"prompt": to_chat(tok, r["prompt"]), "ground_truth": r["ground_truth"]}
        for r in records
    ])

    cfg = GRPOConfig(
        output_dir=str(out_dir / "raw"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        run_name=args.run_name,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_funcs,
        args=cfg,
        train_dataset=dataset,
        tokenizer=tok,
    )
    trainer.train()
    model.save_pretrained(out_dir / "best")
    print(f"GRPO saved to {out_dir/'best'}")

if __name__ == "__main__":
    main()
