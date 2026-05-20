"""DPO trainer: continue training the SFT LoRA on (chosen, rejected) preference pairs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

from train.common import load_base_model, to_chat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-adapter", default="results/checkpoints/sft/best")
    parser.add_argument("--dpo-data", default="data/dpo_data.jsonl")
    parser.add_argument("--out-dir", default="results/checkpoints/dpo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_base_model()
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    ref_model, _ = load_base_model()
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_adapter)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    records = [json.loads(l) for l in open(args.dpo_data)]
    def to_dpo(r):
        prompt_chat = to_chat(tok, r["prompt"])  # already adds generation prompt
        return {"prompt": prompt_chat, "chosen": r["chosen"], "rejected": r["rejected"]}
    dataset = Dataset.from_list([to_dpo(r) for r in records])

    cfg = DPOConfig(
        output_dir=str(out_dir / "raw"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        report_to="wandb",
        run_name="logic-zero-dpo",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=dataset,
        tokenizer=tok,
    )
    trainer.train()
    model.save_pretrained(out_dir / "best")
    print(f"DPO saved to {out_dir/'best'}")

if __name__ == "__main__":
    main()
