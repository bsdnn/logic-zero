"""DPO trainer: continue training the SFT LoRA on (chosen, rejected) preference pairs."""
from __future__ import annotations
import argparse
import json
import os
import shutil
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
    parser.add_argument("--drive-backup-dir", default=None,
                        help="If set, copy the final DPO 'best/' checkpoint to "
                             "this Drive directory after training completes.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-pick precision: bf16 on Ampere+ (A100/L4), fp16 on T4
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp16'} "
          f"(CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    print("[load] base model + SFT adapter (trainable)...", flush=True)
    model, tok = load_base_model()
    if torch.cuda.is_available():
        model = model.to("cuda")
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    # Same base+adapter as a frozen reference. DPOTrainer also accepts ref_model=None
    # in newer trl, which auto-builds one internally (saves memory), but to keep
    # behavior explicit we build it ourselves.
    print("[load] reference model (frozen copy of SFT)...", flush=True)
    ref_model, _ = load_base_model()
    if torch.cuda.is_available():
        ref_model = ref_model.to("cuda")
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_adapter)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    records = [json.loads(l) for l in open(args.dpo_data)]
    print(f"[data] {len(records)} preference pairs", flush=True)

    def to_dpo(r):
        # to_chat appends the assistant generation prompt — DPOTrainer expects
        # bare prompt text + completion-only chosen/rejected.
        prompt_chat = to_chat(tok, r["prompt"])
        return {"prompt": prompt_chat, "chosen": r["chosen"], "rejected": r["rejected"]}
    dataset = Dataset.from_list([to_dpo(r) for r in records])

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"

    cfg = DPOConfig(
        output_dir=str(out_dir / "raw"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        report_to=report_to,
        run_name="logic-zero-dpo",
    )

    # trl renamed tokenizer→processing_class around 0.13; support both.
    try:
        trainer = DPOTrainer(
            model=model, ref_model=ref_model, args=cfg,
            train_dataset=dataset, processing_class=tok,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model, ref_model=ref_model, args=cfg,
            train_dataset=dataset, tokenizer=tok,
        )

    trainer.train()
    final_dir = out_dir / "best"
    model.save_pretrained(final_dir)
    print(f"DPO saved to {final_dir}")

    if args.drive_backup_dir:
        drive_dst = Path(args.drive_backup_dir) / "best"
        if drive_dst.exists():
            shutil.rmtree(drive_dst)
        drive_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(final_dir, drive_dst)
        print(f"DPO checkpoint backed up to {drive_dst}")


if __name__ == "__main__":
    main()
