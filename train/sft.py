"""SFT trainer: LoRA-fit Qwen2.5-1.5B on K&K puzzle→completion pairs.
Selects best checkpoint by dev-set accuracy (spec §4)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from train.common import load_base_model, make_lora_config, extract_answer, to_chat

class DevAccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer, dev_records: list[dict], out_dir: Path):
        self.tok = tokenizer
        self.dev = dev_records
        self.out_dir = out_dir
        self.best_acc = -1.0
        self.best_epoch = -1

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model.eval()
        correct = 0
        with torch.no_grad():
            for rec in self.dev:
                prompt = to_chat(self.tok, rec["puzzle"])
                inputs = self.tok(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    **inputs, max_new_tokens=400, do_sample=False, temperature=0.0,
                    pad_token_id=self.tok.eos_token_id,
                )
                resp = self.tok.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pred = extract_answer(resp, n=len(rec["ground_truth"]))
                if pred == rec["ground_truth"]:
                    correct += 1
        acc = correct / len(self.dev)
        print(f"[epoch {state.epoch:.0f}] dev_acc={acc:.3f}", flush=True)
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = int(state.epoch)
            model.save_pretrained(self.out_dir / "best")
            (self.out_dir / "best" / "dev_acc.json").write_text(json.dumps({"acc": acc, "epoch": self.best_epoch}))
        model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", default="data/sft_data.jsonl")
    parser.add_argument("--dev-data", default="data/dev_data.jsonl")
    parser.add_argument("--out-dir", default="results/checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_base_model()
    model = get_peft_model(model, make_lora_config())

    sft_records = [json.loads(l) for l in open(args.sft_data)]
    dev_records = [json.loads(l) for l in open(args.dev_data)]

    def to_record(r):
        return {"text": to_chat(tok, r["puzzle"], r["completion"])}

    dataset = Dataset.from_list([to_record(r) for r in sft_records])

    cfg = SFTConfig(
        output_dir=str(out_dir / "raw"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="no",  # we save ourselves in the callback
        max_seq_length=args.max_seq_length,
        report_to="wandb",
        run_name="logic-zero-sft",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        tokenizer=tok,
        callbacks=[DevAccuracyCallback(tok, dev_records, out_dir)],
    )
    trainer.train()
    print(f"Best dev_acc={trainer.callback_handler.callbacks[-1].best_acc:.3f} at epoch {trainer.callback_handler.callbacks[-1].best_epoch}")

if __name__ == "__main__":
    main()
