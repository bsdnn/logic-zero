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
        # Training mode forces use_cache=False + gradient_checkpointing.
        # For generation we want the opposite — KV cache is critical
        # (10x faster), and ckpt is meaningless without backward.
        model.eval()
        prev_cache = getattr(model.config, "use_cache", True)
        # Always try to disable ckpt (the `is_gradient_checkpointing` flag is
        # unreliable on PEFT-wrapped models — just call disable and catch).
        ckpt_was_on = False
        for tgt in (model, getattr(model, "base_model", None)):
            if tgt is None:
                continue
            try:
                tgt.gradient_checkpointing_disable()
                ckpt_was_on = True
            except Exception:
                pass
        model.config.use_cache = True

        import time
        t0 = time.time()
        correct = 0
        try:
            with torch.no_grad():
                for i, rec in enumerate(self.dev):
                    prompt = to_chat(self.tok, rec["puzzle"])
                    inputs = self.tok(prompt, return_tensors="pt").to(model.device)
                    output = model.generate(
                        **inputs, max_new_tokens=700, do_sample=False,
                        pad_token_id=self.tok.eos_token_id,
                        # Stop as soon as </answer> token shown up to save
                        # ~30% of generation time on n=2/3 where reasoning
                        # is short.
                        stop_strings=["</answer>"],
                        tokenizer=self.tok,
                        # Explicitly unset sampling params so transformers
                        # doesn't warn 3 times per puzzle.
                        temperature=None, top_p=None, top_k=None,
                    )
                    resp = self.tok.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    pred = extract_answer(resp, n=len(rec["ground_truth"]))
                    if pred == rec["ground_truth"]:
                        correct += 1
                    if (i + 1) % 25 == 0:
                        elapsed = time.time() - t0
                        print(f"  [dev] {i+1}/{len(self.dev)}  acc_so_far={correct/(i+1):.3f}  {elapsed:.0f}s elapsed", flush=True)
        finally:
            # Restore training-time settings.
            model.config.use_cache = prev_cache
            if ckpt_was_on:
                try:
                    model.gradient_checkpointing_enable()
                except Exception:
                    pass
            model.train()

        acc = correct / len(self.dev)
        print(f"[epoch {state.epoch:.0f}] dev_acc={acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = int(state.epoch)
            model.save_pretrained(self.out_dir / "best")
            (self.out_dir / "best" / "dev_acc.json").write_text(json.dumps({"acc": acc, "epoch": self.best_epoch}))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", default="data/sft_data.jsonl")
    parser.add_argument("--dev-data", default="data/dev_data.jsonl")
    parser.add_argument("--out-dir", default="results/checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)  # was 4 → OOM on L4
    parser.add_argument("--grad-accum", type=int, default=8)  # was 4; keep effective batch=16
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="Disable gradient checkpointing (faster but uses much more VRAM).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-pick precision: bf16 on Ampere+ (L4/A100/etc), fp16 on T4.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp16'} (CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    model, tok = load_base_model()
    model = get_peft_model(model, make_lora_config())

    # Gradient checkpointing: trades ~20% speed for ~40% activation-memory cut.
    # Essential on 22GB L4 with bs=2, seq=1024.  PEFT-on-frozen-base needs
    # enable_input_require_grads() so ckpt'd activations can be reconstructed.
    use_grad_ckpt = not args.no_grad_ckpt
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        # PEFT wraps the base model; disable cache for ckpt to work.
        if hasattr(model, "config"):
            model.config.use_cache = False

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
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=use_grad_ckpt,
        logging_steps=10,
        save_strategy="no",  # we save ourselves in the callback
        max_seq_length=args.max_seq_length,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="logic-zero-sft",
    )

    # trl renamed tokenizer→processing_class around 0.13; support both.
    try:
        trainer = SFTTrainer(
            model=model, args=cfg, train_dataset=dataset,
            processing_class=tok,
            callbacks=[DevAccuracyCallback(tok, dev_records, out_dir)],
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model, args=cfg, train_dataset=dataset,
            tokenizer=tok,
            callbacks=[DevAccuracyCallback(tok, dev_records, out_dir)],
        )
    trainer.train()
    print(f"Best dev_acc={trainer.callback_handler.callbacks[-1].best_acc:.3f} at epoch {trainer.callback_handler.callbacks[-1].best_epoch}")

if __name__ == "__main__":
    main()
