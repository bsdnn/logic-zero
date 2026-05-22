"""Run a model over the held-out eval set, producing per-bucket accuracy.

Defaults to greedy-only (T=0) since 3-seed sampled adds 4x cost and is
only useful for tight statistical CIs which we don't need at this stage
(spec §5.5 "sampled = secondary"). Pass --with-sampled to include them.

Fixed vs. earlier version:
- model.to('cuda') after load (load_base_model returns CPU model)
- max_new_tokens 512 → 1000 (was truncating n≥5 generations)
- stop_strings=['</answer>'] (10x faster — stops at end-of-answer)
- Progress prints every 25 puzzles + ETA
- Resume support: writes intermediate progress every 50 puzzles
- Per-n breakdown in output JSON
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel

from train.common import load_base_model, extract_answer, check_format, to_chat


def generate(model, tok, prompt: str, temperature: float, seed: int | None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    output = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=None if not do_sample else 0.95,
        top_k=None,
        pad_token_id=tok.eos_token_id,
        stop_strings=["</answer>"],
        tokenizer=tok,
    )
    return tok.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def eval_one_pass(
    model,
    tok,
    eval_recs: list[dict],
    temperature: float,
    seed: int | None,
    pass_label: str = "greedy",
    progress_every: int = 25,
    resume_path: Path | None = None,
):
    per_bucket_correct: dict[int, int] = defaultdict(int)
    per_bucket_total: dict[int, int] = defaultdict(int)
    format_ok = 0
    t0 = time.time()
    n_recs = len(eval_recs)
    for i, rec in enumerate(eval_recs):
        n = len(rec["ground_truth"])
        prompt = to_chat(tok, rec["puzzle"])
        resp = generate(model, tok, prompt, temperature, seed)
        per_bucket_total[n] += 1
        if check_format(resp):
            format_ok += 1
        pred = extract_answer(resp, n=n)
        if pred == rec["ground_truth"]:
            per_bucket_correct[n] += 1
        if (i + 1) % progress_every == 0:
            total_c = sum(per_bucket_correct.values())
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_recs - i - 1)
            print(
                f"  [{pass_label}] {i+1:4d}/{n_recs}  acc={total_c/(i+1):.3f}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                flush=True,
            )
            # Resume snapshot every 2*progress_every (default 50) puzzles
            if resume_path and (i + 1) % (progress_every * 2) == 0:
                snapshot = {
                    "pass_label": pass_label,
                    "completed": i + 1,
                    "per_bucket_correct": dict(per_bucket_correct),
                    "per_bucket_total": dict(per_bucket_total),
                    "format_compliance_so_far": format_ok / (i + 1),
                }
                resume_path.write_text(json.dumps(snapshot, indent=2))
    return {
        "per_bucket_correct": dict(per_bucket_correct),
        "per_bucket_total": dict(per_bucket_total),
        "per_bucket_acc": {
            str(n): per_bucket_correct[n] / per_bucket_total[n]
            for n in per_bucket_total
        },
        "total_correct": sum(per_bucket_correct.values()),
        "total": n_recs,
        "overall_acc": sum(per_bucket_correct.values()) / n_recs,
        "format_compliance": format_ok / n_recs,
        "duration_sec": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter; omit for base model")
    parser.add_argument("--eval-data", default="data/eval_data.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--with-sampled", action="store_true",
                        help="Also run 3-seed T=0.7 sampled passes (4x slower)")
    parser.add_argument("--seeds", type=int, nargs=3, default=[1, 2, 3])
    parser.add_argument("--limit", type=int, default=None,
                        help="Only eval first N puzzles (for testing)")
    args = parser.parse_args()

    print(f"[load] base model...", flush=True)
    t0 = time.time()
    model, tok = load_base_model()
    # load_base_model returns CPU model — explicit GPU move required
    if torch.cuda.is_available():
        model = model.to("cuda")
    if args.adapter:
        print(f"[load] adapter from {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    dev = next(model.parameters()).device
    assert dev.type == "cuda", f"Model on {dev}, refusing to run on CPU"
    print(f"[load] done in {time.time()-t0:.0f}s, device={dev}", flush=True)

    eval_recs = [json.loads(l) for l in open(args.eval_data)]
    if args.limit:
        eval_recs = eval_recs[: args.limit]
    print(f"[data] {len(eval_recs)} puzzles", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = out_path.with_suffix(".inprogress.json")

    results = {
        "model": "sft+lora" if args.adapter else "base",
        "adapter": args.adapter,
        "eval_data": args.eval_data,
        "n_puzzles": len(eval_recs),
        "greedy": None,
        "sampled": [],
    }

    print(f"\n[greedy pass] T=0", flush=True)
    with torch.no_grad():
        results["greedy"] = eval_one_pass(
            model, tok, eval_recs,
            temperature=0.0, seed=None,
            pass_label="greedy",
            resume_path=resume_path,
        )
    print(f"[greedy] overall acc = {results['greedy']['overall_acc']:.3f}  "
          f"per-n = {results['greedy']['per_bucket_acc']}", flush=True)

    if args.with_sampled:
        for seed in args.seeds:
            print(f"\n[sampled pass seed={seed}] T=0.7", flush=True)
            with torch.no_grad():
                r = eval_one_pass(
                    model, tok, eval_recs,
                    temperature=0.7, seed=seed,
                    pass_label=f"seed-{seed}",
                    resume_path=resume_path,
                )
            results["sampled"].append({"seed": seed, **r})

    out_path.write_text(json.dumps(results, indent=2))
    if resume_path.exists():
        resume_path.unlink()
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
