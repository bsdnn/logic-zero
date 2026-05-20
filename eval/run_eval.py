"""Run a model over the held-out eval set, producing per-bucket accuracy.
Reports greedy (T=0) as primary and 3-seed sampled (T=0.7) as secondary (spec §5.5)."""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel

from train.common import load_base_model, extract_answer, check_format
from train.common import to_chat

def generate(model, tok, prompt: str, temperature: float, seed: int | None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    output = model.generate(
        **inputs, max_new_tokens=512,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def eval_one_pass(model, tok, eval_recs: list[dict], temperature: float, seed: int | None):
    per_bucket_correct = defaultdict(int)
    per_bucket_total = defaultdict(int)
    format_ok = 0
    for rec in eval_recs:
        n = len(rec["ground_truth"])
        prompt = to_chat(tok, rec["puzzle"])
        resp = generate(model, tok, prompt, temperature, seed)
        per_bucket_total[n] += 1
        if check_format(resp):
            format_ok += 1
        pred = extract_answer(resp, n=n)
        if pred == rec["ground_truth"]:
            per_bucket_correct[n] += 1
    return {
        "per_bucket_correct": dict(per_bucket_correct),
        "per_bucket_total": dict(per_bucket_total),
        "format_compliance": format_ok / len(eval_recs),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=False, default=None,
                        help="Path to LoRA adapter; omit for base model")
    parser.add_argument("--eval-data", default="data/eval_data.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs=3, default=[1, 2, 3])
    args = parser.parse_args()

    model, tok = load_base_model()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    eval_recs = [json.loads(l) for l in open(args.eval_data)]
    results = {"greedy": None, "sampled": []}

    print("Greedy pass...")
    with torch.no_grad():
        results["greedy"] = eval_one_pass(model, tok, eval_recs, temperature=0.0, seed=None)
    for seed in args.seeds:
        print(f"Sampled pass seed={seed}...")
        with torch.no_grad():
            results["sampled"].append({"seed": seed, **eval_one_pass(model, tok, eval_recs, temperature=0.7, seed=seed)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
