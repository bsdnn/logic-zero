"""External baselines on the same 1800-puzzle eval set.
- Prompted-Qwen: base model + CoT prompt, no training.
- DeepSeek-V3: zero-shot API call.
Same output format as eval/run_eval.py."""
from __future__ import annotations
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI

from train.common import load_base_model, extract_answer, check_format
from train.common import to_chat
from eval.run_eval import generate, eval_one_pass

COT_INSTRUCTION = (
    "Solve the following Knights and Knaves puzzle. Reason step by step inside <think></think> tags, "
    'then give the answer inside <answer></answer> tags in the format "A: knight, B: knave, ...".\n\n'
)

def prompted_qwen(out_path: str, eval_path: str):
    model, tok = load_base_model()
    model.eval()
    eval_recs = [json.loads(l) for l in open(eval_path)]
    # Prepend the CoT instruction to each puzzle
    for r in eval_recs:
        r["puzzle"] = COT_INSTRUCTION + r["puzzle"]
    # Greedy only (no need for 3-seed since base is mostly deterministic at T=0)
    with torch.no_grad():
        results = {"greedy": eval_one_pass(model, tok, eval_recs, temperature=0.0, seed=None),
                   "sampled": []}
    Path(out_path).write_text(json.dumps(results, indent=2))

def deepseek_baseline(out_path: str, eval_path: str, sample_seeds: int = 3):
    load_dotenv()
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    eval_recs = [json.loads(l) for l in open(eval_path)]

    def one_pass(temperature: float, seed: int | None):
        per_bucket_correct = defaultdict(int)
        per_bucket_total = defaultdict(int)
        format_ok = 0
        for rec in eval_recs:
            n = len(rec["ground_truth"])
            content = COT_INSTRUCTION + rec["puzzle"]
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": content}],
                        temperature=temperature,
                        seed=seed if seed is not None else None,
                        max_tokens=800,
                    )
                    text = resp.choices[0].message.content
                    break
                except Exception as e:
                    print(f"API error attempt {attempt}: {e}")
                    time.sleep(5)
            else:
                text = ""
            per_bucket_total[n] += 1
            if check_format(text): format_ok += 1
            if extract_answer(text, n=n) == rec["ground_truth"]:
                per_bucket_correct[n] += 1
        return {
            "per_bucket_correct": dict(per_bucket_correct),
            "per_bucket_total": dict(per_bucket_total),
            "format_compliance": format_ok / len(eval_recs),
        }

    results = {"greedy": one_pass(0.0, None), "sampled": []}
    for seed in range(1, sample_seeds + 1):
        results["sampled"].append({"seed": seed, **one_pass(0.7, seed)})
    Path(out_path).write_text(json.dumps(results, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["qwen", "deepseek"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-data", default="data/eval_data.jsonl")
    args = parser.parse_args()
    if args.mode == "qwen":
        prompted_qwen(args.out, args.eval_data)
    else:
        deepseek_baseline(args.out, args.eval_data)

if __name__ == "__main__":
    main()
