"""DPO data: sample 4 responses per puzzle from SFT model, label correct/incorrect,
keep puzzles with at least one of each. Stratified floor of 100 pairs per n (spec §5.3)."""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel

from train.common import (
    load_base_model,
    extract_answer,
    count_solutions,
    VerifierTimeout,
    to_chat,
)
from data.gen_puzzles import generate_puzzle
from data.gen_eval_data import hash_puzzle, TRAIN_SEED_START

# Per-n raw targets follow the 1:1:2:2:2 ratio from spec §5.2 / Task 5.
# n=2 puzzle space is small (only ~67 distinct unique-solution puzzles given
# the current statement templates; see commit 9ea632d), so we cap n=2 at 30
# raw puzzles to avoid an infinite loop. Other buckets keep plan values.
PER_N_RAW = {2: 30, 3: 250, 4: 500, 5: 500, 6: 500}
# Per-bucket pair floors: n=3..6 keep the plan's value of 100; n=2 is lowered
# to 10 because the ~67 unique-puzzle ceiling makes >=100 pairs infeasible at
# any plausible SFT skill level.
MIN_PAIRS_PER_N = {2: 10, 3: 100, 4: 100, 5: 100, 6: 100}

def sample_n_responses(model, tok, prompt: str, k: int = 4):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    responses = []
    for i in range(k):
        torch.manual_seed(1000 + i)
        output = model.generate(
            **inputs, max_new_tokens=512,
            do_sample=True, temperature=0.8,
            pad_token_id=tok.eos_token_id,
        )
        responses.append(tok.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
    return responses

def fresh_puzzle(n: int, seed_state: list[int], excluded: set[str]):
    while True:
        seed_state[0] += 1
        text, gt, stmts = generate_puzzle(n=n, seed=seed_state[0], return_statements=True)
        try:
            if count_solutions(stmts, n=n, timeout_ms=5000) != 1:
                continue
        except VerifierTimeout:
            continue
        h = hash_puzzle(text)
        if h in excluded:
            continue
        excluded.add(h)
        return text, gt, h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-adapter", default="results/checkpoints/sft/best")
    parser.add_argument("--out", default="data/dpo_data.jsonl")
    parser.add_argument("--hashes", default="data/eval_hashes.json")
    parser.add_argument("--max-rounds", type=int, default=3, help="Per-bucket top-up rounds")
    args = parser.parse_args()

    model, tok = load_base_model()
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model.eval()

    excluded = set(json.loads(Path(args.hashes).read_text()))
    seed_state = [TRAIN_SEED_START + 10_000_000]  # disjoint from SFT range
    pairs_by_n = defaultdict(list)
    rng = random.Random(0)

    for round_idx in range(args.max_rounds):
        for n, target_raw in PER_N_RAW.items():
            if len(pairs_by_n[n]) >= MIN_PAIRS_PER_N[n]:
                continue
            print(f"[round {round_idx}] n={n} pairs={len(pairs_by_n[n])}", flush=True)
            for _ in range(target_raw):
                text, gt, h = fresh_puzzle(n, seed_state, excluded)
                prompt = to_chat(tok, text)
                with torch.no_grad():
                    responses = sample_n_responses(model, tok, prompt, k=4)
                labels = [extract_answer(r, n=n) == gt for r in responses]
                correct = [r for r, ok in zip(responses, labels) if ok]
                incorrect = [r for r, ok in zip(responses, labels) if not ok]
                if not correct or not incorrect:
                    continue
                pairs_by_n[n].append({
                    "prompt": text, "n": n, "hash": h,
                    "chosen": rng.choice(correct),
                    "rejected": rng.choice(incorrect),
                })
        if all(len(pairs_by_n[n]) >= MIN_PAIRS_PER_N[n] for n in PER_N_RAW):
            print("All buckets met floor.", flush=True)
            break

    # Persist
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for n in PER_N_RAW:
            for rec in pairs_by_n[n]:
                f.write(json.dumps(rec) + "\n")
    print({n: len(pairs_by_n[n]) for n in PER_N_RAW})

if __name__ == "__main__":
    main()
