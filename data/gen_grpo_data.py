"""GRPO data: puzzles + ground truth, no labels needed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from data.gen_puzzles import generate_puzzle
from train.common import count_solutions, VerifierTimeout
from data.gen_eval_data import hash_puzzle, TRAIN_SEED_START

# n=2 capped at 30 (puzzle space limit, see commit 9ea632d); rest per plan.
# Total = 1780 instead of 2000; ratio is 30:250:500:500:500.
TARGETS = {2: 30, 3: 250, 4: 500, 5: 500, 6: 500}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/grpo_data.jsonl")
    parser.add_argument("--hashes", default="data/eval_hashes.json")
    args = parser.parse_args()

    excluded = set(json.loads(Path(args.hashes).read_text()))
    seed = TRAIN_SEED_START + 20_000_000  # disjoint from SFT (3M) and DPO (13M) ranges
    out = []
    for n, target in TARGETS.items():
        produced = 0
        while produced < target:
            text, gt, stmts = generate_puzzle(n=n, seed=seed, return_statements=True)
            seed += 1
            try:
                if count_solutions(stmts, n=n, timeout_ms=5000) != 1:
                    continue
            except VerifierTimeout:
                continue
            h = hash_puzzle(text)
            if h in excluded:
                continue
            excluded.add(h)
            out.append({"prompt": text, "ground_truth": gt, "n": n, "hash": h})
            produced += 1
        print(f"  n={n} done", flush=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in out))

if __name__ == "__main__":
    main()
