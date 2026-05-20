"""Generate held-out eval (1800) + dev (200) puzzles in one run.
Writes their hashes to eval_hashes.json so subsequent training-data scripts can exclude them."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from data.gen_puzzles import generate_puzzle
from train.common import count_solutions, VerifierTimeout

# Per-n targets. n=2 puzzle space is small (only ~67 distinct unique-solution
# puzzles given the current statement templates), so we cap n=2 below the
# spec's 300/40 to avoid an infinite loop. Other buckets keep spec values.
EVAL_TARGETS = {2: 30, 3: 300, 4: 300, 5: 300, 6: 300, 7: 300}  # total 1530
DEV_TARGETS  = {2: 10, 3: 40,  4: 40,  5: 40,  6: 40}            # total 170

EVAL_SEED_START = 1_000_000   # arbitrary disjoint seed range
DEV_SEED_START  = 2_000_000
TRAIN_SEED_START = 3_000_000  # documented here; used by other scripts

def hash_puzzle(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def collect(n: int, target: int, seed_start: int, taken_hashes: set[str]):
    """Generate `target` unique valid puzzles of size n. Returns list of dicts."""
    out = []
    seed = seed_start
    while len(out) < target:
        text, gt, stmts = generate_puzzle(n=n, seed=seed, return_statements=True)
        seed += 1
        try:
            count = count_solutions(stmts, n=n, timeout_ms=5000)
        except VerifierTimeout:
            continue
        if count != 1:
            continue
        h = hash_puzzle(text)
        if h in taken_hashes:
            continue
        taken_hashes.add(h)
        out.append({"puzzle": text, "ground_truth": gt, "hash": h})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.environ.get("EVAL_OUT_DIR", "data"))
    args = parser.parse_args()
    # env var wins if both present (per spec)
    out_dir = Path(os.environ.get("EVAL_OUT_DIR", args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    taken = set()
    eval_records = []
    for n, target in EVAL_TARGETS.items():
        print(f"[eval] generating n={n} target={target}...", file=sys.stderr)
        eval_records.extend(collect(n, target, EVAL_SEED_START + n * 100_000, taken))
    dev_records = []
    for n, target in DEV_TARGETS.items():
        print(f"[dev]  generating n={n} target={target}...", file=sys.stderr)
        dev_records.extend(collect(n, target, DEV_SEED_START + n * 100_000, taken))
    (out_dir / "eval_data.jsonl").write_text("\n".join(json.dumps(r) for r in eval_records))
    (out_dir / "dev_data.jsonl").write_text("\n".join(json.dumps(r) for r in dev_records))
    (out_dir / "eval_hashes.json").write_text(json.dumps(sorted(taken)))
    print(f"Wrote {len(eval_records)} eval + {len(dev_records)} dev, total {len(taken)} hashes.")

if __name__ == "__main__":
    main()
