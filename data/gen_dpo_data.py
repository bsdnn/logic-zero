"""DPO data: sample 4 responses per puzzle from SFT model, label correct/incorrect,
keep puzzles with at least one of each. Stratified floor of pairs per n (spec §5.3).

Resume support: each accepted pair is appended to the output JSONL immediately,
and the file is mirrored to --drive-backup-dir every N pairs. If the process
crashes or Colab disconnects, re-running the script picks up where it left off
by reading the existing JSONL and skipping puzzle hashes already covered.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import shutil
import time
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

# Option B (reduced) targets — see chat history 2026-05-22. Halved from the
# original 1780-puzzle plan to fit DPO data gen into a single Colab session
# (~6-8h on A100) while still giving DPO enough preference signal per bucket.
PER_N_RAW = {2: 30, 3: 100, 4: 200, 5: 200, 6: 200}
MIN_PAIRS_PER_N = {2: 5, 3: 50, 4: 50, 5: 50, 6: 50}

# Mirror to Drive every this many newly-added pairs.
DRIVE_BACKUP_EVERY = 10


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


def load_existing_pairs(out_path: Path):
    """Read previously-written pairs from JSONL; return (pairs_by_n, hashes_seen)."""
    pairs_by_n: dict[int, list] = defaultdict(list)
    hashes_seen: set[str] = set()
    if not out_path.exists() or out_path.stat().st_size == 0:
        return pairs_by_n, hashes_seen
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pairs_by_n[rec["n"]].append(rec)
            hashes_seen.add(rec["hash"])
    return pairs_by_n, hashes_seen


def restore_from_drive(out_path: Path, drive_dir: Path | None):
    """If local JSONL missing but Drive has a backup, copy it down."""
    if drive_dir is None:
        return
    drive_jsonl = drive_dir / out_path.name
    if drive_jsonl.exists() and not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(drive_jsonl, out_path)
        print(f"[restore] {drive_jsonl} -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-adapter", default="results/checkpoints/sft/best")
    parser.add_argument("--out", default="data/dpo_data.jsonl")
    parser.add_argument("--hashes", default="data/eval_hashes.json")
    parser.add_argument("--max-rounds", type=int, default=3, help="Per-bucket top-up rounds")
    parser.add_argument("--drive-backup-dir", default=None,
                        help="If set, mirrors the JSONL here every "
                             f"{DRIVE_BACKUP_EVERY} new pairs (use a Drive path "
                             "to survive Colab disconnects).")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drive_dir = Path(args.drive_backup_dir) if args.drive_backup_dir else None
    if drive_dir is not None:
        drive_dir.mkdir(parents=True, exist_ok=True)

    # Restore from Drive if local got wiped (runtime recycled)
    restore_from_drive(out_path, drive_dir)

    # Read existing progress
    pairs_by_n, hashes_seen = load_existing_pairs(out_path)
    if hashes_seen:
        print(f"[resume] existing pairs by n: "
              f"{ {n: len(pairs_by_n[n]) for n in PER_N_RAW} }, "
              f"{len(hashes_seen)} puzzle hashes covered", flush=True)

    print("[load] base model + SFT adapter...", flush=True)
    t_load = time.time()
    model, tok = load_base_model()
    if torch.cuda.is_available():
        model = model.to("cuda")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model.eval()
    print(f"[load] done in {time.time()-t_load:.0f}s", flush=True)

    excluded = set(json.loads(Path(args.hashes).read_text())) | hashes_seen
    seed_state = [TRAIN_SEED_START + 10_000_000]  # disjoint from SFT range
    rng = random.Random(0)

    # Open output file in append mode for incremental writes
    out_f = open(out_path, "a")
    new_pairs_since_backup = 0
    total_attempts_by_n: dict[int, int] = defaultdict(int)
    t_start = time.time()

    def backup_to_drive():
        if drive_dir is None:
            return
        try:
            drive_jsonl = drive_dir / out_path.name
            shutil.copy(out_path, drive_jsonl)
        except Exception as e:
            print(f"[backup] WARN drive backup failed: {e}", flush=True)

    try:
        for round_idx in range(args.max_rounds):
            for n, target_raw in PER_N_RAW.items():
                if len(pairs_by_n[n]) >= MIN_PAIRS_PER_N[n]:
                    continue
                print(
                    f"[round {round_idx}] n={n} pairs={len(pairs_by_n[n])}/"
                    f"{MIN_PAIRS_PER_N[n]} (target {target_raw} raw puzzles)",
                    flush=True,
                )
                for puzzle_i in range(target_raw):
                    text, gt, h = fresh_puzzle(n, seed_state, excluded)
                    total_attempts_by_n[n] += 1
                    prompt = to_chat(tok, text)
                    with torch.no_grad():
                        responses = sample_n_responses(model, tok, prompt, k=4)
                    labels = [extract_answer(r, n=n) == gt for r in responses]
                    correct = [r for r, ok in zip(responses, labels) if ok]
                    incorrect = [r for r, ok in zip(responses, labels) if not ok]
                    if not correct or not incorrect:
                        continue  # all-correct or all-wrong → no preference signal
                    pair = {
                        "prompt": text, "n": n, "hash": h,
                        "chosen": rng.choice(correct),
                        "rejected": rng.choice(incorrect),
                    }
                    pairs_by_n[n].append(pair)
                    # Append immediately + flush so a crash doesn't lose it
                    out_f.write(json.dumps(pair) + "\n")
                    out_f.flush()
                    os.fsync(out_f.fileno())
                    new_pairs_since_backup += 1

                    if new_pairs_since_backup >= DRIVE_BACKUP_EVERY:
                        backup_to_drive()
                        new_pairs_since_backup = 0
                        elapsed = time.time() - t_start
                        totals = {nn: len(pairs_by_n[nn]) for nn in PER_N_RAW}
                        print(
                            f"  [progress] {sum(totals.values())} pairs, "
                            f"by n={totals}, {elapsed:.0f}s elapsed",
                            flush=True,
                        )

                    if len(pairs_by_n[n]) >= MIN_PAIRS_PER_N[n]:
                        break  # bucket done

            if all(len(pairs_by_n[nn]) >= MIN_PAIRS_PER_N[nn] for nn in PER_N_RAW):
                print("[done] all buckets met floor", flush=True)
                break
    finally:
        out_f.close()
        backup_to_drive()  # final backup

    print("\n=== Final per-n counts ===")
    totals = {n: len(pairs_by_n[n]) for n in PER_N_RAW}
    for n in PER_N_RAW:
        print(f"  n={n}: pairs={totals[n]}/{MIN_PAIRS_PER_N[n]} "
              f"(attempts this run: {total_attempts_by_n[n]})")
    print(f"Total pairs: {sum(totals.values())}")
    print(f"Output: {out_path}")
    if drive_dir is not None:
        print(f"Drive backup: {drive_dir / out_path.name}")


if __name__ == "__main__":
    main()
