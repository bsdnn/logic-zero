"""Generate SFT training examples by asking DeepSeek-V3 to solve K&K puzzles,
keeping only those whose answer matches ground truth.
Per-n quota table: n=2 capped at 30 (puzzle space ~67 unique solutions),
n=3: 250, n=4-6: 500 each. Total target ~1780 verified."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from data.gen_puzzles import generate_puzzle
from train.common import count_solutions, extract_answer, VerifierTimeout
from data.gen_eval_data import hash_puzzle, TRAIN_SEED_START

# n=2: puzzle space exhausts at ~67 unique solutions; eval+dev already
# consume 40 (30+10), so SFT capped at 20 to leave ~7 headroom.
# n=3: 250; n=4-6: 500 each.  Total target ≈ 1770 verified examples.
TARGETS = {2: 20, 3: 250, 4: 500, 5: 500, 6: 500}

PROMPT_TEMPLATE = """Solve this Knights and Knaves puzzle. Show step-by-step reasoning inside <think></think> tags, then give the final answer inside <answer></answer> tags in the format "A: knight, B: knave, ...".

Puzzle: {puzzle}"""

def call_deepseek(client: OpenAI, puzzle: str) -> str:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(puzzle=puzzle)}],
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sft_data.jsonl")
    parser.add_argument("--hashes", default="data/eval_hashes.json")
    parser.add_argument("--retry-once", action="store_true", help="Retry failed n=5,6 puzzles once.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    excluded = set(json.loads(Path(args.hashes).read_text()))
    seed = TRAIN_SEED_START
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume support: if output file exists, load its hashes into excluded so
    # we don't regenerate already-saved puzzles, AND tally existing verified
    # counts per-n so we skip already-done buckets.
    existing_per_n: dict[int, int] = {n: 0 for n in TARGETS}
    if out_path.exists() and out_path.stat().st_size > 0:
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            excluded.add(rec["hash"])
            if rec["n"] in existing_per_n:
                existing_per_n[rec["n"]] += 1
        print(f"[resume] found existing records per n: {existing_per_n}", file=sys.stderr)
    else:
        out_path.write_text("")  # create empty

    for n, target in TARGETS.items():
        verified = existing_per_n.get(n, 0)
        attempts = 0
        retried = set()
        if verified >= target:
            print(f"[n={n}] already at {verified}/{target}, skipping", file=sys.stderr)
            continue
        print(f"[n={n}] target={target} (have {verified})", file=sys.stderr)
        while verified < target:
            # Generate a fresh valid puzzle not in excluded
            while True:
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
                break
            attempts += 1
            try:
                completion = call_deepseek(client, text)
            except Exception as e:
                print(f"  API error: {e}; sleeping 5s", file=sys.stderr)
                time.sleep(5)
                continue
            pred = extract_answer(completion, n=n)
            if pred == gt:
                # Persist
                with open(out_path, "a") as f:
                    f.write(json.dumps({"puzzle": text, "completion": completion, "n": n, "hash": h}) + "\n")
                verified += 1
                if verified % 25 == 0:
                    print(f"  n={n} verified={verified}/{target} attempts={attempts}", file=sys.stderr)
            elif args.retry_once and n in (5, 6) and h not in retried:
                retried.add(h)
                # Pop this puzzle back into the queue by undoing the exclusion;
                # but only count this once via `retried` set.
                try:
                    completion = call_deepseek(client, text)
                except Exception:
                    continue
                pred = extract_answer(completion, n=n)
                if pred == gt:
                    with open(out_path, "a") as f:
                        f.write(json.dumps({"puzzle": text, "completion": completion, "n": n, "hash": h}) + "\n")
                    verified += 1
        print(f"  n={n} DONE: {verified} verified after {attempts} attempts", file=sys.stderr)

if __name__ == "__main__":
    main()
