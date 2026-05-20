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

# n=2: puzzle space exhausts at ~67 unique solutions; cap at 30 to match eval/dev.
# n=3: 250; n=4-6: 500 each.  Total target ≈ 1780 verified examples.
TARGETS = {2: 30, 3: 250, 4: 500, 5: 500, 6: 500}

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
    # Append mode: if output already exists (resumed session), keep prior work.
    # Truncate only if file is empty or doesn't exist.
    if not out_path.exists() or out_path.stat().st_size == 0:
        out_path.write_text("")  # create / clear

    for n, target in TARGETS.items():
        verified = 0
        attempts = 0
        retried = set()
        print(f"[n={n}] target={target}", file=sys.stderr)
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
