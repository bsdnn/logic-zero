import pytest
from data.gen_puzzles import generate_puzzle, Statement

def test_generates_n_inhabitants():
    text, gt = generate_puzzle(n=3, seed=0)
    assert len(gt) == 3
    assert set(gt.keys()) == {"A", "B", "C"}
    assert all(v in ("knight", "knave") for v in gt.values())

def test_puzzle_text_mentions_each_person():
    text, gt = generate_puzzle(n=4, seed=1)
    for label in "ABCD":
        assert label in text

def test_statements_are_consistent_with_identities():
    """For each person, their statement must be true iff they're a knight."""
    from data.gen_puzzles import evaluate_statement
    text, gt = generate_puzzle(n=5, seed=2)
    # generate_puzzle should expose the structured statements too
    _, _, statements = generate_puzzle(n=5, seed=2, return_statements=True)
    for label, stmt in statements.items():
        truth_value = evaluate_statement(stmt, gt)
        is_knight = gt[label] == "knight"
        assert truth_value == is_knight, f"{label} ({gt[label]}) says: {stmt}, eval={truth_value}"

def test_seed_determinism():
    a = generate_puzzle(n=3, seed=42)
    b = generate_puzzle(n=3, seed=42)
    assert a == b


import json
import subprocess
from pathlib import Path
import hashlib

@pytest.mark.slow
def test_gen_eval_produces_correct_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    # Run as a subprocess so we exercise the script's main()
    result = subprocess.run(
        ["python", "-m", "data.gen_eval_data", "--out-dir", "data"],
        cwd=Path(__file__).parent.parent,  # project root
        env={**__import__("os").environ, "EVAL_OUT_DIR": str(tmp_path / "data")},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    eval_recs = [json.loads(l) for l in (tmp_path / "data" / "eval_data.jsonl").read_text().splitlines()]
    dev_recs = [json.loads(l) for l in (tmp_path / "data" / "dev_data.jsonl").read_text().splitlines()]
    hashes = json.loads((tmp_path / "data" / "eval_hashes.json").read_text())
    # n=2 capped at 30 eval + 10 dev because puzzle space exhausts at ~67
    # unique distinct unique-solution puzzles. Other buckets at spec values.
    assert len(eval_recs) == 1530  # 30 + 300*5
    assert len(dev_recs) == 170    # 10 + 40*4
    assert len(hashes) == 1700     # all eval + dev hashes
    from collections import Counter
    eval_buckets = Counter(len(r["ground_truth"]) for r in eval_recs)
    assert eval_buckets[2] == 30
    assert all(eval_buckets[n] == 300 for n in range(3, 8))
    dev_buckets = Counter(len(r["ground_truth"]) for r in dev_recs)
    assert dev_buckets[2] == 10
    assert all(dev_buckets[n] == 40 for n in range(3, 7))
