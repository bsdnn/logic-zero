# Logic-Zero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Qwen2.5-1.5B-Instruct on Knights & Knaves logic puzzles through a full SFT → DPO → GRPO post-training pipeline, plus an SFT → GRPO ablation, with quantified comparison against external baselines.

**Architecture:** All training uses LoRA adapters (r=16, target the Qwen attention projections), stacked across stages so each stage starts from the previous checkpoint. Reward signal is rule-based via a z3 SAT verifier. Data, training, and evaluation share one `extract_answer` parser and one `check_format` predicate (in `train/common.py`) so reward and metrics never disagree. Eval and dev sets are generated first and hash-locked before any training data is produced.

**Tech Stack:** Python 3.10+, PyTorch 2.4, transformers 4.46, trl 0.13, peft 0.13, z3-solver, OpenAI Python SDK (for DeepSeek-V3, which serves OpenAI-compatible API), wandb. Single-GPU on Colab Pro (A100 40GB target, L4 24GB fallback).

**Spec:** [docs/specs/2026-05-18-logic-zero-design.md](../specs/2026-05-18-logic-zero-design.md)

---

## File Structure

Locked from spec §8 with minor additions for tests:

```
logic-zero/
├── README.md                       # written in Task 22
├── requirements.txt                # written in Task 1
├── .gitignore                      # written in Task 1
├── pyproject.toml                  # ruff + pytest config (Task 1)
├── data/
│   ├── __init__.py
│   ├── gen_puzzles.py              # Task 2 — puzzle generator
│   ├── gen_eval_data.py            # Task 5 — eval (1800) + dev (200) in one run
│   ├── gen_sft_data.py             # Task 6 — DeepSeek-driven SFT data
│   ├── gen_dpo_data.py             # Task 11 — DPO pair construction
│   ├── gen_grpo_data.py            # Task 14 — GRPO prompts + GT
│   ├── eval_data.jsonl             # (generated artefact)
│   ├── dev_data.jsonl              # (generated artefact)
│   ├── eval_hashes.json            # (generated artefact)
│   ├── sft_data.jsonl              # (generated artefact)
│   ├── dpo_data.jsonl              # (generated artefact)
│   └── grpo_data.jsonl             # (generated artefact)
├── train/
│   ├── __init__.py
│   ├── common.py                   # Tasks 3, 4 — verify_puzzle, extract_answer, check_format, model loading
│   ├── sft.py                      # Task 7
│   ├── dpo.py                      # Task 12
│   ├── grpo.py                     # Tasks 16, 17, 18 — main + ablation via config flag
│   ├── reward.py                   # Task 16 — uses common.py functions
│   ├── test_common.py              # Tasks 3, 4 — tests for verifier + parsers
│   └── test_extract.py             # Task 4 — focused extract_answer fixtures
├── eval/
│   ├── __init__.py
│   ├── run_eval.py                 # Task 8 — greedy + 3-seed sampled
│   ├── baselines.py                # Task 9 — prompted-Qwen + DeepSeek-V3
│   ├── compare.py                  # Task 20 — table + charts
│   └── analyze_responses.py        # Task 21 — qualitative spot-checks
├── notebooks/
│   ├── 01_sft.ipynb                # Task 7 sidekick
│   ├── 02_dpo.ipynb                # Task 12 sidekick
│   ├── 03_grpo.ipynb               # Tasks 17, 18 sidekick (config flag for main vs ablation)
│   └── 04_eval.ipynb               # Tasks 10, 13, 19 sidekick
├── results/
│   ├── accuracy_table.md           # Task 20
│   ├── training_curves.png         # Task 20
│   ├── reward_distribution.png     # Task 20
│   └── example_responses.md        # Task 21
└── docs/
    ├── specs/2026-05-18-logic-zero-design.md
    └── plans/2026-05-19-logic-zero.md   # this file
```

**File responsibilities:**

| File | Single purpose |
|---|---|
| `gen_puzzles.py` | Random K&K puzzle generation; outputs `(puzzle_text, ground_truth_dict)`. Pure Python. |
| `common.py` | Shared functions used by both training and eval: `verify_puzzle`, `extract_answer`, `check_format`, `load_model_and_tokenizer`, `make_lora_config`. |
| `gen_eval_data.py` | Builds eval and dev sets first, writes their hashes to `eval_hashes.json`. |
| Each `gen_*_data.py` | One-shot script: load `eval_hashes.json` → generate fresh puzzles → write its own `*.jsonl`. |
| `train/{sft,dpo,grpo}.py` | One trainer each; reads the relevant jsonl, builds Trainer, fits, saves LoRA adapter. |
| `train/reward.py` | Single `reward()` function used by `grpo.py`. Imports from `common.py`. |
| `eval/run_eval.py` | Loads a model + adapter, runs greedy + 3-seed sampled over the held-out eval set, writes per-bucket accuracy JSON. |
| `eval/baselines.py` | Calls a base/instruct Qwen with CoT prompt (local) and DeepSeek-V3 (API). Same output format as `run_eval.py`. |
| `eval/compare.py` | Reads every eval JSON, produces `results/accuracy_table.md` + matplotlib charts. |

---

## Task 1: Project bootstrap

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pyproject.toml`, `README.md` (stub), `data/__init__.py`, `train/__init__.py`, `eval/__init__.py`

- [ ] **Step 1: Initialize git repo and create directory skeleton**

```bash
cd C:/Users/86177/logic-zero
git init
git add docs/  # the spec already lives here
git commit -m "chore: import design spec"
mkdir -p data train eval notebooks results
touch data/__init__.py train/__init__.py eval/__init__.py
```

- [ ] **Step 2: Write `requirements.txt` with pinned versions from spec §4**

```text
torch==2.4.0
transformers==4.46.0
trl==0.13.0
peft==0.13.0
datasets==3.0.0
accelerate==1.0.0
bitsandbytes==0.44.0
z3-solver==4.13.0
wandb==0.18.0
openai==1.50.0
pytest==8.3.0
ruff==0.6.0
matplotlib==3.9.0
huggingface-hub==0.26.0
python-dotenv==1.0.0
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.env
wandb/
results/*.png
results/checkpoints/
data/*.jsonl
data/eval_hashes.json
*.ipynb_checkpoints/
.DS_Store
```

- [ ] **Step 4: Write `pyproject.toml` with ruff + pytest config**

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["train", "eval"]
python_files = "test_*.py"
addopts = "-v --tb=short"
```

- [ ] **Step 5: Write `README.md` stub**

```markdown
# Logic-Zero

Train Qwen2.5-1.5B on Knights & Knaves logic puzzles via SFT → DPO → GRPO.

See [design spec](docs/specs/2026-05-18-logic-zero-design.md) and [implementation plan](docs/plans/2026-05-19-logic-zero.md).

Status: in development. Final results table will replace this stub.
```

- [ ] **Step 6: Install deps in a fresh venv to confirm pinning works**

```bash
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on Linux
pip install -r requirements.txt
```
Expected: clean install, no version conflicts.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore pyproject.toml README.md data/__init__.py train/__init__.py eval/__init__.py
git commit -m "chore: project bootstrap with pinned deps"
```

---

## Task 2: Knights & Knaves puzzle generator

**Files:**
- Create: `data/gen_puzzles.py`
- Test: `data/test_gen_puzzles.py`

Approach: each person gets a randomly assigned identity (knight/knave). Each then makes ONE statement chosen randomly from a template. For a knight, the statement must be true given the assignment; for a knave, it must be false. After construction, we'll run the SAT verifier (Task 3) to confirm uniqueness.

- [ ] **Step 1: Write the failing tests**

`data/test_gen_puzzles.py`:
```python
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
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest data/test_gen_puzzles.py -v
```
Expected: ImportError or `ModuleNotFoundError`.

- [ ] **Step 3: Implement `data/gen_puzzles.py`**

```python
"""Knights & Knaves puzzle generator."""
from __future__ import annotations
import random
import string
from dataclasses import dataclass
from typing import Literal

Identity = Literal["knight", "knave"]

@dataclass(frozen=True)
class Statement:
    """A single statement made by one inhabitant."""
    kind: str  # "is", "same", "diff", "at_least_knights", "at_least_knaves", "self_knight", "self_knave"
    args: tuple  # interpretation depends on kind

def evaluate_statement(stmt: Statement, gt: dict[str, Identity]) -> bool:
    if stmt.kind == "is":
        target, claimed = stmt.args
        return gt[target] == claimed
    if stmt.kind == "same":
        a, b = stmt.args
        return gt[a] == gt[b]
    if stmt.kind == "diff":
        a, b = stmt.args
        return gt[a] != gt[b]
    if stmt.kind == "at_least_knights":
        k, = stmt.args
        return sum(1 for v in gt.values() if v == "knight") >= k
    if stmt.kind == "at_least_knaves":
        k, = stmt.args
        return sum(1 for v in gt.values() if v == "knave") >= k
    if stmt.kind == "self_knight":
        speaker, = stmt.args
        return gt[speaker] == "knight"
    if stmt.kind == "self_knave":
        speaker, = stmt.args
        return gt[speaker] == "knave"
    raise ValueError(f"Unknown statement kind: {stmt.kind}")

def _render_statement(speaker: str, stmt: Statement) -> str:
    if stmt.kind == "is":
        target, claimed = stmt.args
        return f'- {speaker} says: "{target} is a {claimed}."'
    if stmt.kind == "same":
        a, b = stmt.args
        return f'- {speaker} says: "{a} and {b} are of the same kind."'
    if stmt.kind == "diff":
        a, b = stmt.args
        return f'- {speaker} says: "{a} and {b} are of different kinds."'
    if stmt.kind == "at_least_knights":
        k, = stmt.args
        return f'- {speaker} says: "At least {k} of us are knights."'
    if stmt.kind == "at_least_knaves":
        k, = stmt.args
        return f'- {speaker} says: "At least {k} of us are knaves."'
    if stmt.kind == "self_knight":
        return f'- {speaker} says: "I am a knight."'
    if stmt.kind == "self_knave":
        return f'- {speaker} says: "I am a knave."'
    raise ValueError(stmt.kind)

def _random_statement(rng: random.Random, speaker: str, labels: list[str], gt: dict[str, Identity], n: int) -> Statement:
    """Construct a statement whose truth value matches the speaker's identity."""
    must_be_true = gt[speaker] == "knight"
    # Try candidates until one matches required truth value.
    for _ in range(50):
        kind = rng.choice([
            "is", "same", "diff", "at_least_knights", "at_least_knaves",
            "self_knight", "self_knave",
        ])
        if kind == "is":
            target = rng.choice(labels)
            claimed = rng.choice(("knight", "knave"))
            stmt = Statement("is", (target, claimed))
        elif kind in ("same", "diff"):
            a, b = rng.sample(labels, 2)
            stmt = Statement(kind, (a, b))
        elif kind == "at_least_knights":
            k = rng.randint(1, n)
            stmt = Statement(kind, (k,))
        elif kind == "at_least_knaves":
            k = rng.randint(1, n)
            stmt = Statement(kind, (k,))
        elif kind == "self_knight":
            stmt = Statement(kind, (speaker,))
        else:
            stmt = Statement(kind, (speaker,))
        if evaluate_statement(stmt, gt) == must_be_true:
            return stmt
    # Fallback: a "I am a knight/knave" statement always satisfies the constraint.
    return Statement("self_knight" if must_be_true else "self_knave", (speaker,))

def generate_puzzle(n: int, seed: int, return_statements: bool = False):
    assert 2 <= n <= 7
    rng = random.Random(seed)
    labels = list(string.ascii_uppercase[:n])
    gt = {lab: rng.choice(("knight", "knave")) for lab in labels}
    statements = {lab: _random_statement(rng, lab, labels, gt, n) for lab in labels}
    intro = (
        "On an island, every inhabitant is either a knight (always tells truth) "
        f"or a knave (always lies). You meet {n} people: {', '.join(labels)}.\n"
    )
    rendered = "\n".join(_render_statement(lab, statements[lab]) for lab in labels)
    text = intro + rendered + "\nDetermine each person's identity."
    if return_statements:
        return text, gt, statements
    return text, gt
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
pytest data/test_gen_puzzles.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add data/gen_puzzles.py data/test_gen_puzzles.py
git commit -m "feat(data): K&K puzzle generator with truth-consistent statements"
```

---

## Task 3: SAT verifier in common.py

**Files:**
- Create: `train/common.py`, `train/test_common.py`

The verifier takes a puzzle's structured statements + a candidate assignment and returns whether (a) the assignment is consistent, and uses z3 to (b) enumerate solutions for uniqueness checks during data generation. Exposes two timeout modes per spec §5.1.

- [ ] **Step 1: Write the failing tests**

`train/test_common.py`:
```python
from train.common import verify_puzzle, count_solutions, VerifierTimeout
from data.gen_puzzles import generate_puzzle

def test_verify_correct_assignment():
    text, gt, stmts = generate_puzzle(n=3, seed=10, return_statements=True)
    assert verify_puzzle(stmts, gt, mode="generation") is True

def test_verify_wrong_assignment():
    _, gt, stmts = generate_puzzle(n=3, seed=11, return_statements=True)
    wrong = dict(gt)
    # Flip one identity
    k = next(iter(wrong))
    wrong[k] = "knave" if wrong[k] == "knight" else "knight"
    assert verify_puzzle(stmts, wrong, mode="generation") is False

def test_count_solutions_unique():
    """Most generated puzzles should have exactly 1 solution. Run a few and check."""
    unique_count = 0
    for seed in range(20):
        _, _, stmts = generate_puzzle(n=3, seed=seed, return_statements=True)
        if count_solutions(stmts, n=3, timeout_ms=5000) == 1:
            unique_count += 1
    assert unique_count >= 10, "Expected most random puzzles to have a unique solution"

def test_reward_mode_has_tighter_timeout():
    """Smoke check: reward mode call returns without raising for normal input."""
    _, gt, stmts = generate_puzzle(n=3, seed=42, return_statements=True)
    assert verify_puzzle(stmts, gt, mode="reward") is True
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest train/test_common.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement the verifier portion of `train/common.py`**

```python
"""Shared helpers used by both training and evaluation."""
from __future__ import annotations
import z3
from typing import Literal

from data.gen_puzzles import Statement

class VerifierTimeout(Exception):
    pass

_TIMEOUT_MS = {"generation": 5000, "reward": 500}

def _stmt_to_z3(stmt: Statement, vars: dict[str, z3.BoolRef]) -> z3.BoolRef:
    """Convert a Statement into a z3 boolean expression that is True iff the statement is true.
    vars[label] is a Bool that's True for knight."""
    k = stmt.kind
    if k == "is":
        target, claimed = stmt.args
        return vars[target] if claimed == "knight" else z3.Not(vars[target])
    if k == "same":
        a, b = stmt.args
        return vars[a] == vars[b]
    if k == "diff":
        a, b = stmt.args
        return vars[a] != vars[b]
    if k == "at_least_knights":
        kk, = stmt.args
        return z3.Sum([z3.If(v, 1, 0) for v in vars.values()]) >= kk
    if k == "at_least_knaves":
        kk, = stmt.args
        return z3.Sum([z3.If(v, 0, 1) for v in vars.values()]) >= kk
    if k == "self_knight":
        speaker, = stmt.args
        return vars[speaker]
    if k == "self_knave":
        speaker, = stmt.args
        return z3.Not(vars[speaker])
    raise ValueError(k)

def _build_constraints(statements: dict[str, Statement]) -> tuple[z3.Solver, dict[str, z3.BoolRef]]:
    """Build z3 solver where each person's statement-truth must match their knight-hood."""
    s = z3.Solver()
    vars = {lab: z3.Bool(lab) for lab in statements.keys()}
    for speaker, stmt in statements.items():
        truth = _stmt_to_z3(stmt, vars)
        # Knight ↔ statement is true.   Equivalent: vars[speaker] == truth
        s.add(vars[speaker] == truth)
    return s, vars

def verify_puzzle(statements: dict[str, Statement], assignment: dict[str, str], mode: Literal["generation", "reward"] = "generation") -> bool:
    """Check whether `assignment` is consistent with the puzzle. Returns False on timeout."""
    s, vars = _build_constraints(statements)
    s.set("timeout", _TIMEOUT_MS[mode])
    for lab, identity in assignment.items():
        s.add(vars[lab] == (identity == "knight"))
    result = s.check()
    if result == z3.unknown:
        return False  # treat timeout as incorrect (see spec §5.1)
    return result == z3.sat

def count_solutions(statements: dict[str, Statement], n: int, timeout_ms: int = 5000, cap: int = 2) -> int:
    """Return number of distinct satisfying assignments, capped at `cap` (we only care
    whether the count is 0, 1, or >1)."""
    s, vars = _build_constraints(statements)
    s.set("timeout", timeout_ms)
    found = 0
    while found < cap + 1:
        result = s.check()
        if result == z3.unknown:
            raise VerifierTimeout()
        if result == z3.unsat:
            return found
        found += 1
        model = s.model()
        # Block this exact assignment.
        block = z3.Or([vars[lab] != model.eval(vars[lab]) for lab in vars])
        s.add(block)
    return found  # > cap
```

- [ ] **Step 4: Run tests**

```bash
pytest train/test_common.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add train/common.py train/test_common.py
git commit -m "feat(train): z3 SAT verifier with dual timeout modes"
```

---

## Task 4: extract_answer + check_format in common.py

**Files:**
- Modify: `train/common.py` (append functions)
- Create: `train/test_extract.py`

- [ ] **Step 1: Write the failing tests**

`train/test_extract.py`:
```python
import pytest
from train.common import extract_answer, check_format

# === check_format ===
def test_format_well_formed():
    assert check_format("<think>foo</think><answer>A: knight</answer>") is True

def test_format_missing_think():
    assert check_format("<answer>A: knight</answer>") is False

def test_format_missing_answer():
    assert check_format("<think>foo</think>") is False

# === extract_answer strict ===
def test_strict_two_people():
    r = "<answer>A: knight, B: knave</answer>"
    assert extract_answer(r, n=2) == {"A": "knight", "B": "knave"}

def test_strict_three_people():
    r = "<think>...</think><answer>A: knave, B: knight, C: knave</answer>"
    assert extract_answer(r, n=3) == {"A": "knave", "B": "knight", "C": "knave"}

def test_strict_extra_whitespace():
    r = "<answer>  A : knight ,  B : knave  </answer>"
    assert extract_answer(r, n=2) == {"A": "knight", "B": "knave"}

# === fallbacks ===
def test_fallback_case_insensitive():
    r = "<answer>A: KNIGHT, B: Knave</answer>"
    assert extract_answer(r, n=2) == {"A": "knight", "B": "knave"}

def test_fallback_is_a_separator():
    r = "<answer>A is a knight, B is a knave</answer>"
    assert extract_answer(r, n=2) == {"A": "knight", "B": "knave"}

def test_fallback_equals_separator():
    r = "<answer>A=knight, B=knave</answer>"
    assert extract_answer(r, n=2) == {"A": "knight", "B": "knave"}

def test_fallback_newline_separator():
    r = "<answer>A: knight\nB: knave</answer>"
    assert extract_answer(r, n=2) == {"A": "knight", "B": "knave"}

def test_fallback_no_tags_uses_tail():
    r = "Long reasoning... finally I conclude: A: knight, B: knave."
    assert extract_answer(r, n=2) == {"A": "knight", "B": "knave"}

# === failures (return None) ===
def test_partial_missing_person():
    r = "<answer>A: knight</answer>"
    assert extract_answer(r, n=2) is None

def test_duplicate_person():
    r = "<answer>A: knight, A: knave</answer>"
    assert extract_answer(r, n=2) is None

def test_garbage():
    assert extract_answer("hello world", n=3) is None

def test_hack_empty_answer():
    assert extract_answer("<answer></answer>", n=2) is None
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest train/test_extract.py -v
```
Expected: ImportError.

- [ ] **Step 3: Append `extract_answer` and `check_format` to `train/common.py`**

Append to existing file:
```python
import re

def check_format(response: str) -> bool:
    """Canonical 'well-formed output' predicate. Used by reward function and SFT
    format-compliance metric so the two never disagree (spec §6.2)."""
    has_think = "<think>" in response and "</think>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    return has_think and has_answer

_STRICT_RE = re.compile(
    r"<answer>\s*([A-Z]\s*:\s*(?:knight|knave)(?:\s*,\s*[A-Z]\s*:\s*(?:knight|knave))*)\s*</answer>",
    re.IGNORECASE,
)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_PAIR_RE = re.compile(
    r"([A-Z])\s*(?::|=|→|\bis\s+a\b)\s*(knight|knave)",
    re.IGNORECASE,
)

def _parse_pairs(text: str, n: int) -> dict[str, str] | None:
    seen: dict[str, str] = {}
    for m in _PAIR_RE.finditer(text):
        label = m.group(1).upper()
        identity = m.group(2).lower()
        if label in seen:
            return None  # duplicate
        seen[label] = identity
    if len(seen) != n:
        return None
    expected_labels = set(chr(ord("A") + i) for i in range(n))
    if set(seen.keys()) != expected_labels:
        return None
    return seen

def extract_answer(response: str, n: int) -> dict[str, str] | None:
    """Extract identity assignment from a model response.
    Strict-first, with case-insensitive + alt-separator fallbacks (spec §6.1)."""
    # Strict pattern attempt
    m = _STRICT_RE.search(response)
    if m:
        body = m.group(1)
        parsed = _parse_pairs(body, n)
        if parsed is not None:
            return parsed
    # Fallback 1: relaxed parse inside <answer> tags
    block_match = _ANSWER_BLOCK_RE.search(response)
    if block_match:
        parsed = _parse_pairs(block_match.group(1), n)
        if parsed is not None:
            return parsed
    # Fallback 2: tail of response (last 200 chars) when no answer tags
    tail = response[-200:]
    parsed = _parse_pairs(tail, n)
    if parsed is not None:
        return parsed
    return None
```

- [ ] **Step 4: Run tests**

```bash
pytest train/test_extract.py -v
```
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add train/common.py train/test_extract.py
git commit -m "feat(train): extract_answer + check_format shared parsers"
```

---

## Task 5: Eval + dev set generator (frozen first)

**Files:**
- Create: `data/gen_eval_data.py`
- Test: append to `data/test_gen_puzzles.py`

This script MUST run before any training-data script (spec §5.5). It produces both `eval_data.jsonl` (1800 puzzles, 300 per n=2..7) and `dev_data.jsonl` (200 puzzles, 40 per n=2..6), AND `eval_hashes.json` containing every puzzle's hash.

- [ ] **Step 1: Write a failing integration test**

Append to `data/test_gen_puzzles.py`:
```python
import json
import subprocess
from pathlib import Path
import hashlib

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
    assert len(eval_recs) == 1800
    assert len(dev_recs) == 200
    assert len(hashes) == 2000  # all eval + dev hashes
    # Per-bucket counts
    from collections import Counter
    eval_buckets = Counter(len(r["ground_truth"]) for r in eval_recs)
    assert all(eval_buckets[n] == 300 for n in range(2, 8))
    dev_buckets = Counter(len(r["ground_truth"]) for r in dev_recs)
    assert all(dev_buckets[n] == 40 for n in range(2, 7))
```

(This test is slow — ~1 min — so mark with `@pytest.mark.slow` if running often.)

- [ ] **Step 2: Run, confirm failure**

```bash
pytest data/test_gen_puzzles.py::test_gen_eval_produces_correct_counts -v
```
Expected: subprocess fails (script doesn't exist).

- [ ] **Step 3: Implement `data/gen_eval_data.py`**

```python
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

EVAL_PER_N = 300
DEV_PER_N = 40
EVAL_NS = range(2, 8)  # n=2..7
DEV_NS = range(2, 7)   # n=2..6

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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taken = set()
    eval_records = []
    for n in EVAL_NS:
        print(f"[eval] generating n={n}...", file=sys.stderr)
        eval_records.extend(collect(n, EVAL_PER_N, EVAL_SEED_START + n * 100_000, taken))
    dev_records = []
    for n in DEV_NS:
        print(f"[dev]  generating n={n}...", file=sys.stderr)
        dev_records.extend(collect(n, DEV_PER_N, DEV_SEED_START + n * 100_000, taken))
    (out_dir / "eval_data.jsonl").write_text("\n".join(json.dumps(r) for r in eval_records))
    (out_dir / "dev_data.jsonl").write_text("\n".join(json.dumps(r) for r in dev_records))
    (out_dir / "eval_hashes.json").write_text(json.dumps(sorted(taken)))
    print(f"Wrote {len(eval_records)} eval + {len(dev_records)} dev, total {len(taken)} hashes.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the integration test**

```bash
pytest data/test_gen_puzzles.py::test_gen_eval_produces_correct_counts -v
```
Expected: 1 passed (after ~1 min).

- [ ] **Step 5: Actually run the script and check the artefacts**

```bash
python -m data.gen_eval_data
ls -l data/*.jsonl data/eval_hashes.json
```
Expected: 3 files; eval_data.jsonl ~1MB, dev_data.jsonl ~100KB.

- [ ] **Step 6: Commit (artefacts gitignored, only the script)**

```bash
git add data/gen_eval_data.py data/test_gen_puzzles.py
git commit -m "feat(data): eval(1800) + dev(200) generator with frozen hashes"
```

---

## Task 6: SFT data generator (DeepSeek + per-n quotas)

**Files:**
- Create: `data/gen_sft_data.py`
- Modify: `.env.example`

- [ ] **Step 1: Add API key template**

Create `.env.example`:
```
DEEPSEEK_API_KEY=sk-replace-me
```
Add `.env` to `.gitignore` if not already there. Tell engineer: `cp .env.example .env` then put real key in `.env`.

- [ ] **Step 2: Write `data/gen_sft_data.py`**

```python
"""Generate SFT training examples by asking DeepSeek-V3 to solve K&K puzzles,
keeping only those whose answer matches ground truth.
Per-n quota table (spec §5.2): 250/250/500/500/500 verified for n=2..6."""
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

TARGETS = {2: 250, 3: 250, 4: 500, 5: 500, 6: 500}

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
    out_path.write_text("")  # truncate

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
```

- [ ] **Step 3: Smoke test with tiny quotas first (don't burn the budget)**

Temporarily edit TARGETS to `{2: 5, 3: 5}` and run:
```bash
python -m data.gen_sft_data --out data/sft_data_smoke.jsonl
wc -l data/sft_data_smoke.jsonl
```
Expected: ~10 lines. Inspect one record manually:
```bash
head -1 data/sft_data_smoke.jsonl | python -m json.tool
```
Confirm structure: `puzzle`, `completion` (containing `<think>` + `<answer>`), `n`, `hash`.

- [ ] **Step 4: Restore the real TARGETS dict and run the full generation**

```bash
python -m data.gen_sft_data --retry-once
wc -l data/sft_data.jsonl
```
Expected: 2000 lines (may take 1-2 hours; rate-limited by DeepSeek API).

- [ ] **Step 5: Verify per-n distribution**

```bash
python -c "
import json
from collections import Counter
recs = [json.loads(l) for l in open('data/sft_data.jsonl')]
print(Counter(r['n'] for r in recs))
"
```
Expected: `Counter({4: 500, 5: 500, 6: 500, 2: 250, 3: 250})`.

- [ ] **Step 6: Commit**

```bash
git add data/gen_sft_data.py .env.example
git commit -m "feat(data): DeepSeek-driven SFT data generator with per-n quotas"
```

---

## Task 7: SFT training

**Files:**
- Create: `train/sft.py`, `notebooks/01_sft.ipynb` (light wrapper)

- [ ] **Step 1: Add model-loading helpers to `train/common.py`**

Append to `train/common.py`:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def load_base_model(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    return model, tok

def make_lora_config(r: int = 16, alpha: int = 32):
    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
```

- [ ] **Step 2: Write `train/sft.py`**

```python
"""SFT trainer: LoRA-fit Qwen2.5-1.5B on K&K puzzle→completion pairs.
Selects best checkpoint by dev-set accuracy (spec §4)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from train.common import load_base_model, make_lora_config, extract_answer

def to_chat(tokenizer, puzzle: str, completion: str | None = None) -> str:
    messages = [{"role": "user", "content": puzzle}]
    if completion is not None:
        messages.append({"role": "assistant", "content": completion})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(completion is None))

class DevAccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer, dev_records: list[dict], out_dir: Path):
        self.tok = tokenizer
        self.dev = dev_records
        self.out_dir = out_dir
        self.best_acc = -1.0
        self.best_epoch = -1

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model.eval()
        correct = 0
        with torch.no_grad():
            for rec in self.dev:
                prompt = to_chat(self.tok, rec["puzzle"])
                inputs = self.tok(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    **inputs, max_new_tokens=400, do_sample=False, temperature=0.0,
                    pad_token_id=self.tok.eos_token_id,
                )
                resp = self.tok.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pred = extract_answer(resp, n=len(rec["ground_truth"]))
                if pred == rec["ground_truth"]:
                    correct += 1
        acc = correct / len(self.dev)
        print(f"[epoch {state.epoch:.0f}] dev_acc={acc:.3f}", flush=True)
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = int(state.epoch)
            model.save_pretrained(self.out_dir / "best")
            (self.out_dir / "best" / "dev_acc.json").write_text(json.dumps({"acc": acc, "epoch": self.best_epoch}))
        model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", default="data/sft_data.jsonl")
    parser.add_argument("--dev-data", default="data/dev_data.jsonl")
    parser.add_argument("--out-dir", default="results/checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_base_model()
    model = get_peft_model(model, make_lora_config())

    sft_records = [json.loads(l) for l in open(args.sft_data)]
    dev_records = [json.loads(l) for l in open(args.dev_data)]

    def to_record(r):
        return {"text": to_chat(tok, r["puzzle"], r["completion"])}

    dataset = Dataset.from_list([to_record(r) for r in sft_records])

    cfg = SFTConfig(
        output_dir=str(out_dir / "raw"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="no",  # we save ourselves in the callback
        max_seq_length=args.max_seq_length,
        report_to="wandb",
        run_name="logic-zero-sft",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        tokenizer=tok,
        callbacks=[DevAccuracyCallback(tok, dev_records, out_dir)],
    )
    trainer.train()
    print(f"Best dev_acc={trainer.callback_handler.callbacks[-1].best_acc:.3f} at epoch {trainer.callback_handler.callbacks[-1].best_epoch}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run a 10-step smoke test before committing the full run**

```bash
python -m train.sft --sft-data data/sft_data_smoke.jsonl --epochs 1 --out-dir results/checkpoints/sft_smoke
```
Expected: trains for a few minutes on A100/L4, no crashes, dev_acc printed.

- [ ] **Step 4: Run the real SFT training**

```bash
python -m train.sft --out-dir results/checkpoints/sft
```
Expected: ~2h on A100. Final log: `Best dev_acc=0.XX at epoch N`. Should be ≥ 0.30 per spec §2.

- [ ] **Step 5: Verify the best checkpoint loads**

```bash
python -c "
from peft import PeftModel
from train.common import load_base_model
model, tok = load_base_model()
model = PeftModel.from_pretrained(model, 'results/checkpoints/sft/best')
print(model.peft_config)
"
```
Expected: no errors; prints LoRA config.

- [ ] **Step 6: Commit code (not checkpoints)**

```bash
git add train/sft.py train/common.py
git commit -m "feat(train): SFT trainer with dev-set checkpoint selection"
```

---

## Task 8: Eval harness (greedy + 3-seed sampled)

**Files:**
- Create: `eval/run_eval.py`

- [ ] **Step 1: Write `eval/run_eval.py`**

```python
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
from train.sft import to_chat

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
```

- [ ] **Step 2: Smoke test on 10 puzzles**

```bash
head -10 data/eval_data.jsonl > data/eval_smoke.jsonl
python -m eval.run_eval --eval-data data/eval_smoke.jsonl --out results/smoke_base.json
cat results/smoke_base.json
```
Expected: valid JSON with greedy + 3 sampled passes, all 10 puzzles accounted for.

- [ ] **Step 3: Run full eval on Base model (no adapter)**

```bash
python -m eval.run_eval --out results/eval_base.json
```
Expected: ~1.5h on A100. JSON contains per-bucket counts for n=2..7.

- [ ] **Step 4: Run full eval on SFT checkpoint**

```bash
python -m eval.run_eval --adapter results/checkpoints/sft/best --out results/eval_sft.json
```
Expected: ~1.5h. SFT accuracy should beat base by ≥10 pp aggregated (spec §2).

- [ ] **Step 5: Commit**

```bash
git add eval/run_eval.py
git commit -m "feat(eval): greedy + 3-seed sampled eval harness with per-bucket reporting"
```

---

## Task 9: External baselines (prompted-Qwen + DeepSeek-V3)

**Files:**
- Create: `eval/baselines.py`

- [ ] **Step 1: Write `eval/baselines.py`**

```python
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
from train.sft import to_chat
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
```

- [ ] **Step 2: Smoke test both baselines on 10 puzzles**

```bash
python -m eval.baselines --mode qwen --out results/smoke_qwen.json --eval-data data/eval_smoke.jsonl
python -m eval.baselines --mode deepseek --out results/smoke_deepseek.json --eval-data data/eval_smoke.jsonl
```
Expected: both produce valid JSON, DeepSeek's accuracy should be visibly higher than prompted-Qwen.

- [ ] **Step 3: Run full baselines**

```bash
python -m eval.baselines --mode qwen --out results/eval_baseline_qwen.json
python -m eval.baselines --mode deepseek --out results/eval_baseline_deepseek.json
```
Expected: Qwen ~1h locally, DeepSeek ~30-60min via API (cost ~$7-8 per spec budget).

- [ ] **Step 4: Commit**

```bash
git add eval/baselines.py
git commit -m "feat(eval): prompted-Qwen + DeepSeek-V3 external baselines"
```

---

## Task 10: Sanity check (first comparison report at this point)

This is a checkpoint task — no new file, just stop and verify the project is on track before investing in DPO/GRPO.

- [ ] **Step 1: Aggregate accuracy from eval JSON files**

```bash
python -c "
import json
for label, path in [
    ('Base', 'results/eval_base.json'),
    ('SFT', 'results/eval_sft.json'),
    ('Qwen+CoT', 'results/eval_baseline_qwen.json'),
    ('DeepSeek-V3', 'results/eval_baseline_deepseek.json'),
]:
    d = json.load(open(path))['greedy']
    c, t = sum(d['per_bucket_correct'].values()), sum(d['per_bucket_total'].values())
    print(f'{label:15s} {c}/{t} = {c/t:.1%}  (format {d[\"format_compliance\"]:.1%})')
"
```
Expected output:
- Base: ~5-15%
- SFT: ≥30% (spec target)
- Qwen+CoT: ~10-25%
- DeepSeek-V3: ~50-75%

- [ ] **Step 2: Decision gate**

If SFT accuracy < 25%, investigate before proceeding:
- Is DeepSeek's reasoning actually being learned? Inspect SFT data quality.
- Is the dev-set callback picking the right checkpoint?
- Try doubling SFT epochs to 5.
**Do not proceed to Task 11 if SFT < 25%.**

- [ ] **Step 3: Commit the eval JSON files**

```bash
git add results/eval_*.json
git commit -m "results: Base + SFT + external baselines on 1800-puzzle eval set"
```

---

## Task 11: DPO data construction

**Files:**
- Create: `data/gen_dpo_data.py`

- [ ] **Step 1: Write `data/gen_dpo_data.py`**

```python
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

from train.common import load_base_model, extract_answer
from train.sft import to_chat
from data.gen_puzzles import generate_puzzle
from train.common import count_solutions, VerifierTimeout
from data.gen_eval_data import hash_puzzle, TRAIN_SEED_START

PER_N_RAW = {2: 250, 3: 250, 4: 500, 5: 500, 6: 500}
MIN_PAIRS_PER_N = 100

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
            if len(pairs_by_n[n]) >= MIN_PAIRS_PER_N:
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
        if all(len(pairs_by_n[n]) >= MIN_PAIRS_PER_N for n in PER_N_RAW):
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
```

- [ ] **Step 2: Smoke test (set targets to 5 per bucket temporarily)**

Edit `PER_N_RAW` to `{2: 5, 3: 5, 4: 5, 5: 5, 6: 5}` and `MIN_PAIRS_PER_N` to 1; run:
```bash
python -m data.gen_dpo_data --out data/dpo_smoke.jsonl
head -1 data/dpo_smoke.jsonl | python -m json.tool
```
Expected: at least 1 pair per bucket; record has `prompt`, `chosen`, `rejected`, `n`, `hash`.

- [ ] **Step 3: Restore real targets and run full generation**

```bash
python -m data.gen_dpo_data
wc -l data/dpo_data.jsonl
```
Expected: 800-1000 lines, 5-10 hours wall-clock (SFT sampling 4 responses per puzzle is the bottleneck).

- [ ] **Step 4: Verify stratification**

```bash
python -c "
import json
from collections import Counter
recs = [json.loads(l) for l in open('data/dpo_data.jsonl')]
print(Counter(r['n'] for r in recs))
"
```
Expected: each n=2..6 has ≥100 pairs; document any gaps (spec §5.3 mitigation).

- [ ] **Step 5: Commit**

```bash
git add data/gen_dpo_data.py
git commit -m "feat(data): DPO data builder with stratified per-n floor"
```

---

## Task 12: DPO training

**Files:**
- Create: `train/dpo.py`

- [ ] **Step 1: Write `train/dpo.py`**

```python
"""DPO trainer: continue training the SFT LoRA on (chosen, rejected) preference pairs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

from train.common import load_base_model
from train.sft import to_chat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-adapter", default="results/checkpoints/sft/best")
    parser.add_argument("--dpo-data", default="data/dpo_data.jsonl")
    parser.add_argument("--out-dir", default="results/checkpoints/dpo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_base_model()
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    ref_model, _ = load_base_model()
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_adapter)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    records = [json.loads(l) for l in open(args.dpo_data)]
    def to_dpo(r):
        prompt_chat = to_chat(tok, r["prompt"])  # already adds generation prompt
        return {"prompt": prompt_chat, "chosen": r["chosen"], "rejected": r["rejected"]}
    dataset = Dataset.from_list([to_dpo(r) for r in records])

    cfg = DPOConfig(
        output_dir=str(out_dir / "raw"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        report_to="wandb",
        run_name="logic-zero-dpo",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=dataset,
        tokenizer=tok,
    )
    trainer.train()
    model.save_pretrained(out_dir / "best")
    print(f"DPO saved to {out_dir/'best'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run DPO training**

```bash
python -m train.dpo
```
Expected: ~1.5h on A100. Watch wandb for `rewards/accuracies` rising to ≥0.7.

- [ ] **Step 3: Eval DPO checkpoint**

```bash
python -m eval.run_eval --adapter results/checkpoints/dpo/best --out results/eval_dpo.json
```
Expected: aggregated accuracy ≥ SFT + 10 pp (spec §2).

- [ ] **Step 4: Commit**

```bash
git add train/dpo.py results/eval_dpo.json
git commit -m "feat(train): DPO trainer; eval shows +N pp over SFT"
```

---

## Task 13: GRPO data + trl LoRA-as-reference smoke test

**Files:**
- Create: `data/gen_grpo_data.py`
- Create: `train/test_grpo_smoke.py` (open question #5 from spec §11)

- [ ] **Step 1: Write `data/gen_grpo_data.py`**

```python
"""GRPO data: puzzles + ground truth, no labels needed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from data.gen_puzzles import generate_puzzle
from train.common import count_solutions, VerifierTimeout
from data.gen_eval_data import hash_puzzle, TRAIN_SEED_START

TARGETS = {2: 250, 3: 250, 4: 500, 5: 500, 6: 500}  # 1:1:2:2:2 = 2000

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
```

Run it:
```bash
python -m data.gen_grpo_data
wc -l data/grpo_data.jsonl
```
Expected: 2000 lines, ~10 min.

- [ ] **Step 2: Write the GRPO LoRA-as-reference smoke test**

`train/test_grpo_smoke.py`:
```python
"""Verify that trl 0.13's GRPOTrainer works with a LoRA adapter as the reference policy
sharing the base model (spec §11 open question #5).
If this fails, fall back per spec to merging the reference adapter or shipping two models."""
import torch
from peft import PeftModel, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from train.common import load_base_model, make_lora_config

def trivial_reward(prompts, completions, **kwargs):
    return [float(len(c) > 10) for c in completions]

def test_grpo_with_lora_reference_does_not_oom(tmp_path):
    model, tok = load_base_model()
    model = get_peft_model(model, make_lora_config())

    # Reference: a second LoRA adapter on top of the same base model.
    # trl 0.13 either supports passing a PeftModel as ref_model or uses PEFT's
    # "disable adapter" context as the reference.
    dummy_data = Dataset.from_list([{"prompt": "hi"} for _ in range(4)])

    cfg = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=32,
        learning_rate=1e-6,
        logging_steps=1,
        max_steps=1,  # one step is enough to verify
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=trivial_reward,
        args=cfg,
        train_dataset=dummy_data,
        tokenizer=tok,
    )
    trainer.train()
    # If we got here without OOM, the LoRA-as-ref pattern works.
```

- [ ] **Step 3: Run the smoke test on the target GPU (Colab L4 minimum)**

```bash
pytest train/test_grpo_smoke.py -v
```
Expected: passes. If it OOMs or errors on ref-model handling, **stop and apply spec §11 Q#5 fallback** before continuing to Task 16.

- [ ] **Step 4: Commit**

```bash
git add data/gen_grpo_data.py train/test_grpo_smoke.py
git commit -m "feat(data,train): GRPO data + trl LoRA-as-ref smoke test"
```

---

## Task 14: Reward function

**Files:**
- Create: `train/reward.py`
- Test: append to `train/test_common.py`

- [ ] **Step 1: Write reward tests**

Append to `train/test_common.py`:
```python
from train.reward import compute_reward

def test_reward_correct_full():
    gt = {"A": "knight", "B": "knave"}
    response = "<think>" + "x" * 100 + "</think><answer>A: knight, B: knave</answer>"
    assert compute_reward(response, gt) == 0.5 + 2.0 + 0.3  # format + correct + length

def test_reward_wrong_answer_no_correctness():
    gt = {"A": "knight", "B": "knave"}
    response = "<think>" + "x" * 100 + "</think><answer>A: knave, B: knave</answer>"
    assert compute_reward(response, gt) == 0.5 + 0.3  # format + length only

def test_reward_no_format_no_length():
    gt = {"A": "knight"}
    assert compute_reward("knight", gt) == 0.0  # no tags, response too short

def test_reward_only_format():
    gt = {"A": "knight"}
    response = "<think></think><answer></answer>"  # short, no answer, has tags
    assert compute_reward(response, gt) == 0.5  # format only
```

- [ ] **Step 2: Write `train/reward.py`**

```python
"""Rule-based reward for GRPO (spec §6)."""
from __future__ import annotations
from train.common import extract_answer, check_format

def compute_reward(response: str, ground_truth: dict) -> float:
    score = 0.0
    if check_format(response):
        score += 0.5
    n = len(ground_truth)
    pred = extract_answer(response, n=n)
    if pred == ground_truth:
        score += 2.0
    if len(response) > 80:
        score += 0.3
    return score

def grpo_reward_funcs(prompts, completions, ground_truth, **kwargs):
    """trl 0.13 reward_funcs signature: lists in, list of floats out.
    `ground_truth` is forwarded from the dataset by trl when present as a column."""
    return [compute_reward(c, gt) for c, gt in zip(completions, ground_truth)]
```

- [ ] **Step 3: Run tests**

```bash
pytest train/test_common.py -v
```
Expected: all previous + 4 new tests pass.

- [ ] **Step 4: Commit**

```bash
git add train/reward.py train/test_common.py
git commit -m "feat(train): rule-based reward function with shared parsers"
```

---

## Task 15: GRPO main training

**Files:**
- Create: `train/grpo.py`

- [ ] **Step 1: Write `train/grpo.py`**

```python
"""GRPO trainer. Handles both the main pipeline (ref = DPO) and ablation (ref = SFT)
via the --ref-adapter flag (spec §4 ablation branch)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

from train.common import load_base_model
from train.sft import to_chat
from train.reward import grpo_reward_funcs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-adapter", required=True, help="LoRA to begin training from (DPO ckpt for main, SFT ckpt for ablation)")
    parser.add_argument("--ref-adapter", required=True, help="Frozen LoRA used as reference for KL (same as --start-adapter typically)")
    parser.add_argument("--grpo-data", default="data/grpo_data.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--run-name", default="logic-zero-grpo-main")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_base_model()
    model = PeftModel.from_pretrained(model, args.start_adapter, is_trainable=True)
    ref_model, _ = load_base_model()
    ref_model = PeftModel.from_pretrained(ref_model, args.ref_adapter)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    records = [json.loads(l) for l in open(args.grpo_data)]
    dataset = Dataset.from_list([
        {"prompt": to_chat(tok, r["prompt"]), "ground_truth": r["ground_truth"]}
        for r in records
    ])

    cfg = GRPOConfig(
        output_dir=str(out_dir / "raw"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        run_name=args.run_name,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_funcs,
        args=cfg,
        train_dataset=dataset,
        tokenizer=tok,
    )
    trainer.train()
    model.save_pretrained(out_dir / "best")
    print(f"GRPO saved to {out_dir/'best'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run main GRPO (from DPO ckpt, DPO as ref)**

```bash
python -m train.grpo \
  --start-adapter results/checkpoints/dpo/best \
  --ref-adapter results/checkpoints/dpo/best \
  --out-dir results/checkpoints/grpo_main \
  --run-name logic-zero-grpo-main
```
Expected: ~6h on A100. Monitor wandb: reward should climb; if accuracy on dev drops while reward rises, expect reward-hacking debugging.

- [ ] **Step 3: Reward-hacking spot check**

Watch wandb panel:
- `train/reward_mean` rising over time
- `train/completions_length_mean` not collapsing to ~30 chars (would mean empty think hack)
- `train/diversity` (or use a side script to sample 20 responses to dev puzzles and inspect manually mid-training)

If a reward-hacking pattern appears (e.g., constant "all-knights" output), apply spec §6 mitigation: add `len(extract_think_block(response)) > 30` clause to reward function and re-run.

- [ ] **Step 4: Eval main GRPO**

```bash
python -m eval.run_eval --adapter results/checkpoints/grpo_main/best --out results/eval_grpo_main.json
```
Expected: aggregated accuracy ≥ DPO + 15 pp (spec target ≥65% if SFT baseline was ~30%).

- [ ] **Step 5: Commit**

```bash
git add train/grpo.py results/eval_grpo_main.json
git commit -m "feat(train): main GRPO (SFT→DPO→GRPO) + eval"
```

---

## Task 16: GRPO ablation (SFT→GRPO, skip DPO)

**Files:**
- (No new file; reuse `train/grpo.py`)

- [ ] **Step 1: Run ablation GRPO from SFT checkpoint, SFT as reference**

```bash
python -m train.grpo \
  --start-adapter results/checkpoints/sft/best \
  --ref-adapter results/checkpoints/sft/best \
  --out-dir results/checkpoints/grpo_no_dpo \
  --run-name logic-zero-grpo-no-dpo
```
Expected: ~6h on A100. Same hyperparameters as main GRPO.

- [ ] **Step 2: Eval ablation**

```bash
python -m eval.run_eval --adapter results/checkpoints/grpo_no_dpo/best --out results/eval_grpo_no_dpo.json
```
Expected: comparable to main GRPO — within ±5 pp tells us DPO is redundant given GRPO (spec §2 ablation criterion).

- [ ] **Step 3: Commit**

```bash
git add results/eval_grpo_no_dpo.json
git commit -m "feat(train): SFT→GRPO ablation (skip DPO) + eval"
```

---

## Task 17: Comparison report + charts

**Files:**
- Create: `eval/compare.py`

- [ ] **Step 1: Write `eval/compare.py`**

```python
"""Aggregate every eval JSON into a comparison table + charts (spec §10)."""
from __future__ import annotations
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

SYSTEMS = [
    ("Base", "results/eval_base.json"),
    ("Qwen+CoT", "results/eval_baseline_qwen.json"),
    ("SFT", "results/eval_sft.json"),
    ("DPO", "results/eval_dpo.json"),
    ("GRPO (main)", "results/eval_grpo_main.json"),
    ("GRPO-no-DPO", "results/eval_grpo_no_dpo.json"),
    ("DeepSeek-V3", "results/eval_baseline_deepseek.json"),
]
BUCKETS = [2, 3, 4, 5, 6, 7]

def bucket_acc(passdata, n):
    c = passdata["per_bucket_correct"].get(str(n), 0)
    t = passdata["per_bucket_total"].get(str(n), 0)
    return c / t if t else 0.0

def main():
    table_rows = []
    chart_data = {}
    for label, path in SYSTEMS:
        if not Path(path).exists():
            print(f"missing {path}; skipping")
            continue
        d = json.loads(Path(path).read_text())
        greedy_per_bucket = [bucket_acc(d["greedy"], n) for n in BUCKETS]
        sampled_per_bucket = {n: [bucket_acc(s, n) for s in d.get("sampled", [])] for n in BUCKETS}
        row = [label]
        for n, g in zip(BUCKETS, greedy_per_bucket):
            sampled = sampled_per_bucket[n]
            if sampled:
                mean = statistics.mean(sampled)
                std = statistics.stdev(sampled) if len(sampled) > 1 else 0.0
                row.append(f"{g:.1%} / {mean:.1%}±{std:.1%}")
            else:
                row.append(f"{g:.1%}")
        # Aggregated in-distribution (n=2-6) and OOD (n=7)
        in_dist = sum(d["greedy"]["per_bucket_correct"].get(str(n), 0) for n in (2,3,4,5,6)) / \
                  max(1, sum(d["greedy"]["per_bucket_total"].get(str(n), 0) for n in (2,3,4,5,6)))
        ood = bucket_acc(d["greedy"], 7)
        row.extend([f"{in_dist:.1%}", f"{ood:.1%}"])
        table_rows.append(row)
        chart_data[label] = greedy_per_bucket

    header = ["System"] + [f"n={n}" for n in BUCKETS] + ["n=2-6 agg", "n=7 OOD"]
    md = ["# Logic-Zero Comparison\n",
          "| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    for row in table_rows:
        md.append("| " + " | ".join(row) + " |")
    Path("results/accuracy_table.md").write_text("\n".join(md))
    print("Wrote results/accuracy_table.md")

    # Chart: line plot of accuracy vs n
    plt.figure(figsize=(10, 6))
    for label, data in chart_data.items():
        plt.plot(BUCKETS, data, marker="o", label=label)
    plt.xlabel("n (inhabitants)")
    plt.ylabel("Accuracy (greedy)")
    plt.title("Accuracy vs puzzle size, by system")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("results/training_curves.png", dpi=120, bbox_inches="tight")
    print("Wrote results/training_curves.png")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python -m eval.compare
cat results/accuracy_table.md
```
Expected: markdown table with 7 systems × 6 buckets + 2 aggregate columns; PNG chart.

- [ ] **Step 3: Commit**

```bash
git add eval/compare.py results/accuracy_table.md results/training_curves.png
git commit -m "feat(eval): 7-system comparison table and accuracy-vs-n chart"
```

---

## Task 18: Qualitative response analysis

**Files:**
- Create: `eval/analyze_responses.py`

- [ ] **Step 1: Write the analyzer**

```python
"""Pull 10 example puzzles where systems disagree most; emit a markdown side-by-side."""
from __future__ import annotations
import json
from pathlib import Path
from peft import PeftModel
import torch

from train.common import load_base_model, extract_answer
from train.sft import to_chat

ADAPTERS = {
    "Base": None,
    "SFT": "results/checkpoints/sft/best",
    "DPO": "results/checkpoints/dpo/best",
    "GRPO": "results/checkpoints/grpo_main/best",
}

def generate(model, tok, prompt: str) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def main():
    eval_recs = [json.loads(l) for l in open("data/eval_data.jsonl")][:50]  # first 50 only
    samples = []
    base_model_pretrained = None
    for label, adapter in ADAPTERS.items():
        model, tok = load_base_model()
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        for i, rec in enumerate(eval_recs):
            prompt = to_chat(tok, rec["puzzle"])
            resp = generate(model, tok, prompt)
            samples.append({"i": i, "system": label, "response": resp, "gt": rec["ground_truth"], "n": len(rec["ground_truth"]), "puzzle": rec["puzzle"]})
        del model

    # Group by puzzle, find ones with most disagreement
    by_puzzle = {}
    for s in samples:
        by_puzzle.setdefault(s["i"], []).append(s)
    scored = []
    for i, group in by_puzzle.items():
        correct_set = tuple(extract_answer(g["response"], n=g["n"]) == g["gt"] for g in group)
        # Most interesting: 1 or 2 systems correct out of 4
        if 1 <= sum(correct_set) <= 2:
            scored.append((i, group))
    md = ["# Qualitative Response Comparisons\n"]
    for i, group in scored[:10]:
        md.append(f"## Puzzle #{i} (n={group[0]['n']})\n")
        md.append("```\n" + group[0]["puzzle"] + "\n```\n")
        md.append(f"**Ground truth:** `{group[0]['gt']}`\n")
        for g in group:
            correct = "✅" if extract_answer(g["response"], n=g["n"]) == g["gt"] else "❌"
            md.append(f"### {g['system']} {correct}\n```\n{g['response']}\n```\n")
    Path("results/example_responses.md").write_text("\n".join(md))
    print("Wrote results/example_responses.md")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python -m eval.analyze_responses
head -50 results/example_responses.md
```
Expected: 10 side-by-side comparisons where the systems split.

- [ ] **Step 3: Commit**

```bash
git add eval/analyze_responses.py results/example_responses.md
git commit -m "feat(eval): qualitative side-by-side response comparison"
```

---

## Task 19: HuggingFace push + README

**Files:**
- Modify: `README.md` (replace stub)

- [ ] **Step 1: Push LoRA adapters to HF Hub**

```bash
huggingface-cli login  # if not already
python -c "
from huggingface_hub import HfApi
api = HfApi()
for name, path in [
    ('logic-zero-sft', 'results/checkpoints/sft/best'),
    ('logic-zero-dpo', 'results/checkpoints/dpo/best'),
    ('logic-zero-grpo', 'results/checkpoints/grpo_main/best'),
    ('logic-zero-grpo-no-dpo', 'results/checkpoints/grpo_no_dpo/best'),
]:
    api.create_repo(repo_id=f'shengdeb/{name}', exist_ok=True)
    api.upload_folder(folder_path=path, repo_id=f'shengdeb/{name}')
    print(f'pushed {name}')
"
```
Expected: 4 repos created on HF Hub.

- [ ] **Step 2: Rewrite `README.md` with results**

```markdown
# Logic-Zero

Train Qwen2.5-1.5B-Instruct on Knights & Knaves logic puzzles through a full SFT → DPO → GRPO post-training pipeline. Includes a SFT → GRPO ablation that isolates DPO's contribution.

## Results

(paste contents of `results/accuracy_table.md` here)

![Accuracy vs puzzle size](results/training_curves.png)

**Headline:** [TODO: fill once eval runs complete, e.g. "GRPO reaches 67% greedy accuracy on n=2-6, vs 12% for base Qwen and 71% for DeepSeek-V3"]

## Repo layout

See [docs/plans/2026-05-19-logic-zero.md](docs/plans/2026-05-19-logic-zero.md).

## Reproducing

1. `pip install -r requirements.txt`
2. `cp .env.example .env`, fill in `DEEPSEEK_API_KEY`
3. `python -m data.gen_eval_data` (generates eval + dev first)
4. `python -m data.gen_sft_data --retry-once` (~$5 API)
5. `python -m train.sft`
6. `python -m eval.run_eval --adapter results/checkpoints/sft/best --out results/eval_sft.json`
7. ... (follow plan §11-19 for DPO, GRPO, ablation, comparison)

## Checkpoints on HuggingFace

- `shengdeb/logic-zero-sft`
- `shengdeb/logic-zero-dpo`
- `shengdeb/logic-zero-grpo`
- `shengdeb/logic-zero-grpo-no-dpo`

## Design

See [design spec](docs/specs/2026-05-18-logic-zero-design.md).
```

- [ ] **Step 3: Fill in the headline + paste the table**

Manually open `results/accuracy_table.md`, paste into README, fill in the headline with real numbers.

- [ ] **Step 4: Commit and push**

```bash
git add README.md
git commit -m "docs: README with final results + HF Hub links"
git push  # if a remote is configured
```

---

## Task 20: Blog post

**Files:**
- Create: `docs/blog/post.md`

- [ ] **Step 1: Outline the blog post**

Five sections:
1. **What and why** (200 words) — K&K, why this task, what SFT/DPO/GRPO do
2. **Pipeline at a glance** (300 words) — show the diagram from spec §4
3. **The reward hacking story** (400 words) — concrete incident from Task 15 step 3 (if any) or hypothetical with the spec's documented expectations
4. **DPO necessity ablation** (300 words) — main vs no-DPO numbers, take a position
5. **Lessons + what would I do differently** (300 words) — honest reflection

- [ ] **Step 2: Write the post in `docs/blog/post.md`**

(Author writes prose during Week 5; no template needed beyond the outline above.)

- [ ] **Step 3: Commit**

```bash
git add docs/blog/post.md
git commit -m "docs: blog post on Logic-Zero pipeline + ablation findings"
```

---

## Self-Review

**1. Spec coverage check.** Walked through spec sections:
- §2 success criteria → Tasks 8, 10, 17 produce the numbers; SFT≥30% gate at Task 10 Step 2 ✓
- §3 task definition → Task 2 generator ✓
- §4 architecture (SFT → DPO → GRPO + ablation) → Tasks 7, 12, 15, 16 ✓
- §4 pinned deps → Task 1 Step 2 ✓
- §5.1 puzzle generator + dual-mode SAT timeout → Tasks 2, 3 (modes: `generation`/`reward`) ✓
- §5.2 SFT data per-n quota → Task 6 TARGETS ✓
- §5.3 DPO data + stratified floor → Task 11 MIN_PAIRS_PER_N ✓
- §5.4 GRPO data 1:1:2:2:2 → Task 13 Step 1 TARGETS ✓
- §5.5 eval/dev frozen-first → Task 5 ✓
- §6 reward function → Task 14 ✓
- §6.1 extract_answer + tests → Task 4 ✓
- §6.2 check_format → Task 4 ✓
- §6.3 GRPO config (ref model, β, num_generations) → Task 15 args ✓
- §7 Colab constraints → notebooks called out as sidekicks (Tasks 7, 12, 15, 16 happen in their notebooks); checkpoint saving to results/ which a user can sync to Drive themselves ✓
- §10 deliverables (5 checkpoints, results page, blog, resume line) → Tasks 17, 18, 19, 20 ✓
- §11 open Q#5 (trl smoke test) → Task 13 Step 2-3 ✓

**2. Placeholder scan.** Searched for TBD/TODO/"implement later"/"add appropriate":
- README has a `[TODO: fill once eval runs complete]` — intentional placeholder for actual numbers; flagged in Task 19 step 3 as a manual fill ✓
- No other TBDs.

**3. Type/name consistency.**
- `extract_answer(response, n)` signature used everywhere consistently ✓
- `check_format(response)` consistent ✓
- `verify_puzzle(statements, assignment, mode)` matches between common.py and callers ✓
- `to_chat(tok, puzzle, completion=None)` consistent across SFT, DPO, GRPO, eval ✓
- Hash field `hash_puzzle(text) → 16-char hex` consistent ✓
- Seed ranges: eval starts at 1M, dev at 2M, SFT-train at 3M, DPO at 13M, GRPO at 23M — all disjoint ✓
- `--adapter` flag for run_eval, `--start-adapter` + `--ref-adapter` for grpo — distinct names because they mean different things ✓

**4. Scope.** Plan is for one coherent project (Logic-Zero). Not decomposable further without losing the pipeline narrative. Single plan is appropriate.

One thing worth calling out for the engineer: **Tasks 6, 11, 15, 16 are long-running (hours each)**. Schedule them in Colab sessions with checkpoint syncing to Drive. The plan does not embed Colab-specific orchestration (notebooks are listed but their content is "run the CLI command above") — keeping CLI as the source of truth makes the project reproducible outside Colab too.
