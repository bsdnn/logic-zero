# Logic-Zero · Design Spec

**Date:** 2026-05-18
**Author:** shengdeb@uci.edu
**Status:** Draft → Pending user review

---

## 1. Project Summary

Train **Qwen2.5-1.5B-Instruct** to solve **Knights & Knaves (K&K)** logic puzzles using a full three-stage post-training pipeline: **SFT → DPO → GRPO**.

The model learns to output in the form:
```
<think>step-by-step deduction</think>
<answer>P1: knight, P2: knave, ...</answer>
```

The project demonstrates end-to-end understanding of modern LLM post-training, with a differentiated task choice (logic puzzles, not GSM8K math) and quantified ablation across all three stages.

---

## 2. Goals & Success Criteria

### Primary Goals
1. Build a reproducible 3-stage post-training pipeline on a single Colab Pro GPU.
2. Show measurable improvement after each stage on a held-out test set.
3. Produce a resume-quality GitHub repo with clear README, training curves, and a comparison table.

### Quantitative Success Criteria

All targets measured on the 1800-puzzle held-out eval set (see §5.5), using **greedy decoding (T=0)** as the primary number. Sampled decoding (T=0.7, 3 seeds) is reported alongside as mean ± std to show stability. Multi-seed does **not** reduce binomial uncertainty (300 puzzles per bucket → ±5.6 pp at p=0.5 is the hard floor); per-bucket differences below 6 pp should not be claimed as significant.

| Stage | Primary metric | Target | Secondary metric |
|---|---|---|---|
| After SFT | Answer accuracy (greedy) | ≥ 30% | Format compliance ≥ 95% |
| After DPO | Accuracy improvement over SFT | ≥ +10 pp | — |
| After GRPO (main) | Accuracy improvement over DPO | ≥ +15 pp (overall ≥ 65%, **provisional — revise after Week 1 SFT measurement**) | — |
| Ablation: GRPO-no-DPO | Accuracy vs main GRPO | Within ±5 pp = "DPO redundant" finding | — |
| Generalization | Accuracy on n=7 (trained on n=2-6) | ≥ 30% | — |

Format compliance is a sanity check, not the success bar — a model can output
the right tags and still reason badly, so accuracy is the metric that matters
at every stage. The "65%" target is a planning placeholder based on TinyZero-style results; the actual reachable ceiling depends on what SFT baseline accuracy turns out to be, which we won't know until Week 1.

### External Baselines (for reference, not training targets)

To put the in-house improvements in context, the eval set is also scored against:

- **Qwen2.5-1.5B-Instruct + CoT prompt** (no training): shows what prompt
  engineering alone can achieve on the base model.
- **DeepSeek-V3 zero-shot**: shows what a strong frontier model gets — establishes
  the task difficulty ceiling.

Without these, claiming "GRPO added 25 pp" is meaningless — the reader can't tell
whether the model learned reasoning or just learned the answer format.

### Non-Goals
- Beating GPT-4 / Claude on reasoning benchmarks.
- Supporting other puzzle types (Einstein puzzles, Sudoku) in v1.
- Multi-GPU or distributed training.
- Deploying the model as a public service.

---

## 3. Task: Knights & Knaves

### Why this task
- **Unique solution** — every well-formed puzzle has exactly one assignment of knights/knaves; verifiable with a SAT solver.
- **Programmatic generation** — templates + random sampling yield unlimited puzzles. Difficulty controllable via `n` (number of inhabitants).
- **Small answer space** — each person is either knight or knave; trivial regex extraction.
- **True deductive reasoning** — no memorizable formulas, unlike arithmetic.
- **Differentiated** — GitHub has very few K&K post-training projects; most reasoning RL work targets GSM8K/MATH.

### Example
```
Puzzle: On an island, every inhabitant is either a knight (always tells truth)
or a knave (always lies). You meet 3 people: A, B, C.
- A says: "B is a knave."
- B says: "A and C are of the same kind."
- C says: "I am a knight."
Determine each person's identity.

Expected response:
<think>
Assume A is knight → B is knave. Then B's statement is false, so A and C
are of different kinds → C is knave. Check C's statement: "I am a knight" —
that would be true, but C is knave who must lie. Contradiction.
Assume A is knave → B is knight. Then B's statement is true: A and C same kind →
C is knave. Check C: claims to be knight; as knave this is a lie. Consistent.
</think>
<answer>A: knave, B: knight, C: knave</answer>
```

### Difficulty Levels
- **Easy:** n = 2-3 inhabitants
- **Medium:** n = 4-5
- **Hard:** n = 6
- **Out-of-distribution:** n = 7

Training data covers n = 2-6. Held-out evaluation primarily targets n = 2-6 (in-distribution), with n = 7 included to measure generalization to a harder regime.

Earlier draft used training n=2-5 with eval at n=8. n=8 was unrealistic — the answer space is 2^8 = 256 assignments, and a 1.5B model has effectively no chance of getting consistent reasoning at that depth. Pulling training up to n=6 and capping OOD at n=7 keeps the generalization claim defensible while still demonstrating real difficulty progression.

---

## 4. Architecture: Three-Stage Pipeline

```
Qwen2.5-1.5B-Instruct  (base, no training)
        │
        ▼  SFT (LoRA r=16, up to 3 epochs, lr=2e-4)
        │  Goal: learn output format + basic deduction
        │  Data: 2000 (puzzle, reasoning, answer) tuples, n=2-6
        │  Checkpoint selection: best dev-set accuracy (200-puzzle dev set, §5.5)
        │
   [SFT ckpt]
        │
        ▼  DPO (LoRA continued, 1-2 epochs, lr=5e-6, β=0.1)
        │  Goal: prefer correct over incorrect reasoning
        │  Data: ~1000 (prompt, chosen=correct, rejected=incorrect) pairs
        │
   [DPO ckpt]
        │
        ▼  GRPO (LoRA continued, 3 epochs, lr=1e-6, num_generations=4)
        │  Goal: maximize answer accuracy via rule-based reward
        │  Data: 2000 prompts only
        │  Reference policy: frozen SFT+DPO model; KL β = 0.04 (see §6.3)
        │
   [GRPO ckpt — main branch]

ABLATION BRANCH (runs in parallel to the main GRPO stage):
   [SFT ckpt]
        │
        ▼  GRPO-skip-DPO (same config as main GRPO, but reference = frozen SFT)
        │  Goal: test whether DPO actually adds value or is redundant given GRPO
        │
   [GRPO-no-DPO ckpt]

        ▼  Evaluation: Base / SFT / DPO / GRPO / GRPO-no-DPO + 2 external baselines
        │  (prompted-Qwen, DeepSeek-V3) on 1800 held-out puzzles
```

### Why the SFT→GRPO ablation

A frequent finding in R1-style work is that DPO contributes little once GRPO is in the pipeline — the RL stage subsumes the preference signal. Without this ablation, we can show the full pipeline works but cannot answer the obvious interview question: *"Is DPO doing anything you couldn't get from SFT+GRPO alone?"*

Adding a second GRPO run from the SFT checkpoint (same hyperparameters, same data, same reward) lets us directly compare:
- **SFT → DPO → GRPO** (main pipeline)
- **SFT → GRPO** (ablation, DPO skipped)

If the two end up within ~2 pp, DPO is decorative and the blog post says so honestly. If DPO meaningfully helps, we can explain why. Either outcome is a valuable result.

**Cost:** one additional GRPO run = +6 hours GPU. Worth it.

### Why LoRA throughout
- Memory fits on single A100 40GB (Colab Pro best case) and even L4 24GB (Colab Pro typical).
- Each stage's adapter is small (~30MB) — easy to ship to HuggingFace.
- Stages compose cleanly: SFT adapter → load and continue training for DPO → same for GRPO.

### Pinned Dependencies (critical for reproducibility)

`trl`'s GRPOTrainer API has changed substantially across 2024-2025 releases (reward function signature, generation kwargs, KL handling). Reproducibility requires a hard pin. `requirements.txt` will use:

```
torch==2.4.0
transformers==4.46.0
trl==0.13.0
peft==0.13.0
datasets==3.0.0
accelerate==1.0.0
bitsandbytes==0.44.0
z3-solver==4.13.0
wandb==0.18.0
openai==1.50.0   # for DeepSeek API (OpenAI-compatible)
```

If a newer trl release ships before Week 3 and substantially changes GRPO defaults, evaluate before upgrading — re-pinning is preferable to silently shifting training behavior mid-project.

---

## 5. Data Design

### 5.1 Puzzle Generator (offline, deterministic)

`data/gen_puzzles.py` — pure Python, no model needed.

```python
def generate_puzzle(n: int, seed: int) -> tuple[str, dict]:
    """
    Args:
        n: number of inhabitants (2-7)
        seed: RNG seed for reproducibility
    Returns:
        (puzzle_text, ground_truth_dict)
        ground_truth_dict: {"A": "knight", "B": "knave", ...}
    """
```

**Statement templates** (each inhabitant makes one):
- "X is a knight/knave"
- "X and Y are of the same/different kind"
- "At least k of us are knights/knaves"
- "I am a knight/knave"

**Validity check:** after random generation, run the SAT verifier (`z3-solver`) to ensure exactly one solution. Discard puzzles with 0 or >1 solution. The verifier is implemented once, in `train/common.py` (function `verify_puzzle(puzzle, assignment, mode)`), and imported by every caller — data generation scripts, DPO labeler, eval scorer, and the GRPO reward function. There is no separate `data/verify.py`; keeping a single implementation avoids two SAT solvers drifting apart.

**Timeout policy (two different settings depending on call site):**
- **Data generation path** (puzzle validity check, SFT verifier, DPO labeler, eval scoring): `solver.set("timeout", 5000)` — 5 seconds. Generous, because we run these offline and want to maximize valid puzzles. Timeouts get discarded and logged.
- **GRPO reward path** (called inside the training loop): `solver.set("timeout", 500)` — **0.5 seconds**. Reward 0 on timeout. Rationale: at 4 generations × batch 8 × 750 steps = **~24,000 reward calls per full training run** (~8,000 per epoch over 3 epochs). Even 1% of calls hitting a 5-second cap adds ~20 minutes of pure waiting per run; a malicious-looking output that triggers many timeouts could blow training time up by 5-10x. A 0.5-second timeout caps the absolute-worst-case (100% of calls timing out) at ~3.3 hours of verifier wait per run — realistically the wait will be a small fraction of that.

The verifier wrapper in `common.py` exposes both modes via a `mode={"generation","reward"}` kwarg. Reward-path timeouts are tracked in a counter and surfaced to wandb so we can spot a reward-hacking pattern that exploits SAT-solver slowness.

### 5.2 SFT Data (2000 examples)

Pipeline:
1. Generate **~3000 raw puzzles** via `gen_puzzles.py`, allocated per the per-n quota table below (designed so that after DeepSeek verification yields ~2000 examples in the target 1:1:2:2:2 ratio across n=2..6).
2. For each puzzle, prompt DeepSeek-V3 to produce a step-by-step solution.
3. Verify DeepSeek's answer against ground truth; discard mismatches.
4. If any n bucket falls short of its target after verification, generate more raw puzzles for that n only and retry.
5. Format as Qwen chat template.

**DeepSeek prompt template:**
```
Solve this Knights and Knaves puzzle. Show step-by-step reasoning inside
<think></think> tags, then give the final answer inside <answer></answer> tags
in the format "A: knight, B: knave, ...".

Puzzle: {puzzle}
```

**Per-n target distribution (after verification, total 2000):**

| n | Weight | Target verified | Expected DeepSeek yield | Raw puzzles to generate |
|---|---|---|---|---|
| 2 | 1 | 250 | ~90% | ~280 |
| 3 | 1 | 250 | ~90% | ~280 |
| 4 | 2 | 500 | ~70% | ~720 |
| 5 | 2 | 500 | ~70% | ~720 |
| 6 | 2 | 500 | ~50% | ~1000 |
| **Total** | | **2000** | — | **~3000** |

**Why per-n over-generation:** if we generate 3000 puzzles uniformly across difficulty and filter, the final distribution skews toward easy (n=2,3) because DeepSeek's failure rate is concentration at n=5,6. That would leave the model under-trained on the hard cases that matter most. Per-n quotas guarantee the final SFT data hits the intended 1:1:2:2:2 weighting.

**Cost:**
- ~3000 calls × ~$0.001/call (≈500 input + 300 output tokens) ≈ **$3-5**.
- If actual yield at n=5,6 falls below the table estimates after Week 1 measurement, retry failed n=5,6 puzzles once more before discarding. Worst-case cost cap: ~$8.

**Stored format** (`data/sft_data.jsonl`):
```json
{"prompt": "<puzzle>", "completion": "<think>...</think><answer>...</answer>"}
```

### 5.3 DPO Data (~1000 pairs)

Pipeline:
1. Take 2000 new puzzles, sampled in the same **1:1:2:2:2** ratio across n=2..6 as SFT (i.e., 250/250/500/500/500). Eval and dev hashes are excluded as usual.
2. For each, generate **4 responses** from the SFT model (temperature=0.8).
3. Use SAT solver to mark each response correct/incorrect.
4. **Only keep puzzles with at least one correct AND one incorrect response.** Form pair: chosen = random correct response, rejected = random incorrect response.
5. Discard "all correct" cases. The earlier draft picked shortest-correct vs longest-correct to teach brevity, but that conflates correctness and length preferences in a single training signal — and K&K specifically rewards complete reasoning chains, so it would hurt. The GRPO reward function handles only length sanity (a floor), not brevity preference.
6. Discard "all incorrect" cases.

**Expected yield:** the (≥1 correct AND ≥1 incorrect) constraint excludes puzzles the SFT model already always-solves or never-solves. Realistic yield ~40-50% → 800-1000 pairs from 2000 puzzles. If yield falls short, generate more puzzles rather than relaxing the constraint.

**Known difficulty-distribution skew (and mitigation):** the filter automatically over-represents puzzles where SFT is around 50% accurate — usually the medium-difficulty bucket — and *under*-represents both ends:
- At n=2, if SFT is ~80% accurate, most puzzles produce 4-of-4 correct → discarded.
- At n=6, if SFT is ~10% accurate, most puzzles produce 4-of-4 incorrect → discarded.

The resulting DPO pair distribution is **not** the input 1:1:2:2:2 — it skews toward whatever bucket SFT happens to land near 50% on (likely n=4 or n=5).

**Mitigation: stratified target.** After raw sampling, enforce a per-n floor on the final pair set — at least 100 pairs from each of n=2, 3, 4, 5, 6. If a bucket falls short, generate additional puzzles for that bucket and try again. This may waste some samples on easy/hard buckets but prevents the DPO step from being a "tune the middle, ignore the edges" pass.

If after 2x over-generation a bucket still cannot hit 100 pairs (e.g., SFT is 95% on n=2 — almost no "incorrect" responses to pair against), document the gap and proceed with what we have; do not artificially inject bad data.

**Stored format** (`data/dpo_data.jsonl`):
```json
{"prompt": "<puzzle>", "chosen": "<good response>", "rejected": "<bad response>"}
```

### 5.4 GRPO Data (2000 prompts, 3 epochs)

Pipeline:
1. Generate 2000 fresh puzzles in the **1:1:2:2:2** ratio across n=2..6 (same as SFT and DPO; eval/dev hashes excluded). No DeepSeek calls — GRPO only needs puzzle + ground truth.
2. Store puzzle + ground truth (reward function needs ground truth at runtime).

**Stored format** (`data/grpo_data.jsonl`):
```json
{"prompt": "<puzzle>", "ground_truth": {"A": "knight", ...}}
```

**Sizing rationale:** earlier draft used 5000 prompts × 1 epoch ≈ 625 gradient steps at batch 8. RL needs many more gradient updates than that — TinyZero and most R1 reproductions run several thousand. 2000 prompts × 3 epochs gives ~750 steps with each prompt seen 3 times, producing more stable gradient estimates than 5000-once. If reward curve is still climbing at the end of training, extend to 4-5 epochs rather than adding more prompts.

### 5.5 Evaluation Set (1800 hold-out) + Dev Set (200)

**Generation order (matters):** eval and dev sets are generated **first**, before any training data, by a single script `data/gen_eval_data.py` (one script keeps the hash bookkeeping in one place). Their combined puzzle-text hashes are frozen and persisted to `data/eval_hashes.json`. All subsequent data generation (SFT, DPO, GRPO) skips any puzzle whose hash is in that set. This guarantees eval and dev are true held-out distributions that training cannot influence. The earlier draft had this backwards (excluded training hashes from eval) — which would allow training to silently shape eval composition.

**Eval set composition:**
- 300 puzzles each at n = 2, 3, 4, 5, 6 (in-distribution) + 300 at n = 7 (OOD generalization). Total: 1800 puzzles.
- Stored as `data/eval_data.jsonl`.

**Dev set (200 puzzles):** separate from both training and eval (its own hash block). Used for picking the best SFT checkpoint, monitoring DPO/GRPO mid-training, and quick sanity checks. Composition: 40 puzzles at each of n = 2, 3, 4, 5, 6. Stored as `data/dev_data.jsonl`. Never used for reporting final numbers.

**Reporting protocol** (clarifying the earlier overstatement on multi-seed):
- For each trained model and external baseline, run eval twice:
  - **Greedy** (temperature=0): primary reported number — reproducible, deployment-realistic.
  - **Sampled** (temperature=0.7, 3 seeds): reported as mean ± std to show decoding stability.
- 3-seed averaging does **not** reduce the binomial uncertainty on the eval set. At p=0.5, the 95% Wilson interval on 300 samples is ±5.6 pp — that is the hard floor regardless of seeds. Multi-seed only reveals how much the model's output flips under decoding randomness.
- **Implication for success criteria:** the ≥10 pp stage-to-stage targets are above the noise floor for in-distribution buckets (n=2-6 aggregated → 1500 samples → ±2.5 pp CI). For per-bucket numbers (300 samples each), differences <6 pp should be reported but not claimed as significant.

---

## 6. Reward Function (GRPO)

Located in `train/reward.py`:

```python
def reward(prompt: str, response: str, ground_truth: dict) -> float:
    score = 0.0

    # 1. Format reward (small). Uses the same predicate as the SFT format-
    # compliance metric (§6.2) so that "format" means the same thing
    # everywhere in the project.
    if check_format(response):
        score += 0.5

    # 2. Correctness reward (binary; avoids rewarding random guessing).
    # Earlier draft gave fractional credit (correct/n), but for n=2 random
    # guessing already nets ~0.5 — the reward function would teach the model
    # to guess rather than reason. Binary is harsher but unambiguous.
    pred = extract_answer(response)
    if pred == ground_truth:
        score += 2.0

    # 3. Length floor (only — no upper bound).
    # Earlier draft used 80 < rlen < 600, but n=6 puzzles can legitimately
    # need 700-1000+ chars of full reasoning. An upper bound would either
    # punish correct n=6 solutions, or teach the model to truncate reasoning
    # on hard cases. Use floor only; the KL term and the correctness reward
    # together discourage runaway repetition.
    rlen = len(response)
    if rlen > 80:
        score += 0.3

    return score
```

**Anticipated reward hacking:**
- Model guesses "all knights" or "all knaves" → empty `<think>` block. Mitigation: add a "think block must be non-trivial (>30 chars)" rule if observed.
- Model output collapses to a fixed template. Mitigation: monitor response diversity; if collapses, raise the KL β below.

**Reward hacking handling is explicitly part of the project's story** — discovering and patching one such case will be documented in the final blog post.

### 6.1 Answer Extraction (used by reward function AND eval)

`extract_answer(response: str) -> dict | None` lives in `train/common.py` and is reused by:
- the GRPO reward function (§6),
- the DPO data labeler (§5.3),
- the eval harness (§5.5).

Using one parser everywhere is critical: if reward and eval use different parsers, the model can be "correct" by one and "wrong" by the other, silently breaking the training signal.

**Strict pattern (preferred match):**
```
<answer>\s*([A-Z]\s*:\s*(knight|knave)(\s*,\s*[A-Z]\s*:\s*(knight|knave))*)\s*</answer>
```
Matches `<answer>A: knight, B: knave, C: knave</answer>` and trivial whitespace variants.

**Fallback rules** (applied in order, only if strict fails):
1. Case-insensitive on `knight`/`knave` and on person labels.
2. Tolerate `=`, `is a`, or `→` as separators between label and identity (`A is a knight`, `A=knight`, `A→knight`).
3. Tolerate `;` or newline between entries instead of `,`.
4. If the `<answer>` tags are missing but a recognizable identity list is found in the last 200 chars of the response, parse from there.

**Return:**
- `dict` like `{"A": "knight", "B": "knave"}` if parsing succeeds AND all `n` people are accounted for (no duplicates, no missing).
- `None` otherwise.

**Why not laxer:** with very loose parsing, the model can wander into outputs that *look* answer-shaped to a regex but encode no real prediction — reward hacking. The fallback rules above cover real format drift but require all `n` identities to be claimed.

**Test fixture:** `train/test_extract.py` covers ~30 hand-crafted strings (correct, partial, hack attempts). Both reward function and eval harness import the same `extract_answer` from `common.py` — never re-implement.

### 6.2 Format Compliance Check (also shared)

`check_format(response: str) -> bool` also lives in `common.py`. It is the canonical "did the model produce well-formed output" predicate, used by:
- the §6 reward function (the `has_think and has_answer` block),
- the SFT eval that reports the "Format compliance ≥ 95%" secondary metric in §2,
- the DPO labeler when filtering out malformed SFT samples.

Definition:
```python
def check_format(response: str) -> bool:
    has_think = "<think>" in response and "</think>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    return has_think and has_answer
```

It is intentionally trivial. The point of defining it as a shared function is **identical wording across reward, SFT eval, and DPO** — so the 95% format-compliance number reported for SFT is the exact same predicate that gates the reward's format bonus. If we ever tighten the format check (e.g., to require non-empty `<think>`), it changes in one place.

### 6.3 GRPO Configuration Details

**Reference policy** (the model GRPO's KL term pulls toward):
- **Main pipeline (SFT → DPO → GRPO):** frozen DPO checkpoint serves as reference. Rationale: DPO already shifted the policy in the preferred direction; we want GRPO's RL to refine that, not undo it.
- **Ablation (SFT → GRPO):** frozen SFT checkpoint serves as reference.
- Reference model is loaded as a frozen LoRA adapter on top of the same base model — does not double GPU memory.

**KL coefficient β:**
- Start at **β = 0.04** (trl default for GRPOTrainer as of 0.13.0).
- Too low → policy drifts far from reference, mode-collapses to reward-hacking outputs.
- Too high → policy can't move; reward curve flat.
- Decision rule: if reward improves but eval accuracy *drops* (overfitting to reward hacks), raise β to 0.08. If reward curve is flat after 200 steps, drop to 0.02.

**Other GRPO knobs:**
- `num_generations = 4` (group size for advantage estimation; raise to 8 if A100 acquired)
- Sampling temperature for generation: 1.0 (encourages exploration of diverse reasoning chains)
- `max_completion_length = 512` (covers all observed n=2-6 solutions in SFT data; raise only if truncation observed)
- Old-policy update: synchronous (refresh reference adapter only between epochs, not every step) — saves I/O

---

## 7. Engineering: Colab Pro Constraints

| Concern | Mitigation |
|---|---|
| Session disconnect (12–24h limit) | Checkpoint to Google Drive every 100 steps; resumable training |
| GPU lottery (A100 vs L4 vs V100) | Code defaults to L4 settings (batch=2, grad_accum=8); A100 path uses batch=4, grad_accum=4 |
| Disk space (~100GB on Colab) | Models + adapters + data ≈ 20GB; well within limit |
| Reproducibility | All random seeds fixed; `requirements.txt` pins versions |
| Monitoring | wandb online; can survive Colab disconnects |

**Per-stage runtime estimates** (on A100 40GB):
| Stage | Steps | Wall-clock |
|---|---|---|
| SFT | ~750 (2000 ex × 3 epochs / batch 8 effective) | ~2 hours |
| DPO | ~250 (1000 pairs × 2 epochs / batch 8) | ~1.5 hours |
| GRPO (main) | ~750 (2000 prompts × 3 epochs / batch 8) with K=4 gen each | ~6 hours |
| GRPO-no-DPO (ablation) | same config as main GRPO | ~6 hours |
| Eval per system (greedy: 1800 puzzles) + (3-seed sampled) | — | ~1.5 hour per model |

Training fits within Colab sessions if split per stage. **Total training compute ≈ 16 hours** (SFT + DPO + 2× GRPO). Eval across 7 systems (Base, SFT, DPO, GRPO, GRPO-no-DPO, prompted-Qwen, DeepSeek-V3) ≈ 10 hours; split across 2-3 sessions. Worst case (L4 only): training ~2x slower, all stages still individually fit in one session.

### Total Budget

| Cost item | Estimate |
|---|---|
| Colab Pro subscription | $10 / month × ~2 months = **$20** |
| DeepSeek API — SFT data generation | $3-5 (worst case $8) |
| DeepSeek API — baseline eval (1800 puzzles × greedy + 3 sampled = 7200 calls × ~$0.001) | $5-8 |
| HuggingFace Hub storage (LoRA adapters, <1GB total) | $0 |
| wandb logging | $0 (free tier) |
| **Total** | **~$30-40** |

The dominant cost is Colab Pro itself. API costs are a rounding error relative to GPU time. If anything blows the budget it will be reward-hacking debugging in Week 4 extending into a 3rd month of Colab Pro ($10 more).

---

## 8. Repository Structure

```
logic-zero/
├── README.md                    # Demo, accuracy table, training curves
├── requirements.txt
├── data/
│   ├── gen_puzzles.py            # Knights & Knaves puzzle generator (templates + RNG)
│   ├── gen_sft_data.py            # Calls DeepSeek API (imports verifier from train/common.py)
│   ├── gen_dpo_data.py            # Uses SFT model to sample (imports verifier from train/common.py)
│   ├── gen_grpo_data.py           # Just puzzles + ground truth
│   ├── gen_eval_data.py           # Generates BOTH eval (1800) and dev (200) sets in one run
│   ├── sft_data.jsonl              # Generated
│   ├── dpo_data.jsonl
│   ├── grpo_data.jsonl
│   ├── dev_data.jsonl              # 200 puzzles for checkpoint selection
│   ├── eval_data.jsonl             # 1800 puzzles, held-out
│   └── eval_hashes.json            # Frozen hash set; training data excludes these
├── train/
│   ├── sft.py                    # SFTTrainer
│   ├── dpo.py                    # DPOTrainer
│   ├── grpo.py                   # GRPOTrainer (handles both main and ablation runs via config flag)
│   ├── reward.py                  # GRPO reward function (imports extract_answer + check_format from common)
│   ├── common.py                  # Shared: model loading, LoRA config, extract_answer(), check_format(), SAT verifier wrapper
│   └── test_extract.py            # Unit tests for extract_answer (~30 fixtures: correct, partial, hack attempts)
├── eval/
│   ├── run_eval.py                # Run a model: greedy (T=0) + 3-seed sampled (T=0.7), all on 1800 eval puzzles
│   ├── baselines.py               # Prompted-Qwen + DeepSeek-V3 zero-shot
│   ├── compare.py                  # Generate comparison table + charts across all 7 systems
│   └── analyze_responses.py        # Qualitative spot-checks
├── notebooks/
│   ├── 01_sft.ipynb                # Colab notebook for SFT stage
│   ├── 02_dpo.ipynb
│   ├── 03_grpo.ipynb
│   └── 04_eval.ipynb
├── results/
│   ├── accuracy_table.md            # 7 systems × n=2,3,4,5,6,7; greedy primary + 3-seed mean ± std
│   ├── training_curves.png
│   ├── reward_distribution.png
│   └── example_responses.md         # Cherry-picked qualitative comparisons
└── docs/
    └── specs/
        └── 2026-05-18-logic-zero-design.md   # This file
```

---

## 9. Timeline (5 weeks)

| Week | Deliverable |
|---|---|
| 1 | Puzzle generator + SAT verifier (with timeout) + **eval set + dev set generated first, hashes frozen** + DeepSeek SFT data gen (excludes eval/dev hashes) + SFT training (dev-set checkpoint selection) + **SFT accuracy on dev set** (quick sanity number; full eval harness not ready yet) |
| 2 | Eval harness (greedy + 3-seed sampled) + first full eval on Base + SFT + external baselines (prompted-Qwen, DeepSeek-V3) measured + DPO data construction (filter to correct-vs-incorrect pairs) + **trl GRPO LoRA-as-reference smoke test** (§11 Q#5) |
| 3 | DPO training + DPO eval + GRPO data gen + initial main GRPO run |
| 4 | GRPO reward debugging (expected reward-hacking iteration) + final main GRPO eval + **ablation GRPO-no-DPO run** + ablation eval |
| 5 | Full 7-system comparison table + README + blog post (incl. honest DPO-necessity analysis) + push checkpoints to HuggingFace |

**Why 5 weeks not 4:** the multi-seed eval + 2 external baselines + ablation branch + larger eval set (1800 vs 500) adds ~5 days. Better to budget realistically than miss the deadline. If 5 weeks turns out to be tight, the ablation branch (Week 4) is the only optional piece — main pipeline still ships at end of Week 4.

**Time risk:** Week 4's reward-hacking debugging is the most uncertain. Budget 2 extra days; if it slips, simplify reward function (drop length term) rather than extending timeline.

---

## 10. Final Deliverables

1. **GitHub repo** with reproducible code, requirements pinned, README with demo + numbers
2. **5 checkpoints on HuggingFace Hub**: `logic-zero-sft`, `logic-zero-dpo`, `logic-zero-grpo`, `logic-zero-grpo-no-dpo` (ablation), plus the base model reference (LoRA adapters only; base model is upstream Qwen2.5-1.5B-Instruct)
3. **Results page**: accuracy table (7 systems × 6 difficulty buckets; greedy primary + 3-seed mean ± std), training curves screenshots, response comparison examples, ablation finding ("DPO contributed +X pp" or "DPO redundant given GRPO")
4. **Blog post** (~1500 words): journey + reward hacking war story + lessons + honest comparison against external baselines + DPO-necessity ablation result
5. **Resume line**: e.g., *"Built end-to-end SFT→DPO→GRPO post-training pipeline for Qwen2.5-1.5B on Knights & Knaves logic puzzles; trained model reaches X% accuracy (greedy) vs Y% for prompted base and Z% for DeepSeek-V3; ablation shows DPO contributes +N pp over SFT+GRPO."* (fill X/Y/Z/N after eval)

---

## 11. Open Questions (to resolve during implementation)

1. **Tokenizer choice for `<think>`/`<answer>` tags**: add as special tokens or treat as plain text? Default: plain text (simpler; Qwen tokenizer splits them into reasonable subwords already). Switch only if SFT loss curve shows pathology around tag positions.
2. **DPO reference model storage**: keep SFT LoRA loaded as reference (saves ~3GB GPU memory by sharing base weights), or load a second full SFT-merged model (cleaner but heavier)? Default: shared base + dual adapters.
3. **GRPO group size escalation**: start `num_generations=4`; if A100 acquired and reward variance is high, try 8. Decision deferred to mid-training observation.
4. **SAT verifier timeout in reward function**: resolved in §5.1 — reward path uses 0.5s timeout with reward=0 on timeout, and counter tracked via wandb. Open sub-question: if the timeout counter spikes mid-training, do we lower timeout further (0.2s, risking false negatives on legitimate hard puzzles) or whitelist the response shape that's triggering them? Decision deferred until observed.

5. **trl 0.13.0 + LoRA-as-reference for GRPO — verify before Week 3.** §6.3 assumes we can load a frozen LoRA adapter as the GRPO reference policy on top of the same base model, keeping memory at ~1× model weights. trl's GRPOTrainer historically expected a separate `ref_model` argument; PEFT-shared-base support varies by trl version. **Action item:** in Week 2, write a 20-line smoke test that instantiates GRPOTrainer with `model = base+LoRA-A`, `ref_model = None` (or PEFT-disable trick), and confirms (a) it runs without OOM on L4, (b) the KL term is non-zero and reasonable. If unsupported, fallback options in order of preference: (i) merge the reference adapter into a temporary full-weight model held in CPU RAM and stream layers (slow but works); (ii) reduce `num_generations` to 2 and host a second full model on GPU (uses ~3 extra GB); (iii) drop the ablation if memory genuinely won't fit.

These do not block the spec; they will be decided during implementation based on observed behavior. Item 5 in particular is an implementation risk worth de-risking early — a 1-hour smoke test in Week 2 prevents a Week 3 surprise.
