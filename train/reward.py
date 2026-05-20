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
