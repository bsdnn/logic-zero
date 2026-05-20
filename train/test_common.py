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
