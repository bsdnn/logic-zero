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
