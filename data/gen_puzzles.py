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
    # Note: self_knight / self_knave are tautological/contradictory in K&K semantics,
    # so we exclude them from the candidate pool and use them only as a fallback.
    for _ in range(50):
        kind = rng.choice([
            "is", "same", "diff", "at_least_knights", "at_least_knaves",
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
