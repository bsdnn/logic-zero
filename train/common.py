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
        # Knight <-> statement is true.   Equivalent: vars[speaker] == truth
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
        return False  # treat timeout as incorrect (see spec 5.1)
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
