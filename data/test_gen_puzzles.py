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
