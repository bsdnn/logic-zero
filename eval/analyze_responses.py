"""Pull 10 example puzzles where systems disagree most; emit a markdown side-by-side."""
from __future__ import annotations
import json
from pathlib import Path
from peft import PeftModel
import torch

from train.common import load_base_model, extract_answer, to_chat

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
