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
