#!/usr/bin/env python
"""
experiments/robustness.py — does the tuned configuration transfer to topics
that do not look like the dev topics?

Motivation: BEIR's TREC-COVID ships exactly 50 topics and **all 50 were released
as our dev set**. The held-out topics used for the private leaderboard therefore
cannot be a held-back slice of the same 50 -- they are something else, and may
well differ in style or length from what `k1 = 4.5, b = 0.60` was tuned on
(mean 11.7 tokens, natural-language questions).

That is a generalisation risk no amount of cross-validation on the dev topics can
detect, because CV resamples the *same* distribution. This script probes it the
only way available: split the dev topics by a property the held-out set might
differ on, tune on one half, and evaluate on the other. If the optimum moves a
lot across such a split, the configuration is style-sensitive and a flatter,
more conservative setting is the safer bet against unseen topics.

Usage:
    python experiments/robustness.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.evaluate import get_index, load_topics, paired_bootstrap, sweep
from submission._analysis import analyze

REPO = os.path.join(os.path.dirname(__file__), "..")


def main():
    index = get_index(os.path.join(REPO, "data", "full", "corpus.jsonl"))
    queries, qrels = load_topics()
    judged = [(q, t) for q, t in queries if q in qrels]

    lengths = {q: len(analyze(t, index.config)) for q, t in judged}
    median = float(np.median(list(lengths.values())))
    short = sorted(q for q, n in lengths.items() if n <= median)
    long_ = sorted(q for q, n in lengths.items() if n > median)
    print(f"{len(judged)} topics, median length {median:.0f} tokens")
    print(f"  short half: {len(short)} topics (<= {median:.0f})")
    print(f"  long  half: {len(long_)} topics (>  {median:.0f})\n")

    k1s = np.round(np.arange(0.6, 12.0 + 1e-9, 0.3), 3)
    bs = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 3)
    configs = [{"k1": float(a), "b": float(b)} for a in k1s for b in bs]
    rows = sweep(index, judged, qrels, "bm25", configs)

    def mean_over(row, qs):
        return float(np.mean([row["per_query"][q] for q in qs if q in row["per_query"]]))

    def best_on(qs):
        return max(rows, key=lambda r: mean_over(r, qs))

    shipped = next(r for r in rows
                   if abs(r["params"]["k1"] - 4.5) < 1e-6 and abs(r["params"]["b"] - 0.6) < 1e-6)

    print(f"{'tuned on':<12} {'best (k1,b)':>14} {'own half':>10} {'other half':>11} "
          f"{'shipped on other':>17}")
    for name, train, test in (("short", short, long_), ("long", long_, short)):
        b = best_on(train)
        print(f"{name:<12} {f'({b['params']['k1']:g}, {b['params']['b']:g})':>14} "
              f"{mean_over(b, train):>10.4f} {mean_over(b, test):>11.4f} "
              f"{mean_over(shipped, test):>17.4f}")

    print(f"\nshipped config (4.5, 0.60): short {mean_over(shipped, short):.4f}  "
          f"long {mean_over(shipped, long_):.4f}  all {shipped['ndcg@10']:.4f}")

    # How much does the shipped setting give up against a config tuned
    # specifically for each half? That gap bounds the cost of style mismatch.
    for name, qs in (("short", short), ("long", long_)):
        b = best_on(qs)
        gap = mean_over(b, qs) - mean_over(shipped, qs)
        print(f"  cost of using the shipped config on the {name} half: {gap:+.4f}")

    # Sensitivity: how flat is the surface around the shipped point?
    near = [r for r in rows
            if abs(r["params"]["k1"] - 4.5) <= 1.5 and abs(r["params"]["b"] - 0.6) <= 0.15]
    vals = [r["ndcg@10"] for r in near]
    print(f"\nlocal flatness: {len(near)} configs within (k1 +/- 1.5, b +/- 0.15) span "
          f"{min(vals):.4f}-{max(vals):.4f} (range {max(vals)-min(vals):.4f})")

    out = os.path.join(os.path.dirname(__file__), "robustness.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "median_query_length": median,
            "shipped": {"params": shipped["params"], "all": shipped["ndcg@10"],
                        "short": mean_over(shipped, short), "long": mean_over(shipped, long_)},
            "best_short": {"params": best_on(short)["params"],
                           "on_long": mean_over(best_on(short), long_)},
            "best_long": {"params": best_on(long_)["params"],
                          "on_short": mean_over(best_on(long_), short)},
        }, f, indent=2)
    print(f"wrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
