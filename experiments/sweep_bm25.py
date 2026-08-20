#!/usr/bin/env python
"""
experiments/sweep_bm25.py — the k1/b parameter search (assignment Section 6,
"your parameter search procedure for k1 and b").

Sweeps the BM25 grid, logs every configuration to results.jsonl with its
per-query scores, and then does the part that actually matters: picks the
centre of the best *plateau* rather than the argmax, and reports a paired
bootstrap between the two so the choice is defended with a number instead of a
preference.

Why not argmax: with 50 topics the measured standard error of the mean is
~0.039, so dozens of grid points are statistically indistinguishable from the
peak. Taking the peak fits noise, and `max` over noisy estimates is biased
upward -- the reported dev score would overstate held-out performance. The
midpoint of the widest near-optimal region has good neighbours, which is the
property that survives a change of topic set.

Usage:
    python experiments/sweep_bm25.py                    # full grid
    python experiments/sweep_bm25.py --coarse           # fast pass
    python experiments/sweep_bm25.py --scorer bm25plus
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.evaluate import (
    DEFAULT_DATA,
    get_index,
    load_topics,
    paired_bootstrap,
    select_plateau,
    sweep,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results.jsonl")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.join(os.path.dirname(__file__), ".."), text=True,
        ).strip()
    except Exception:
        return "unknown"


def log_rows(rows, tag: str, path: str = RESULTS) -> None:
    """Append every evaluated configuration, with the git SHA that produced it.

    Non-negotiable per the plan: the report's ablation table and the record of
    what was tried are both reconstructed from this file, and a result that
    cannot be traced to a commit cannot be defended.
    """
    sha, stamp = git_sha(), time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({**row, "tag": tag, "git_sha": sha, "timestamp": stamp}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scorer", default="bm25")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--coarse", action="store_true", help="wider grid steps for a fast pass")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--k1-min", type=float, default=0.3)
    ap.add_argument("--k1-max", type=float, default=3.0)
    ap.add_argument("--k1-step", type=float, default=None)
    ap.add_argument("--b-step", type=float, default=None)
    args = ap.parse_args()

    k1_step = args.k1_step or (0.2 if args.coarse else 0.1)
    b_step = args.b_step or (0.1 if args.coarse else 0.05)
    k1_values = np.round(np.arange(args.k1_min, args.k1_max + 1e-9, k1_step), 3)
    b_values = np.round(np.arange(0.0, 1.0 + 1e-9, b_step), 3)

    index = get_index(os.path.join(args.data_dir, "corpus.jsonl"))
    queries, qrels = load_topics(args.data_dir)
    print(f"index N={index.N:,}  topics={len(queries)}  "
          f"grid={len(k1_values)}x{len(b_values)}={len(k1_values)*len(b_values)} configs")

    configs = [{"k1": float(k1), "b": float(b)} for k1 in k1_values for b in b_values]
    # Always evaluate the textbook default, even when the grid step steps over
    # it -- it is the reference every result is reported against.
    if not any(abs(c["k1"] - 1.2) < 1e-6 and abs(c["b"] - 0.75) < 1e-6 for c in configs):
        configs.append({"k1": 1.2, "b": 0.75})
    t0 = time.time()
    rows = sweep(index, queries, qrels, args.scorer, configs)
    print(f"swept in {time.time()-t0:.1f}s")

    tag = args.tag or f"{args.scorer}-k1b-{'coarse' if args.coarse else 'fine'}"
    log_rows(rows, tag)

    by_score = sorted(rows, key=lambda r: -r["ndcg@10"])
    best = by_score[0]
    baseline = next(r for r in rows
                    if abs(r["params"]["k1"] - 1.2) < 1e-6 and abs(r["params"]["b"] - 0.75) < 1e-6)

    print(f"\n{'='*66}\nTop 10 configurations\n{'='*66}")
    print(f"{'k1':>6} {'b':>6} {'nDCG@10':>9} {'SE':>7}")
    for r in by_score[:10]:
        print(f"{r['params']['k1']:>6.2f} {r['params']['b']:>6.2f} "
              f"{r['ndcg@10']:>9.4f} {r['se']:>7.4f}")

    print(f"\nTextbook default (k1=1.2, b=0.75): {baseline['ndcg@10']:.4f}")

    # Plateau selection along each axis, holding the other at the argmax value.
    print(f"\n{'='*66}\nPlateau vs argmax\n{'='*66}")
    chosen = {}
    for param, other in (("k1", "b"), ("b", "k1")):
        slice_rows = [r for r in rows
                      if abs(r["params"][other] - best["params"][other]) < 1e-6]
        sel = select_plateau(slice_rows, param)
        chosen[param] = sel["chosen"]["params"][param]
        print(f"{param}: argmax={sel['argmax']['params'][param]:.2f} "
              f"(nDCG {sel['argmax']['ndcg@10']:.4f})  ->  "
              f"plateau centre={sel['chosen']['params'][param]:.2f} "
              f"(nDCG {sel['chosen']['ndcg@10']:.4f}), "
              f"plateau spans {sel['plateau_range'][0]:.2f}-{sel['plateau_range'][1]:.2f} "
              f"({sel['plateau_size']} points)")

    plateau_row = next(
        (r for r in rows
         if abs(r["params"]["k1"] - chosen["k1"]) < 1e-6
         and abs(r["params"]["b"] - chosen["b"]) < 1e-6),
        best,
    )

    print(f"\n{'='*66}\nPaired bootstrap (10k resamples)\n{'='*66}")
    for label, a, b_ in (
        ("plateau choice vs textbook default", plateau_row, baseline),
        ("argmax          vs textbook default", best, baseline),
        ("argmax          vs plateau choice  ", best, plateau_row),
    ):
        st = paired_bootstrap(a["per_query"], b_["per_query"])
        verdict = "significant" if st["p_value"] < 0.05 else "NOT significant"
        print(f"{label}: delta={st['delta']:+.4f}  p={st['p_value']:.4f}  "
              f"paired SE={st['paired_se']:.4f}  "
              f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}  [{verdict}]")

    unpaired_se = float(np.median([r["se"] for r in rows]))
    print(f"\nMedian unpaired SE across grid: {unpaired_se:.4f}")
    print(f"Configurations within 1 SE of the best: "
          f"{sum(1 for r in rows if r['ndcg@10'] >= best['ndcg@10'] - unpaired_se)} / {len(rows)}")

    print(f"\nRECOMMENDED: k1={chosen['k1']:.2f}, b={chosen['b']:.2f} "
          f"-> nDCG@10 {plateau_row['ndcg@10']:.4f}")
    print(f"logged {len(rows)} rows to {os.path.relpath(RESULTS)}")


if __name__ == "__main__":
    main()
