#!/usr/bin/env python
"""
experiments/cv_select.py — does plateau selection actually generalise better
than argmax?

The plan asserts it does. This tests it, because an untested methodological
claim is worth nothing in the report and less in the oral defense.

Why this experiment is necessary
--------------------------------
Comparing two configurations on the same topics used to *choose* one of them is
circular: the argmax is by construction the best point on that sample, so it
wins any such comparison whether or not it is genuinely better. The measured
+0.0154 "advantage" of argmax over a plateau pick in sweep_bm25.py is exactly
this artefact and is not evidence of anything.

The valid design is nested cross-validation. For each fold: run the selection
rule using only the training topics, then score the selected configuration on
the held-out topics it never saw. Averaged over folds, that is an unbiased
estimate of how each *rule* performs -- which is the actual question, since the
rule is what will be applied before the private topic set is scored.

Three rules are compared:
    default    textbook k1=1.2, b=0.75            (control: no tuning at all)
    argmax     best mean nDCG@10 on training folds
    smoothed   argmax of the neighbourhood-averaged surface (plateau centre)

Reads experiments/results.jsonl, so it re-runs no retrieval.

Usage:
    python experiments/cv_select.py
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.evaluate import cv_folds, paired_bootstrap, smooth_surface

RESULTS = os.path.join(os.path.dirname(__file__), "results.jsonl")
AXES = ("k1", "b")


def load_rows(tag_prefix: str, path: str = RESULTS) -> List[Dict]:
    rows, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("tag", "").startswith(tag_prefix):
                continue
            key = tuple(round(row["params"][a], 6) for a in AXES)
            if key in seen:          # keep the first occurrence of each config
                continue
            seen.add(key)
            rows.append(row)
    return rows


def mean_over(row: Dict, qids) -> float:
    vals = [row["per_query"][q] for q in qids if q in row["per_query"]]
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag-prefix", default="bm25-k1b")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--radius", type=int, default=1)
    args = ap.parse_args()

    rows = load_rows(args.tag_prefix)
    if not rows:
        raise SystemExit(f"no rows tagged {args.tag_prefix!r} in {RESULTS}")
    qids = sorted(rows[0]["per_query"])
    folds = cv_folds(qids, args.folds)
    print(f"{len(rows)} configurations, {len(qids)} topics, {args.folds} folds "
          f"(sizes {[len(f) for f in folds]})")

    default = next(r for r in rows
                   if abs(r["params"]["k1"] - 1.2) < 1e-6 and abs(r["params"]["b"] - 0.75) < 1e-6)

    held_out: Dict[str, Dict[str, float]] = {"default": {}, "argmax": {}, "smoothed": {}}
    picks: Dict[str, List] = {"argmax": [], "smoothed": []}

    for test in folds:
        train = [q for q in qids if q not in set(test)]
        train_scores = {tuple(round(r["params"][a], 6) for a in AXES): mean_over(r, train)
                        for r in rows}
        keyed = {tuple(round(r["params"][a], 6) for a in AXES): r for r in rows}

        pick_argmax = keyed[max(train_scores, key=lambda k: train_scores[k])]
        smoothed = smooth_surface(rows, AXES, train_scores, radius=args.radius)
        pick_smooth = keyed[max(smoothed, key=lambda k: smoothed[k])]

        picks["argmax"].append((pick_argmax["params"]["k1"], pick_argmax["params"]["b"]))
        picks["smoothed"].append((pick_smooth["params"]["k1"], pick_smooth["params"]["b"]))

        # Score each selected configuration on the topics it never saw.
        for q in test:
            held_out["default"][q] = default["per_query"][q]
            held_out["argmax"][q] = pick_argmax["per_query"][q]
            held_out["smoothed"][q] = pick_smooth["per_query"][q]

    print(f"\n{'='*70}\nPer-fold selections\n{'='*70}")
    print(f"{'fold':>5}  {'argmax (k1,b)':>18}  {'smoothed (k1,b)':>18}")
    for i in range(len(folds)):
        a, s = picks["argmax"][i], picks["smoothed"][i]
        print(f"{i:>5}  {f'({a[0]:.2f}, {a[1]:.2f})':>18}  {f'({s[0]:.2f}, {s[1]:.2f})':>18}")

    spread = lambda ps: (np.std([p[0] for p in ps]), np.std([p[1] for p in ps]))
    sa, ss = spread(picks["argmax"]), spread(picks["smoothed"])
    print(f"\nselection stability (std across folds):")
    print(f"  argmax   k1 {sa[0]:.3f}  b {sa[1]:.3f}")
    print(f"  smoothed k1 {ss[0]:.3f}  b {ss[1]:.3f}")

    print(f"\n{'='*70}\nHONEST held-out nDCG@10 (each topic scored by a rule that never saw it)"
          f"\n{'='*70}")
    for name in ("default", "argmax", "smoothed"):
        vals = np.array([held_out[name][q] for q in qids])
        print(f"  {name:<9} {vals.mean():.4f}   (SE {vals.std(ddof=1)/np.sqrt(vals.size):.4f})")

    print(f"\n{'='*70}\nPaired bootstrap on held-out scores\n{'='*70}")
    for label, a, b in (
        ("argmax   vs default ", "argmax", "default"),
        ("smoothed vs default ", "smoothed", "default"),
        ("smoothed vs argmax  ", "smoothed", "argmax"),
    ):
        st = paired_bootstrap(held_out[a], held_out[b])
        verdict = "significant" if st["p_value"] < 0.05 else "not significant"
        print(f"  {label}: delta={st['delta']:+.4f}  p={st['p_value']:.4f}  "
              f"paired SE={st['paired_se']:.4f}  W/L/T={st['wins']}/{st['losses']}/{st['ties']} "
              f"[{verdict}]")

    # The bias the whole exercise is about: how much the in-sample argmax score
    # overstates what that same rule actually delivers on unseen topics.
    full_argmax = max(rows, key=lambda r: r["ndcg@10"])
    honest = np.mean([held_out["argmax"][q] for q in qids])
    print(f"\n{'='*70}\nOptimism of the in-sample argmax\n{'='*70}")
    print(f"  reported on the full dev set : {full_argmax['ndcg@10']:.4f} "
          f"(k1={full_argmax['params']['k1']:.2f}, b={full_argmax['params']['b']:.2f})")
    print(f"  honest cross-validated       : {honest:.4f}")
    print(f"  optimism (overstatement)     : {full_argmax['ndcg@10'] - honest:+.4f}")


if __name__ == "__main__":
    main()
