#!/usr/bin/env python
"""
experiments/tune.py — generic per-scorer grid search with honest selection.

Sweeps any registered scorer over its declared grid, selects by argmax of the
neighbourhood-smoothed surface, and reports the honest cross-validated value
alongside the in-sample one so the optimism is always visible.

Each scorer is tuned on its own before any fusion is attempted. Fusing a tuned
scorer with an untuned one measures the untuned one dragging, not whether the
combination helps.

Usage:
    python experiments/tune.py --scorer lmd
    python experiments/tune.py --scorer bm25plus
    python experiments/tune.py --scorer all
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.evaluate import (
    DEFAULT_DATA,
    cv_folds,
    get_index,
    load_topics,
    paired_bootstrap,
    smooth_surface,
    sweep,
)
from experiments.sweep_bm25 import log_rows

# Grids are declared here rather than passed on the command line so that every
# run of this script searches the identical space -- a result is only comparable
# to another result from the same grid.
GRIDS: Dict[str, Dict[str, np.ndarray]] = {
    "bm25": {
        "k1": np.round(np.arange(0.3, 12.0 + 1e-9, 0.3), 3),
        "b": np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 3),
    },
    "bm25plus": {
        "k1": np.round(np.arange(0.6, 12.0 + 1e-9, 0.6), 3),
        "b": np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 3),
        "delta": np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0]),
    },
    # mu is a pseudo-count of collection-model tokens, so it is scanned
    # geometrically rather than linearly: the difference between 100 and 200 is
    # far larger than between 4000 and 4100.
    "lmd": {
        "mu": np.round(np.geomspace(50, 20000, 40), 1),
    },
}


def build_configs(grid: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    names = list(grid)
    configs = [{}]
    for name in names:
        configs = [{**c, name: float(v)} for c in configs for v in grid[name]]
    return configs


def tune(index, queries, qrels, scorer_name: str, folds: int = 5) -> Dict:
    grid = GRIDS[scorer_name]
    axes = tuple(grid)
    configs = build_configs(grid)
    print(f"\n{'='*70}\n{scorer_name}: {len(configs)} configs over {axes}\n{'='*70}")

    rows = sweep(index, queries, qrels, scorer_name, configs)
    log_rows(rows, f"{scorer_name}-tune")

    keyed = {tuple(round(r["params"][a], 6) for a in axes): r for r in rows}
    in_sample = keyed[max(keyed, key=lambda k: keyed[k]["ndcg@10"])]
    smoothed_full = smooth_surface(rows, axes)
    chosen = keyed[max(smoothed_full, key=lambda k: smoothed_full[k])]

    # Honest estimate: apply the same selection rule using training folds only.
    qids = sorted(rows[0]["per_query"])
    held_out: Dict[str, float] = {}
    for test in cv_folds(qids, folds):
        train = [q for q in qids if q not in set(test)]
        train_scores = {k: float(np.mean([r["per_query"][q] for q in train]))
                        for k, r in keyed.items()}
        pick = keyed[max(smooth_surface(rows, axes, train_scores),
                         key=lambda k: smooth_surface(rows, axes, train_scores)[k])]
        for q in test:
            held_out[q] = pick["per_query"][q]

    honest = float(np.mean(list(held_out.values())))
    print(f"  in-sample best : {in_sample['ndcg@10']:.4f}  {in_sample['params']}")
    print(f"  smoothed pick  : {chosen['ndcg@10']:.4f}  {chosen['params']}")
    print(f"  honest CV      : {honest:.4f}   (optimism {chosen['ndcg@10']-honest:+.4f})")
    return {"scorer": scorer_name, "params": chosen["params"],
            "in_sample": chosen["ndcg@10"], "honest": honest,
            "per_query": chosen["per_query"], "held_out": held_out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scorer", default="all")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    args = ap.parse_args()

    index = get_index(os.path.join(args.data_dir, "corpus.jsonl"))
    queries, qrels = load_topics(args.data_dir)

    names = list(GRIDS) if args.scorer == "all" else [args.scorer]
    results = [tune(index, queries, qrels, n) for n in names]

    print(f"\n{'='*70}\nSummary\n{'='*70}")
    print(f"{'scorer':<10} {'in-sample':>10} {'honest CV':>10}   params")
    for r in results:
        print(f"{r['scorer']:<10} {r['in_sample']:>10.4f} {r['honest']:>10.4f}   {r['params']}")

    if len(results) > 1:
        print(f"\n{'='*70}\nPaired comparisons on honest held-out scores\n{'='*70}")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                a, b = results[i], results[j]
                st = paired_bootstrap(a["held_out"], b["held_out"])
                verdict = "significant" if st["p_value"] < 0.05 else "not significant"
                print(f"  {a['scorer']:>9} vs {b['scorer']:<9}: delta={st['delta']:+.4f}  "
                      f"p={st['p_value']:.4f}  W/L/T={st['wins']}/{st['losses']}/{st['ties']} "
                      f"[{verdict}]")

    out = os.path.join(os.path.dirname(__file__), "tuned_params.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({r["scorer"]: {"params": r["params"], "in_sample": r["in_sample"],
                                 "honest": r["honest"]} for r in results}, f, indent=2)
    print(f"\nwrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
