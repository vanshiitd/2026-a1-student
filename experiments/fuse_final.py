#!/usr/bin/env python
"""
experiments/fuse_final.py — the last fusion question: does score-based fusion
succeed where rank-based fusion failed?

Finding F13 diagnosed RRF's failure precisely: it is *rank-based*, therefore
scale-blind, so it cannot tell that one run is much better than another and
hands a weak ranker equal say. That diagnosis implies a specific remedy —
z-normalise each run's scores per query and combine them with weights, which
*can* express "trust this run less". This script tests that remedy across all
five tuned scorers, plus RRF as the control.

    z(d) = (score(d) - mean_q) / std_q          per query, per run
    fused(d) = sum_r  w_r * z_r(d)

Weights are selected by cross-validation, never on the topics used to report
the result. Documents a run did not retrieve get that run's minimum z-score
rather than zero, so "absent" is treated as "ranked last by this run" instead of
"average" -- with negative z-scores in play, zero-filling would silently reward
a document for being missed.
"""
import json
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import (
    QueryPostings, cv_folds, get_index, load_topics, paired_bootstrap)
from experiments.fuse import agreement
from submission import _scorers
from submission._scorers import CollectionStats
from submission._traverse import rrf_fuse

DEPTH = 1000


def znorm(run):
    """Per-query z-normalised scores, keyed by doc id."""
    if not run:
        return {}, 0.0
    vals = np.array([s for _d, s in run], dtype=np.float64)
    mu, sd = vals.mean(), vals.std()
    if sd <= 0:
        return {d: 0.0 for d, _s in run}, 0.0
    z = (vals - mu) / sd
    return {d: float(v) for (d, _s), v in zip(run, z)}, float(z.min())


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(os.path.dirname(__file__), "tuned_params.json")) as f:
        tuned = {k: v["params"] for k, v in json.load(f).items()}

    index = get_index(os.path.join(root, "data", "full", "corpus.jsonl"))
    queries, qrels = load_topics()
    stats = CollectionStats(index.N, index.avg_doc_len, index.total_tokens)
    names = [n for n in ("bm25", "bm25plus", "lmd", "pl2", "dph") if n in tuned]

    # One postings decode per query, reused by every scorer.
    runs = {n: {} for n in names}
    for qid, text in queries:
        p = QueryPostings(index, text)
        for n in names:
            runs[n][qid] = p.rank(_scorers.get(n), _scorers.resolve_params(n, tuned[n]),
                                  stats, k=DEPTH)

    singles = {n: {q: ndcg_at_k([d for d, _ in runs[n][q][:10]], qrels[q], k=10)
                   for q in qrels if q in runs[n]} for n in names}
    best_name = max(names, key=lambda n: np.mean(list(singles[n].values())))
    print(f"{'scorer':<10} {'nDCG@10':>9}   agreement with bm25 (overlap@10 / tau)")
    for n in names:
        ag = agreement(runs["bm25"], runs[n], 10) if n != "bm25" else {"overlap": 1.0, "kendall_tau": 1.0}
        print(f"{n:<10} {np.mean(list(singles[n].values())):>9.4f}   "
              f"{ag['overlap']:.3f} / {ag['kendall_tau']:.3f}")
    print(f"\nbest single: {best_name}\n")

    qids = sorted(qrels)
    zruns = {n: {q: znorm(runs[n][q]) for q in qids} for n in names}

    def fuse_z(combo, weights, q):
        pool, out = set(), {}
        for n in combo:
            pool |= set(zruns[n][q][0])
        for d in pool:
            out[d] = sum(w * zruns[n][q][0].get(d, zruns[n][q][1])
                         for n, w in zip(combo, weights))
        return sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    print(f"{'combination':<34} {'RRF':>8} {'z-CombSUM':>11} {'z-CV':>8} {'p vs best':>10}")
    weight_grid = [(w, round(1 - w, 2)) for w in np.round(np.arange(0.1, 1.0, 0.1), 2)]
    for size in (2, 3):
        for combo in combinations(names, size):
            if "bm25" not in combo:
                continue
            rrf = {q: ndcg_at_k([d for d, _ in rrf_fuse([runs[n][q] for n in combo], k=10)],
                                qrels[q], k=10) for q in qids}
            eq = [1.0] * len(combo)
            zeq = {q: ndcg_at_k([d for d, _ in fuse_z(combo, eq, q)], qrels[q], k=10)
                   for q in qids}
            # Cross-validated weights (2-run combos only; grid stays interpretable).
            if size == 2:
                allw = {w: {q: ndcg_at_k([d for d, _ in fuse_z(combo, w, q)], qrels[q], k=10)
                            for q in qids} for w in weight_grid}
                held = {}
                for test in cv_folds(qids, 5):
                    tr = [q for q in qids if q not in set(test)]
                    bw = max(weight_grid, key=lambda w: np.mean([allw[w][q] for q in tr]))
                    for q in test:
                        held[q] = allw[bw][q]
                zcv = float(np.mean(list(held.values())))
                st = paired_bootstrap(held, singles[best_name])
            else:
                zcv, st = float(np.mean(list(zeq.values()))), paired_bootstrap(zeq, singles[best_name])
            print(f"  {'+'.join(combo):<32} {np.mean(list(rrf.values())):>8.4f} "
                  f"{np.mean(list(zeq.values())):>11.4f} {zcv:>8.4f} {st['p_value']:>10.4f}")

    print(f"\n  best single ({best_name}) = {np.mean(list(singles[best_name].values())):.4f}")
    print("  z-CV column is cross-validated; RRF and z-CombSUM columns are equal-weight in-sample.")


if __name__ == "__main__":
    main()
