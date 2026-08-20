#!/usr/bin/env python
"""
experiments/fuse.py — RRF fusion skeleton and its diagnostics.

Fusion is not a free lunch. It pays only when the runs being combined disagree:
combining two rankers that return nearly the same documents in nearly the same
order just reproduces one of them. So this script measures agreement *first*
(rank overlap and Kendall tau between runs) and only then asks whether fusing
actually helps, on held-out topics.

Reciprocal Rank Fusion (Cormack et al. 2009):

    score(d) = sum_r  w_r / (rrf_k + rank_r(d))

Rank-based, so it is immune to the score-scale mismatch between BM25 (unbounded
sums of IDF-weighted terms) and a query-likelihood language model (sums of log
probabilities, all negative). That immunity is the whole reason to prefer it
over weighted score summation, which would need per-query normalisation and a
scale hyperparameter per run.

Every comparison here is cross-validated. A fusion weight chosen by maximising
on the same topics used to report the result would be the circular comparison
that finding F10 already caught once.

Usage:
    python experiments/fuse.py
    python experiments/fuse.py --depth 100 --rrf-k 60
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import (
    DEFAULT_DATA,
    QueryPostings,
    cv_folds,
    get_index,
    load_topics,
    paired_bootstrap,
)
from submission import _scorers
from submission._scorers import CollectionStats
from submission._traverse import rrf_fuse

TUNED = os.path.join(os.path.dirname(__file__), "tuned_params.json")


def load_tuned() -> Dict[str, Dict[str, float]]:
    with open(TUNED, encoding="utf-8") as f:
        return {k: v["params"] for k, v in json.load(f).items()}


def per_scorer_runs(index, queries, specs: Sequence[Tuple[str, Dict]],
                    depth: int) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """Ranked lists per (alias, qid), decoding each query's postings once."""
    stats = CollectionStats(index.N, index.avg_doc_len, index.total_tokens)
    runs = {alias: {} for alias, _, _ in specs}
    for qid, text in queries:
        postings = QueryPostings(index, text)
        for alias, name, params in specs:
            scorer = _scorers.get(name)
            resolved = _scorers.resolve_params(name, params)
            runs[alias][qid] = postings.rank(scorer, resolved, stats, k=depth)
    return runs


def agreement(run_a: Dict, run_b: Dict, depth: int = 10) -> Dict[str, float]:
    """How much do two runs actually differ?

    `overlap` is the mean fraction of shared documents in the top `depth`.
    `kendall_tau` is computed over the documents both runs rank, measuring
    whether they order the *shared* documents the same way.

    High overlap and high tau means fusion has nothing to work with.
    """
    overlaps, taus = [], []
    for qid in run_a:
        a = [d for d, _ in run_a[qid][:depth]]
        b = [d for d, _ in run_b[qid][:depth]]
        if not a or not b:
            continue
        shared = set(a) & set(b)
        overlaps.append(len(shared) / max(len(a), len(b)))
        if len(shared) > 1:
            rank_a = {d: i for i, d in enumerate(a)}
            rank_b = {d: i for i, d in enumerate(b)}
            common = sorted(shared)
            concordant = discordant = 0
            for i in range(len(common)):
                for j in range(i + 1, len(common)):
                    x, y = common[i], common[j]
                    s = (rank_a[x] - rank_a[y]) * (rank_b[x] - rank_b[y])
                    if s > 0:
                        concordant += 1
                    elif s < 0:
                        discordant += 1
            total = concordant + discordant
            if total:
                taus.append((concordant - discordant) / total)
    return {
        "overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "kendall_tau": float(np.mean(taus)) if taus else 0.0,
    }


def fused_scores(runs: Dict, aliases: Sequence[str], qrels: Dict,
                 weights: Sequence[float], rrf_k: float, k: int = 10) -> Dict[str, float]:
    """Per-query nDCG@10 of the RRF-fused ranking."""
    out = {}
    for qid in qrels:
        if qid not in runs[aliases[0]]:
            continue
        fused = rrf_fuse([runs[a][qid] for a in aliases], k=k, rrf_k=rrf_k, weights=weights)
        out[qid] = ndcg_at_k([d for d, _ in fused], qrels[qid], k=10)
    return out


def single_scores(runs: Dict, alias: str, qrels: Dict) -> Dict[str, float]:
    return {qid: ndcg_at_k([d for d, _ in runs[alias][qid][:10]], qrels[qid], k=10)
            for qid in qrels if qid in runs[alias]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--depth", type=int, default=1000,
                    help="ranked-list depth fed to fusion (deeper = more to fuse)")
    ap.add_argument("--rrf-k", type=float, default=60.0)
    args = ap.parse_args()

    index = get_index(os.path.join(args.data_dir, "corpus.jsonl"))
    queries, qrels = load_topics(args.data_dir)
    tuned = load_tuned()

    specs = [(name, name, tuned[name]) for name in ("bm25", "bm25plus", "lmd") if name in tuned]
    print(f"scorers: {[s[0] for s in specs]}   depth={args.depth}   rrf_k={args.rrf_k}")
    runs = per_scorer_runs(index, queries, specs, args.depth)

    print(f"\n{'='*70}\nDo these runs actually disagree?\n{'='*70}")
    print(f"{'pair':<22} {'overlap@10':>11} {'overlap@100':>12} {'tau@10':>8}")
    aliases = [s[0] for s in specs]
    for i in range(len(aliases)):
        for j in range(i + 1, len(aliases)):
            a, b = aliases[i], aliases[j]
            top10 = agreement(runs[a], runs[b], 10)
            top100 = agreement(runs[a], runs[b], 100)
            print(f"{a + ' vs ' + b:<22} {top10['overlap']:>11.3f} "
                  f"{top100['overlap']:>12.3f} {top10['kendall_tau']:>8.3f}")

    singles = {a: single_scores(runs, a, qrels) for a in aliases}
    print(f"\n{'='*70}\nSingle scorers (full dev)\n{'='*70}")
    for a in aliases:
        print(f"  {a:<10} {np.mean(list(singles[a].values())):.4f}")

    # Equal-weight fusion of every subset of size >= 2. Equal weights first:
    # any gain here is attributable to fusion itself rather than to a weight
    # fitted on the evaluation topics.
    print(f"\n{'='*70}\nEqual-weight RRF fusion (full dev)\n{'='*70}")
    from itertools import combinations
    fusions = {}
    for size in range(2, len(aliases) + 1):
        for combo in combinations(aliases, size):
            scores = fused_scores(runs, combo, qrels, [1.0] * len(combo), args.rrf_k)
            fusions[combo] = scores
            print(f"  {'+'.join(combo):<28} {np.mean(list(scores.values())):.4f}")

    best_single = max(aliases, key=lambda a: np.mean(list(singles[a].values())))
    print(f"\n{'='*70}\nPaired bootstrap vs best single scorer ({best_single})\n{'='*70}")
    for combo, scores in fusions.items():
        st = paired_bootstrap(scores, singles[best_single])
        verdict = "significant" if st["p_value"] < 0.05 else "not significant"
        print(f"  {'+'.join(combo):<28} delta={st['delta']:+.4f}  p={st['p_value']:.4f}  "
              f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}  [{verdict}]")

    # Cross-validated weight selection: choose the RRF weighting on training
    # folds, score it on held-out topics. Without this the reported fusion gain
    # would include whatever the weights absorbed from the evaluation topics.
    if len(aliases) >= 2:
        print(f"\n{'='*70}\nCross-validated weighted fusion (bm25 + lmd)\n{'='*70}")
        combo = ("bm25", "lmd") if "lmd" in aliases else tuple(aliases[:2])
        weight_grid = [(w, 1.0 - w) for w in np.round(np.arange(0.0, 1.01, 0.1), 2)]
        all_weighted = {w: fused_scores(runs, combo, qrels, list(w), args.rrf_k)
                        for w in weight_grid}
        qids = sorted(qrels)
        held_out = {}
        picks = []
        for test in cv_folds(qids, 5):
            train = [q for q in qids if q not in set(test)]
            best_w = max(weight_grid,
                         key=lambda w: np.mean([all_weighted[w][q] for q in train if q in all_weighted[w]]))
            picks.append(best_w)
            for q in test:
                if q in all_weighted[best_w]:
                    held_out[q] = all_weighted[best_w][q]
        print(f"  per-fold weight picks: {picks}")
        print(f"  honest CV fused      : {np.mean(list(held_out.values())):.4f}")
        print(f"  best single ({best_single:<8}): {np.mean(list(singles[best_single].values())):.4f}")
        st = paired_bootstrap(held_out, singles[best_single])
        verdict = "significant" if st["p_value"] < 0.05 else "not significant"
        print(f"  delta={st['delta']:+.4f}  p={st['p_value']:.4f}  "
              f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}  [{verdict}]")


if __name__ == "__main__":
    main()
