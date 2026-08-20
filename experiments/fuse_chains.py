#!/usr/bin/env python
"""
experiments/fuse_chains.py — fusion across analysis chains.

Finding F13 concluded that fusion failed not because the machinery was wrong but
because the ingredients were: BM25 and LM-Dirichlet were properly decorrelated
(overlap@10 0.52) yet 0.083 nDCG@10 apart in quality, and rank-based fusion is
scale-blind, so it handed a much weaker ranker equal say.

The L1 sweep produced the missing ingredient. Two analysis chains disagree by
*vocabulary* rather than by scoring formula -- a stemmed index and an unstemmed
one retrieve genuinely different documents -- while scoring within ~0.01 of each
other. That is the decorrelated-AND-comparable pair F13 said was needed.

The cost is real and must be reported: fusing chains means persisting TWO
indexes, so the index-size component roughly doubles. That trade only clears if
the gain is real, which is what this measures.

Each chain is re-tuned on a dense k1/b grid first, because the coarse grid used
for chain *selection* understates every chain's ceiling.

Usage:
    python experiments/fuse_chains.py
    python experiments/fuse_chains.py --chains porter+nostop+splitan splitan
"""
import argparse
import json
import os
import sys
from itertools import combinations
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import (
    DEFAULT_DATA,
    QueryPostings,
    config_slug,
    cv_folds,
    get_index,
    load_topics,
    paired_bootstrap,
    smooth_surface,
    sweep,
)
from experiments.fuse import agreement
from experiments.sweep_analysis import CANDIDATES
from experiments.sweep_bm25 import log_rows
from submission import _scorers
from submission._scorers import CollectionStats
from submission._traverse import rrf_fuse

DEFAULT_CHAINS = ["porter+nostop+splitan", "splitan", "porter"]


def tune_chain(name: str, corpus_path: str, queries, qrels) -> Dict:
    """Dense k1/b search for one chain, with the honest CV value alongside."""
    config = CANDIDATES[name]
    index = get_index(corpus_path, config=config)
    k1_values = np.round(np.arange(0.6, 12.0 + 1e-9, 0.3), 3)
    b_values = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 3)
    configs = [{"k1": float(a), "b": float(b)} for a in k1_values for b in b_values]

    rows = sweep(index, queries, qrels, "bm25", configs)
    log_rows(rows, f"chainfine-{name}")

    axes = ("k1", "b")
    keyed = {tuple(round(r["params"][a], 6) for a in axes): r for r in rows}
    sm = smooth_surface(rows, axes)
    chosen = keyed[max(sm, key=lambda k: sm[k])]

    qids = sorted(rows[0]["per_query"])
    held_out = {}
    fold_params = []          # the parameters this chain would use per fold
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in set(test)]
        ts = {k: float(np.mean([r["per_query"][q] for q in train])) for k, r in keyed.items()}
        s = smooth_surface(rows, axes, ts)
        pick = keyed[max(s, key=lambda k: s[k])]
        fold_params.append(pick["params"])
        for q in test:
            held_out[q] = pick["per_query"][q]

    index_bytes = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _s, files in os.walk(f".index_cache-{config_slug(config)}") for f in files)
    return {"name": name, "index": index, "params": chosen["params"],
            "in_sample": chosen["ndcg@10"],
            "honest": float(np.mean(list(held_out.values()))),
            "per_query": chosen["per_query"], "held_out": held_out,
            "fold_params": fold_params, "index_mb": index_bytes / 1e6}


def make_run(entry: Dict, queries, depth: int) -> Dict:
    index = entry["index"]
    stats = CollectionStats(index.N, index.avg_doc_len, index.total_tokens)
    scorer = _scorers.get("bm25")
    params = _scorers.resolve_params("bm25", entry["params"])
    return {qid: QueryPostings(index, text).rank(scorer, params, stats, k=depth)
            for qid, text in queries}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--chains", nargs="+", default=DEFAULT_CHAINS)
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--rrf-k", type=float, default=60.0)
    args = ap.parse_args()

    corpus_path = os.path.join(args.data_dir, "corpus.jsonl")
    queries, qrels = load_topics(args.data_dir)

    entries = {}
    for name in args.chains:
        print(f"--- tuning {name} (dense grid) ---", flush=True)
        entries[name] = tune_chain(name, corpus_path, queries, qrels)
        e = entries[name]
        print(f"  {e['params']}  in-sample {e['in_sample']:.4f}  honest {e['honest']:.4f}  "
              f"index {e['index_mb']:.1f}MB")

    runs = {n: make_run(e, queries, args.depth) for n, e in entries.items()}

    print(f"\n{'='*80}\nPairwise agreement (fusion needs disagreement)\n{'='*80}")
    for a, b in combinations(args.chains, 2):
        ag = agreement(runs[a], runs[b], 10)
        gap = abs(entries[a]["honest"] - entries[b]["honest"])
        print(f"  {a:<24} vs {b:<24} overlap@10={ag['overlap']:.3f}  "
              f"tau={ag['kendall_tau']:.3f}  quality gap={gap:.4f}")

    best_name = max(entries, key=lambda n: entries[n]["honest"])
    best = entries[best_name]
    print(f"\n{'='*80}\nRRF fusion, cross-validated\n{'='*80}")
    print(f"  best single chain: {best_name}  honest {best['honest']:.4f}  "
          f"({best['index_mb']:.1f}MB)")

    qids = sorted(qrels)

    # Postings are independent of k1/b, so decode each query once per chain and
    # reuse across all folds and all combinations.
    cached = {n: {qid: QueryPostings(entries[n]["index"], text) for qid, text in queries}
              for n in args.chains}
    stats = {n: CollectionStats(entries[n]["index"].N, entries[n]["index"].avg_doc_len,
                                entries[n]["index"].total_tokens) for n in args.chains}
    scorer = _scorers.get("bm25")
    folds = cv_folds(qids, 5)

    # Nested CV. Each fold's ranked lists are built with parameters chosen from
    # that fold's TRAINING topics only, then fused and scored on the held-out
    # topics. Without this, the fusion number would carry the optimism of its
    # components' tuning while the single-chain baseline it is compared against
    # does not -- the same apples-to-oranges error finding F10 caught.
    fold_runs: List[Dict[str, Dict]] = []
    for f_idx in range(len(folds)):
        per_chain = {}
        for n in args.chains:
            params = _scorers.resolve_params("bm25", entries[n]["fold_params"][f_idx])
            per_chain[n] = {qid: cached[n][qid].rank(scorer, params, stats[n], k=args.depth)
                            for qid, _t in queries}
        fold_runs.append(per_chain)

    results = []
    for size in range(2, len(args.chains) + 1):
        for combo in combinations(args.chains, size):
            # Equal weights: any gain is attributable to fusion itself, not to a
            # weight fitted on the topics used for evaluation.
            honest_scores, in_sample_scores = {}, {}
            for f_idx, test in enumerate(folds):
                for qid in test:
                    fused = rrf_fuse([fold_runs[f_idx][n][qid] for n in combo], k=10,
                                     rrf_k=args.rrf_k, weights=[1.0] * len(combo))
                    honest_scores[qid] = ndcg_at_k([d for d, _ in fused], qrels[qid], k=10)
            for qid in qids:
                fused = rrf_fuse([runs[n][qid] for n in combo], k=10,
                                 rrf_k=args.rrf_k, weights=[1.0] * len(combo))
                in_sample_scores[qid] = ndcg_at_k([d for d, _ in fused], qrels[qid], k=10)
            mb = sum(entries[n]["index_mb"] for n in combo)
            st = paired_bootstrap(honest_scores, best["held_out"])
            results.append((combo, float(np.mean(list(honest_scores.values()))),
                            float(np.mean(list(in_sample_scores.values()))), mb, st))

    results.sort(key=lambda r: -r[1])
    print(f"\n{'combination':<50} {'honest':>8} {'in-samp':>8} {'MB':>7} {'delta':>8} {'p':>7}")
    for combo, honest, in_samp, mb, st in results:
        print(f"  {'+'.join(combo):<48} {honest:>8.4f} {in_samp:>8.4f} {mb:>7.1f} "
              f"{st['delta']:>+8.4f} {st['p_value']:>7.4f}")
    print(f"\n  (honest = nested CV: fold parameters chosen on training topics only.")
    print(f"   in-samp = same chains tuned on all 50 topics; the gap is the optimism.)")

    print(f"\n{'='*80}\nIndex-size trade (plan.md 6.1b decision rule)\n{'='*80}")
    naive_median_estimate = 279.0
    for combo, score, _in_samp, mb, st in results[:3]:
        gain = score - best["honest"]
        ratio = mb / naive_median_estimate
        rule = "PASS" if (gain >= 0.01 and ratio <= 0.5) else "FAIL"
        print(f"  {'+'.join(combo):<44} gain={gain:+.4f}  {mb:.1f}MB "
              f"({ratio:.2f}x est. median)  -> {rule}")

    out = os.path.join(os.path.dirname(__file__), "fusion_chains.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"best_single": {"chain": best_name, "honest": best["honest"],
                                   "params": best["params"], "index_mb": best["index_mb"]},
                   "fusions": [{"chains": list(c), "ndcg": s, "index_mb": m,
                                "delta": st["delta"], "p": st["p_value"]}
                               for c, s, _i, m, st in results]}, f, indent=2)
    print(f"\nwrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
