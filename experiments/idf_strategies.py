#!/usr/bin/env python
"""
experiments/idf_strategies.py — three strategies aimed at finding F17.

F17 diagnosed the remaining loss as *precision*: 31% of top-10 slots hold
completely non-relevant documents, and the worst topics all fail the same way --
the discriminative term is rare ("flu", "Canada", "origin") while the COVID
terms are near-ubiquitous. Measured here: IDF ratios within a single query reach
35x, and the rarest term carries only 22% of the query's total IDF mass. With
k1=4.5 amplifying term frequency, a document repeating common terms outscores
one containing the rare term.

Multiplicative IDF coverage was already tried and failed (F18) because BM25's
IDF-weighted sum *already* encodes coverage, so scaling by it double-counts.
These three do something different -- they change which terms are scored, how
steeply rarity is rewarded, and which documents are eligible at all:

  A. drop_df_frac  -- omit query terms appearing in more than this fraction of
     the collection. Not the same as the index-side stopword removal rejected in
     F14: document lengths, df and the index are untouched, so only the scoring
     of near-ubiquitous query terms changes.

  B. idf_power     -- weight each term by IDF^p. p > 1 steepens the preference
     for rare terms without altering the tf saturation that k1 controls.

  C. require_rare  -- a hard eligibility filter: a document must contain at
     least one of the query's `n` rarest terms to be ranked at all. Unlike F18's
     reweighting, this removes candidates rather than rescoring them, which is
     what a precision problem actually calls for.

All three are evaluated with the same nested cross-validation as everything
else, because F20 established that in-sample gains on 50 topics run 0.014-0.018
above honest ones.

Usage:
    python experiments/idf_strategies.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import cv_folds, get_index, load_topics, paired_bootstrap
from submission._analysis import analyze
from submission._scorers import robertson_idf

K1, B = 4.5, 0.60


class Q:
    """Per-query postings and IDFs, decoded once and reused across settings."""

    def __init__(self, index, text):
        self.terms = []
        for t in dict.fromkeys(analyze(text, index.config)):
            tid = index.term_id(t)
            if tid < 0:
                continue
            d, tf = index.postings_by_id(tid)
            if d.size == 0:
                continue
            df = int(index.df[tid])
            self.terms.append((robertson_idf(df, index.N), df, d, tf))
        self.terms.sort(key=lambda x: -x[0])          # rarest (highest idf) first


def rank(index, q, k=10, idf_power=1.0, drop_df_frac=1.0, require_rare=0):
    N = index.N
    avgdl = index.avg_doc_len or 1.0
    scores = np.zeros(N)
    touched = np.zeros(N, dtype=bool)
    eligible = None

    ceiling = drop_df_frac * N
    for rank_i, (idf, df, d, tf) in enumerate(q.terms):
        if df > ceiling:
            continue
        if require_rare and rank_i < require_rare:
            eligible = d if eligible is None else np.union1d(eligible, d)
        dl = index.doc_len[d].astype(np.float64)
        norm = K1 * (1.0 - B + B * (dl / avgdl))
        scores[d] += (idf ** idf_power) * (tf * (K1 + 1.0)) / (tf + norm)
        touched[d] = True

    if require_rare and eligible is not None:
        mask = np.zeros(N, dtype=bool)
        mask[eligible] = True
        touched &= mask

    cand = np.flatnonzero(touched)
    if cand.size == 0:
        return []
    v = scores[cand]
    order = np.lexsort((cand, -v))[:k]
    return [(index.doc_ids[int(cand[i])], float(v[i])) for i in order]


def main():
    index = get_index(os.path.join(os.path.dirname(__file__), "..",
                                   "data", "full", "corpus.jsonl"))
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]
    cache = {q: Q(index, t) for q, t in qs}

    def evaluate(**kw):
        return {q: ndcg_at_k([d for d, _ in rank(index, cache[q], **kw)], qrels[q], k=10)
                for q, _t in qs}

    base = evaluate()
    bm = float(np.mean(list(base.values())))
    print(f"baseline BM25 (k1={K1}, b={B}): {bm:.4f}\n")

    results = {}

    print("A. drop query terms above a document-frequency fraction")
    print(f"{'drop_df_frac':>14}{'nDCG@10':>10}{'delta':>9}")
    for frac in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
        sc = evaluate(drop_df_frac=frac)
        results[("drop", frac)] = sc
        m = float(np.mean(list(sc.values())))
        print(f"{frac:>14.2f}{m:>10.4f}{m-bm:>+9.4f}")

    print("\nB. IDF exponent")
    print(f"{'idf_power':>14}{'nDCG@10':>10}{'delta':>9}")
    for p in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        sc = evaluate(idf_power=p)
        results[("power", p)] = sc
        m = float(np.mean(list(sc.values())))
        print(f"{p:>14.2f}{m:>10.4f}{m-bm:>+9.4f}")

    print("\nC. require a document to contain one of the n rarest query terms")
    print(f"{'require_rare':>14}{'nDCG@10':>10}{'delta':>9}")
    for n in (0, 1, 2, 3, 4):
        sc = evaluate(require_rare=n)
        results[("rare", n)] = sc
        m = float(np.mean(list(sc.values())))
        print(f"{n:>14d}{m:>10.4f}{m-bm:>+9.4f}")

    # Joint grid over the two continuous axes, then an honest estimate of the
    # whole "sweep and pick" procedure via nested CV.
    print("\nD. joint (drop_df_frac x idf_power)")
    joint = {}
    for frac in (1.0, 0.3, 0.1, 0.05):
        row = []
        for p in (1.0, 1.25, 1.5, 2.0, 2.5):
            sc = evaluate(drop_df_frac=frac, idf_power=p)
            joint[(frac, p)] = sc
            row.append(float(np.mean(list(sc.values()))))
        print(f"  drop={frac:<5} " + "  ".join(f"p={p}:{v:.4f}" for p, v in
                                               zip((1.0, 1.25, 1.5, 2.0, 2.5), row)))

    qids = sorted(base)
    held = {}
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in set(test)]
        pick = max(joint, key=lambda kk: np.mean([joint[kk][q] for q in train]))
        for q in test:
            held[q] = joint[pick][q]
    best_key = max(joint, key=lambda kk: np.mean(list(joint[kk].values())))
    st = paired_bootstrap(held, base)
    print(f"\n{'='*66}")
    print(f"best in-sample : drop={best_key[0]}, p={best_key[1]} -> "
          f"{np.mean(list(joint[best_key].values())):.4f} "
          f"({np.mean(list(joint[best_key].values()))-bm:+.4f})")
    print(f"honest CV      : {np.mean(list(held.values())):.4f} ({st['delta']:+.4f})  "
          f"p={st['p_value']:.4f}  W/L/T={st['wins']}/{st['losses']}/{st['ties']}")
    print(f"verdict        : {'SIGNIFICANT' if st['p_value'] < 0.05 else 'not significant'}")


if __name__ == "__main__":
    main()
