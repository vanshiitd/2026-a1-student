#!/usr/bin/env python
"""
experiments/rm3_probe.py — feasibility probe for RM3 pseudo-relevance feedback.

Finding F17 showed the remaining headroom is *precision*, not reordering: 31% of
our top-10 slots hold completely non-relevant documents, and no amount of
reordering what we already retrieve can recover that. Only retrieving different
documents can. RM3 is the classical technique that does exactly that -- it
rewrites the query, so it changes the retrieved set rather than its order. That
is why it is worth trying after SDM, coverage and fusion all failed: those three
reorder, RM3 does not.

RM3 (Lavrenko & Croft 2001; Abdul-Jaleel et al. 2004):

    p(w|R)  =  sum over top-F docs of  p(w|d) * p(q|d)          (RM1)
    q'      =  alpha * q_original  +  (1 - alpha) * top-m of RM1  (RM3)

Why this is a *probe* and not an implementation
-----------------------------------------------
Estimating p(w|d) needs the full term vector of each feedback document, i.e. a
forward (doc -> terms) index. This index is inverted only, and adding a forward
index would cost roughly another 48MB on disk -- a real charge against the
index-size component.

So this script does not build one. It transposes the whole postings file into
memory once (~390MB, fine for an experiment, unshippable in a submission) and
answers the only question that matters first: does RM3 help enough to be worth
paying for? Build the persistent structure only if the answer is yes. Measuring
before investing is the point.

Usage:
    python experiments/rm3_probe.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import cv_folds, load_topics, paired_bootstrap
from submission._analysis import analyze
from submission._codecs import unpack_tf_nibbles, vbyte_decode
from submission._scorers import robertson_idf
from submission.indexer import InvertedIndex


def build_forward_in_memory(index):
    """Transpose every posting into doc -> (term_ids, tfs). Experiment-only."""
    n_terms = len(index.terms)
    total = int(index.df.sum())
    gaps = vbyte_decode(index._docid_buf, total)
    tfs = unpack_tf_nibbles(index._tf_packed, 0, total,
                            index._tf_exc_idx, index._tf_exc_val)

    term_of = np.repeat(np.arange(n_terms, dtype=np.int64), index.df)
    starts = np.empty(n_terms, dtype=np.int64)
    starts[0] = 0
    np.cumsum(index.df[:-1], out=starts[1:])
    running = np.cumsum(gaps)
    base = np.zeros(n_terms, dtype=np.int64)
    base[1:] = running[starts[1:] - 1]
    doc_of = running - np.repeat(base, index.df)

    order = np.lexsort((term_of, doc_of))
    doc_of, term_of, tfs = doc_of[order], term_of[order], tfs[order]
    counts = np.bincount(doc_of, minlength=index.N)
    offsets = np.zeros(index.N + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return term_of, tfs, offsets


def bm25_weighted(index, weights, k1, b, k=10, depth=None):
    """BM25 where each query term carries an explicit weight."""
    N, avgdl = index.N, index.avg_doc_len or 1.0
    scores = np.zeros(N)
    touched = np.zeros(N, dtype=bool)
    for tid, w in weights.items():
        if w <= 0:
            continue
        d, tf = index.postings_by_id(tid)
        if d.size == 0:
            continue
        dl = index.doc_len[d].astype(np.float64)
        idf = robertson_idf(int(index.df[tid]), N)
        scores[d] += w * idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
        touched[d] = True
    cand = np.flatnonzero(touched)
    if cand.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    limit = depth or k
    v = scores[cand]
    if cand.size > limit:
        top = np.argpartition(-v, limit - 1)[:limit]
        cand, v = cand[top], v[top]
    order = np.lexsort((cand, -v))
    return cand[order], v[order]


def rm3(index, fwd, query, k1, b, fb_docs=10, fb_terms=20, alpha=0.5, k=10):
    term_of, tfs_all, offsets = fwd
    base_w = defaultdict(float)
    for t in analyze(query, index.config):
        tid = index.term_id(t)
        if tid >= 0:
            base_w[tid] += 1.0
    if not base_w:
        return []
    total = sum(base_w.values())
    for tid in base_w:
        base_w[tid] /= total

    docs, scores = bm25_weighted(index, base_w, k1, b, k=k, depth=fb_docs)
    if docs.size == 0:
        return []

    # p(q|d) from the first-pass scores, normalised to a distribution.
    w_doc = scores - scores.min()
    w_doc = w_doc / w_doc.sum() if w_doc.sum() > 0 else np.ones(docs.size) / docs.size

    rm = defaultdict(float)
    for d, wd in zip(docs, w_doc):
        lo, hi = offsets[d], offsets[d + 1]
        dl = max(int(index.doc_len[d]), 1)
        for tid, tf in zip(term_of[lo:hi], tfs_all[lo:hi]):
            rm[int(tid)] += wd * (tf / dl)      # p(w|d) * p(q|d)

    top = sorted(rm.items(), key=lambda kv: -kv[1])[:fb_terms]
    mass = sum(v for _t, v in top) or 1.0

    final = defaultdict(float)
    for tid, w in base_w.items():
        final[tid] += alpha * w
    for tid, v in top:
        final[tid] += (1 - alpha) * (v / mass)

    docs, vals = bm25_weighted(index, final, k1, b, k=k)
    return [(index.doc_ids[int(d)], float(v)) for d, v in zip(docs, vals)]


def main():
    index = InvertedIndex.load(os.path.join(os.path.dirname(__file__), "..", ".index_cache"))
    queries, qrels = load_topics()
    print("transposing postings in memory (experiment-only) ...", flush=True)
    fwd = build_forward_in_memory(index)
    print(f"forward vectors ready for {index.N:,} docs\n")

    k1, b = 4.5, 0.60
    base = {}
    for qid, text in queries:
        if qid not in qrels:
            continue
        w = defaultdict(float)
        for t in analyze(text, index.config):
            tid = index.term_id(t)
            if tid >= 0:
                w[tid] = 1.0
        d, v = bm25_weighted(index, w, k1, b, k=10)
        base[qid] = ndcg_at_k([index.doc_ids[int(x)] for x in d], qrels[qid], k=10)
    bm = float(np.mean(list(base.values())))
    print(f"baseline BM25: {bm:.4f}  (expect 0.6281)\n")

    print(f"{'F':>4} {'m':>4} {'alpha':>6} {'nDCG@10':>9} {'delta':>8}")
    results = {}
    for fb_docs in (5, 10, 20):
        for fb_terms in (10, 20, 40):
            for alpha in (0.4, 0.6, 0.8):
                sc = {}
                for qid, text in queries:
                    if qid not in qrels:
                        continue
                    r = rm3(index, fwd, text, k1, b, fb_docs, fb_terms, alpha)
                    sc[qid] = ndcg_at_k([d for d, _ in r], qrels[qid], k=10)
                m = float(np.mean(list(sc.values())))
                results[(fb_docs, fb_terms, alpha)] = sc
                flag = "  <--" if m > bm else ""
                print(f"{fb_docs:>4} {fb_terms:>4} {alpha:>6.1f} {m:>9.4f} {m-bm:>+8.4f}{flag}")

    best_key = max(results, key=lambda kk: np.mean(list(results[kk].values())))
    best = results[best_key]
    st = paired_bootstrap(best, base)
    print(f"\nBEST in-sample: F={best_key[0]} m={best_key[1]} alpha={best_key[2]} "
          f"-> {np.mean(list(best.values())):.4f}")
    print(f"paired bootstrap vs BM25: delta={st['delta']:+.4f} p={st['p_value']:.4f} "
          f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}  "
          f"[{'SIGNIFICANT' if st['p_value'] < 0.05 else 'not significant'}]")

    # Honest value of the whole procedure, selecting settings per training fold.
    qids = sorted(base)
    held = {}
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in set(test)]
        pick = max(results, key=lambda kk: np.mean([results[kk][q] for q in train]))
        for q in test:
            held[q] = results[pick][q]
    st2 = paired_bootstrap(held, base)
    print(f"\nHONEST cross-validated RM3: {np.mean(list(held.values())):.4f}  "
          f"vs BM25 {bm:.4f}   delta={st2['delta']:+.4f}  p={st2['p_value']:.4f}")
    print(f"\nForward index would cost ~48MB on disk; decision rule needs >= +0.01.")


if __name__ == "__main__":
    main()
