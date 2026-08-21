#!/usr/bin/env python
"""
experiments/dedup_probe.py — does removing near-duplicates from the top-k help?

Measured on the dev set: 43 of 50 topics have at least one pair of near-duplicate
documents (Jaccard > 0.9 over term sets) inside their top-10, 98 such pairs in
all. CORD-19 is well known for holding the same paper multiple times from
different sources. Topic 1's top-10 contains two documents with Jaccard 1.00,
*both* judged non-relevant, occupying two of ten slots.

That is a different failure from bad scoring, and nothing tried so far addresses
it: every previous idea changed how documents are *ranked*, while this changes
which of the ranked documents are *shown*. nDCG@10 has exactly ten slots, so a
slot spent on a redundant copy is a slot not spent on new evidence.

The effect is not obviously positive, which is why it gets measured rather than
assumed. Dropping a duplicate pair of non-relevant documents frees a slot for
something possibly relevant -- a gain. Dropping one of a duplicate pair of
*relevant* documents (topic 2 has two grade-2 duplicates) costs a known hit for
a speculative one -- a loss. Which dominates is an empirical question.

This probe uses an in-memory forward index, which is far too large to ship. If
the result is positive the production form is a per-document fingerprint
(SimHash over term ids, 8 bytes per document, ~1.4MB) compared by Hamming
distance -- cheap in both space and query time. Measure first, build second.

Usage:
    python experiments/dedup_probe.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import cv_folds, get_index, load_topics, paired_bootstrap
from submission import bm25
from submission._codecs import unpack_tf_nibbles, vbyte_decode

K1, B = 4.5, 0.60


def forward_sets(index):
    """doc -> set of term ids. Experiment-only; ~390MB."""
    tot = int(index.df.sum())
    gaps = vbyte_decode(index._docid_buf, tot)
    term_of = np.repeat(np.arange(len(index.terms)), index.df)
    run = np.cumsum(gaps)
    base = np.zeros(len(index.terms), dtype=np.int64)
    base[1:] = run[index._term_start[1:] - 1]
    doc_of = run - np.repeat(base, index.df)
    order = np.lexsort((term_of, doc_of))
    doc_of, term_of = doc_of[order], term_of[order]
    off = np.zeros(index.N + 1, dtype=np.int64)
    np.cumsum(np.bincount(doc_of, minlength=index.N), out=off[1:])
    return term_of, off


def dedup(ranked, ext2int, term_of, off, threshold, k=10):
    """Greedy: keep the highest-scoring document, then skip any later document
    too similar to one already kept. Order among kept documents is unchanged."""
    kept, kept_sets = [], []
    for doc_id, score in ranked:
        d = ext2int[doc_id]
        s = set(term_of[off[d]:off[d + 1]].tolist())
        if not s:
            continue
        redundant = False
        for prev in kept_sets:
            inter = len(s & prev)
            if inter and inter / len(s | prev) >= threshold:
                redundant = True
                break
        if not redundant:
            kept.append((doc_id, score))
            kept_sets.append(s)
            if len(kept) == k:
                break
    return kept


def main():
    index = get_index(os.path.join(os.path.dirname(__file__), "..",
                                   "data", "full", "corpus.jsonl"))
    bm25.build(index)
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]
    print("building forward sets (experiment only) ...", flush=True)
    term_of, off = forward_sets(index)
    ext2int = {e: i for i, e in enumerate(index.doc_ids)}

    # Retrieve deeper than 10 so dedup has replacements to promote.
    DEPTH = 60
    deep = {q: bm25.score(t, DEPTH, k1=K1, b=B) for q, t in qs}
    base = {q: ndcg_at_k([d for d, _ in deep[q][:10]], qrels[q], k=10) for q, _t in qs}
    bm = float(np.mean(list(base.values())))
    print(f"\nbaseline (plain top-10): {bm:.4f}\n")
    print(f"{'jaccard threshold':>18}{'nDCG@10':>10}{'delta':>9}{'dropped/query':>15}")

    results = {}
    for th in (1.01, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        sc, dropped = {}, []
        for q, _t in qs:
            kept = dedup(deep[q], ext2int, term_of, off, th)
            dropped.append(10 - sum(1 for x in kept[:10] if x in deep[q][:10]))
            sc[q] = ndcg_at_k([d for d, _ in kept], qrels[q], k=10)
        results[th] = sc
        m = float(np.mean(list(sc.values())))
        tag = "  (no-op)" if th > 1 else ""
        print(f"{th:>18.2f}{m:>10.4f}{m-bm:>+9.4f}{np.mean(dropped):>15.2f}{tag}")

    qids = sorted(base)
    held = {}
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in set(test)]
        pick = max(results, key=lambda th: np.mean([results[th][q] for q in train]))
        for q in test:
            held[q] = results[pick][q]
    best = max(results, key=lambda th: np.mean(list(results[th].values())))
    st = paired_bootstrap(held, base)
    print(f"\n{'='*66}")
    print(f"best in-sample : threshold {best} -> "
          f"{np.mean(list(results[best].values())):.4f} "
          f"({np.mean(list(results[best].values()))-bm:+.4f})")
    print(f"honest CV      : {np.mean(list(held.values())):.4f}  ({st['delta']:+.4f})  "
          f"p={st['p_value']:.4f}  W/L/T={st['wins']}/{st['losses']}/{st['ties']}")
    print(f"verdict        : {'SIGNIFICANT' if st['p_value'] < 0.05 else 'not significant'}")


if __name__ == "__main__":
    main()
