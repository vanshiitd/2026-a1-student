#!/usr/bin/env python
"""
experiments/field_probe.py — does a positional pseudo-title field help?

Motivation. These documents are a title concatenated directly onto an abstract
with no delimiter ("Role of endothelin-1 in lung disease Endothelin-1 (ET-1)
is a 21 amino acid peptide..."), and titles carry no terminal punctuation, so
the field boundary cannot be recovered exactly. But the underlying signal --
terms appearing early in a document are more indicative of what it is about --
does not require an exact boundary. Treating the first W tokens as a pseudo-title
field is a smooth approximation of BM25F:

    score(d) = BM25(d over full text) + lambda * BM25(d over first W tokens)

This is document-agnostic: every collection puts its most indicative text first.

Why a probe and not an implementation: shipping this means storing a second term
frequency per posting, which changes the index format. Finding F19 established
the discipline -- measure whether the signal exists before paying for the
structure to hold it. The pseudo-title index here is built in memory and
discarded.

Usage:
    python experiments/field_probe.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import cv_folds, load_topics, paired_bootstrap
from submission._analysis import analyze
from submission._scorers import robertson_idf
from submission.indexer import InvertedIndex, _iter_jsonl

REPO = os.path.join(os.path.dirname(__file__), "..")
WIDTHS = (10, 15, 20, 30, 50)
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def build_title_index(corpus_path, width, config):
    """An index over only the first `width` tokens of each document."""
    docs = []
    for doc_id, text in _iter_jsonl(corpus_path):
        docs.append((doc_id, " ".join(analyze(text, config)[:width])))
    ix = InvertedIndex(config)
    ix.build(docs)
    return ix


def bm25_scores(index, terms, k1, b, out):
    """Accumulate BM25 into `out` (indexed by that index's internal doc ids)."""
    avgdl = index.avg_doc_len or 1.0
    touched = np.zeros(index.N, dtype=bool)
    for term in terms:
        tid = index.term_id(term)
        if tid < 0:
            continue
        d, tf = index.postings_by_id(tid)
        if d.size == 0:
            continue
        dl = index.doc_len[d].astype(np.float64)
        idf = robertson_idf(int(index.df[tid]), index.N)
        out[d] += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
        touched[d] = True
    return touched


def main():
    corpus = os.path.join(REPO, "data", "full", "corpus.jsonl")
    full = InvertedIndex.load(os.path.join(REPO, ".index_cache"))
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]
    k1, b = 4.5, 0.60

    # Baseline: full-text BM25 only.
    base = {}
    for q, t in qs:
        terms = list(dict.fromkeys(analyze(t, full.config)))
        s = np.zeros(full.N)
        touched = bm25_scores(full, terms, k1, b, s)
        cand = np.flatnonzero(touched)
        order = np.lexsort((cand, -s[cand]))[:10]
        base[q] = ndcg_at_k([full.doc_ids[int(cand[i])] for i in order], qrels[q], k=10)
    bm = float(np.mean(list(base.values())))
    print(f"baseline full-text BM25: {bm:.4f}\n")

    results = {}
    print(f"{'W':>4} {'lambda':>8} {'nDCG@10':>9} {'delta':>9}")
    for width in WIDTHS:
        print(f"  building pseudo-title index, W={width} ...", flush=True)
        title = build_title_index(corpus, width, full.config)
        assert title.doc_ids == full.doc_ids, "doc order must match for id reuse"

        for lam in LAMBDAS:
            sc = {}
            for q, t in qs:
                terms = list(dict.fromkeys(analyze(t, full.config)))
                s = np.zeros(full.N)
                touched = bm25_scores(full, terms, k1, b, s)
                if lam:
                    ts = np.zeros(title.N)
                    t_touch = bm25_scores(title, terms, k1, b, ts)
                    s += lam * ts
                    touched |= t_touch
                cand = np.flatnonzero(touched)
                if cand.size == 0:
                    sc[q] = 0.0
                    continue
                order = np.lexsort((cand, -s[cand]))[:10]
                sc[q] = ndcg_at_k([full.doc_ids[int(cand[i])] for i in order], qrels[q], k=10)
            m = float(np.mean(list(sc.values())))
            results[(width, lam)] = sc
            flag = "  <--" if m > bm else ""
            print(f"{width:>4} {lam:>8.2f} {m:>9.4f} {m - bm:>+9.4f}{flag}")

    best_key = max(results, key=lambda k: np.mean(list(results[k].values())))
    best = results[best_key]
    st = paired_bootstrap(best, base)
    print(f"\nBEST in-sample: W={best_key[0]}, lambda={best_key[1]} -> "
          f"{np.mean(list(best.values())):.4f}")
    print(f"paired bootstrap vs baseline: delta={st['delta']:+.4f} p={st['p_value']:.4f} "
          f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}")

    # Honest value of the whole procedure: pick (W, lambda) on training folds only.
    qids = sorted(base)
    held = {}
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in set(test)]
        pick = max(results, key=lambda k: np.mean([results[k][q] for q in train]))
        for q in test:
            held[q] = results[pick][q]
    st2 = paired_bootstrap(held, base)
    honest = float(np.mean(list(held.values())))
    print(f"\nHONEST cross-validated: {honest:.4f} vs baseline {bm:.4f}  "
          f"delta={st2['delta']:+.4f}  p={st2['p_value']:.4f}")
    print(f"\nDecision rule: ship only if the honest gain clears +0.01, since this "
          f"costs a second term frequency per posting.")


if __name__ == "__main__":
    main()
