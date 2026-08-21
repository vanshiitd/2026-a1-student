#!/usr/bin/env python
"""
experiments/structure_probe.py — push the structural signal that F24 found.

F24 established that early-position evidence helps (+0.0114, p=0.011) via a
fixed 10-token pseudo-title field. Three follow-ups, all in the areas the
assignment names as separating a B-grade system from an A-grade one
("term-weighting design ... how you handle short queries and rare terms"):

  A. **Per-field BM25 parameters.** The title field inherits the body's
     k1=4.5, b=0.60. But that field is a fixed 10-token window, so nearly every
     document has the same length in it and length normalisation is close to
     meaningless -- b should arguably be 0. Body and title are different
     distributions and there is no reason one (k1, b) fits both.

  B. **A second positional band.** If tokens [0,10) carry extra signal, tokens
     [10,40) plausibly carry some too, at a lower weight. This is graded
     position weighting rather than a single binary field.

  C. **A sentence-delimited first field.** The assignment permits exactly one
     structural assumption -- "all documents ... have sentences terminated by a
     period" -- and a fixed token count is a crude proxy for it. A first-sentence
     field adapts to each document instead of assuming a global width, which
     should transfer better to a held-out collection whose titles run longer or
     shorter than this one's.

Evaluation discipline follows F24's lesson: each change is tested as a small,
pre-committed variation against the CURRENT shipped configuration, not searched
over a wide grid. Wide grids made selection noise bury a real effect once
already.

Usage:
    python experiments/structure_probe.py
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import load_topics, paired_bootstrap
from submission._analysis import analyze
from submission._scorers import robertson_idf
from submission.indexer import InvertedIndex, _iter_jsonl

REPO = os.path.join(os.path.dirname(__file__), "..")
BODY_K1, BODY_B = 4.5, 0.60
SHIPPED_W, SHIPPED_LAMBDA = 10, 0.10

# The single structural assumption the assignment allows.
_SENTENCE_END = re.compile(r"(?<=\.)\s+")


def build_field(corpus_path, config, transform):
    """Index a derived view of each document (`transform` maps text -> text)."""
    docs = [(doc_id, transform(text)) for doc_id, text in _iter_jsonl(corpus_path)]
    ix = InvertedIndex(config)
    ix.store_doc_ids = False
    ix.build(docs)
    return ix


def window(config, lo, hi):
    return lambda text: " ".join(analyze(text, config)[lo:hi])


def first_sentences(n=1):
    def f(text):
        parts = _SENTENCE_END.split(text, maxsplit=n)
        return " ".join(parts[:n])
    return f


def accumulate(index, terms, scores, touched, k1, b, weight):
    avgdl = index.avg_doc_len or 1.0
    for term in terms:
        tid = index.term_id(term)
        if tid < 0:
            continue
        d, tf = index.postings_by_id(tid)
        if d.size == 0:
            continue
        dl = index.doc_len[d].astype(np.float64)
        idf = robertson_idf(int(index.df[tid]), index.N)
        scores[d] += weight * idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
        touched[d] = True


def evaluate(body, fields, qs, qrels, config):
    """fields: list of (index, k1, b, weight)."""
    out = {}
    for q, text in qs:
        terms = list(dict.fromkeys(analyze(text, config)))
        s = np.zeros(body.N)
        touched = np.zeros(body.N, dtype=bool)
        accumulate(body, terms, s, touched, BODY_K1, BODY_B, 1.0)
        for ix, k1, b, w in fields:
            if w:
                accumulate(ix, terms, s, touched, k1, b, w)
        cand = np.flatnonzero(touched)
        if cand.size == 0:
            out[q] = 0.0
            continue
        order = np.lexsort((cand, -s[cand]))[:10]
        out[q] = ndcg_at_k([body.doc_ids[int(cand[i])] for i in order], qrels[q], k=10)
    return out


def report(name, sc, ref, ref_mean):
    m = float(np.mean(list(sc.values())))
    st = paired_bootstrap(sc, ref)
    flag = "  <--" if m > ref_mean else ""
    print(f"  {name:<46} {m:.4f}  {m - ref_mean:+.4f}  p={st['p_value']:.4f}{flag}")
    return m


def main():
    corpus = os.path.join(REPO, "data", "full", "corpus.jsonl")
    body = InvertedIndex.load(os.path.join(REPO, ".index_cache"))
    cfg = body.config
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]

    print("building field indexes ...", flush=True)
    w0_10 = build_field(corpus, cfg, window(cfg, 0, SHIPPED_W))
    w10_40 = build_field(corpus, cfg, window(cfg, SHIPPED_W, 40))
    sent1 = build_field(corpus, cfg, first_sentences(1))
    print(f"  first-10-token field : avg len {w0_10.avg_doc_len:.1f}, "
          f"{len(w0_10.terms):,} terms")
    print(f"  tokens 10-40 field   : avg len {w10_40.avg_doc_len:.1f}, "
          f"{len(w10_40.terms):,} terms")
    print(f"  first-sentence field : avg len {sent1.avg_doc_len:.1f}, "
          f"{len(sent1.terms):,} terms")

    base_only = evaluate(body, [], qs, qrels, cfg)
    bm = float(np.mean(list(base_only.values())))
    shipped = evaluate(body, [(w0_10, BODY_K1, BODY_B, SHIPPED_LAMBDA)], qs, qrels, cfg)
    sm = float(np.mean(list(shipped.values())))
    print(f"\nbody only          {bm:.4f}")
    print(f"SHIPPED (+title)   {sm:.4f}   <- everything below is measured against this\n")

    print("A. per-field BM25 parameters for the title field")
    for k1, b in ((BODY_K1, 0.0), (BODY_K1, 0.3), (1.2, 0.0), (1.2, BODY_B), (8.0, 0.0)):
        report(f"title k1={k1}, b={b}",
               evaluate(body, [(w0_10, k1, b, SHIPPED_LAMBDA)], qs, qrels, cfg), shipped, sm)

    print("\nB. second positional band, tokens 10-40")
    for w2 in (0.02, 0.05, 0.10):
        report(f"+ band[10,40) weight {w2}",
               evaluate(body, [(w0_10, BODY_K1, BODY_B, SHIPPED_LAMBDA),
                               (w10_40, BODY_K1, BODY_B, w2)], qs, qrels, cfg), shipped, sm)

    print("\nC. sentence-delimited first field (replaces the fixed window)")
    for lam in (0.05, 0.10, 0.15):
        report(f"first-sentence field, weight {lam}",
               evaluate(body, [(sent1, BODY_K1, BODY_B, lam)], qs, qrels, cfg), shipped, sm)
    print("   (and combined with the fixed window)")
    for lam in (0.05, 0.10):
        report(f"title + first-sentence, weight {lam}",
               evaluate(body, [(w0_10, BODY_K1, BODY_B, SHIPPED_LAMBDA),
                               (sent1, BODY_K1, BODY_B, lam)], qs, qrels, cfg), shipped, sm)


if __name__ == "__main__":
    main()
