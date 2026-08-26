#!/usr/bin/env python
"""
experiments/stem_expand.py — stemming as query expansion, not index conflation.

F26 and F28 both tested stemming and both rejected it, at p=0.63 with a 24/19
win-loss split that is indistinguishable from a coin flip. But both tested the
*same* form of stemming: build the index over stems, so "infection",
"infections" and "infected" collapse into one term before anything is scored.

That form has a cost the win-loss split hides. Conflation is irreversible and
symmetric: it buys recall on the queries where the variant matters, and pays
precision on the queries where it does not, because an exact match for the
user's actual word is no longer distinguishable from a match on a cousin of it.
Averaged over 50 topics those cancel, which is exactly the coin flip observed.

This script tests the asymmetric version instead. The index stays UNSTEMMED, so
an exact match is still an exact match at full weight. Each query term is then
additionally expanded to its stem-mates -- the other vocabulary terms sharing
its Porter stem -- at a reduced weight gamma:

    score(q) = sum over query terms t of [ 1.0 * BM25(t)
                                         + gamma * sum over mates m of BM25(m) ]

gamma = 0 is exactly the shipped system. gamma = 1 approximates index-level
stemming. Anything in between is a setting the earlier experiments could not
express, because a conflated index has no way to prefer the word the user
actually typed over its morphological cousins.

Two weightings for the mates:
  flat   every mate shares gamma equally
  idf    mates are weighted by their own idf, so a rare variant counts for
         more than a common one -- the same logic BM25 already applies to
         query terms, extended to the expansion

    python experiments/stem_expand.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.metrics import ndcg_at_k  # noqa: E402
from experiments.evaluate import cv_folds, load_topics, paired_bootstrap  # noqa: E402
from experiments.rm3_refine import idf_vector  # noqa: E402
from experiments.structure_probe import accumulate, build_field, evaluate, window  # noqa: E402
from submission._analysis import analyze  # noqa: E402
from submission.indexer import InvertedIndex  # noqa: E402

K1, B = 4.5, 0.60
TITLE_W, TITLE_LAMBDA = 10, 0.10
K = 10


def build_stem_map(terms):
    """stem -> [term ids sharing it]. Only groups of 2+ are useful."""
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    groups = defaultdict(list)
    for i, t in enumerate(terms):
        try:
            groups[ps.stem(t)].append(i)
        except Exception:
            groups[t].append(i)
    return {s: v for s, v in groups.items() if len(v) > 1}, ps


class StemExpander:
    def __init__(self, body, title):
        self.body, self.title = body, title
        self.term_id = {t: i for i, t in enumerate(body.terms)}
        self.idf = idf_vector(body)
        self.groups, self.ps = build_stem_map(body.terms)
        sizes = [len(v) for v in self.groups.values()]
        print(f"  {len(self.groups):,} stems with >1 surface form; "
              f"mean group {np.mean(sizes):.2f}, max {max(sizes)}")

    def expand(self, query, gamma, weighting):
        terms = list(dict.fromkeys(analyze(query, self.body.config)))
        if not terms:
            return {}
        w = defaultdict(float)
        for t in terms:
            w[t] += 1.0
            if gamma <= 0:
                continue
            tid = self.term_id.get(t)
            if tid is None:
                continue
            mates = [m for m in self.groups.get(self.ps.stem(t), []) if m != tid]
            if not mates:
                continue
            if weighting == "idf":
                mi = self.idf[mates]
                tot = float(mi.sum()) or 1.0
                for m, mw in zip(mates, mi):
                    w[self.body.terms[m]] += gamma * float(mw) / tot
            else:
                for m in mates:
                    w[self.body.terms[m]] += gamma / len(mates)
        return dict(w)

    def rank(self, query, gamma, weighting):
        w = self.expand(query, gamma, weighting)
        if not w:
            return []
        s = np.zeros(self.body.N)
        touched = np.zeros(self.body.N, dtype=bool)
        for term, wt in w.items():
            accumulate(self.body, [term], s, touched, K1, B, wt)
            accumulate(self.title, [term], s, touched, K1, B, wt * TITLE_LAMBDA)
        cand = np.flatnonzero(touched)
        if cand.size == 0:
            return []
        v = s[cand]
        if cand.size > K:
            top = np.argpartition(-v, K - 1)[:K]
            cand, v = cand[top], v[top]
        order = np.lexsort((cand, -v))
        return [self.body.doc_ids[int(cand[i])] for i in order]


def main():
    corpus = os.path.join(REPO, "data", "full", "corpus.jsonl")
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]
    qids = [q for q, _ in qs]

    print("loading plain index + title field ...", flush=True)
    body = InvertedIndex.load(os.path.join(REPO, ".index_cache"))
    cfg = body.config
    title = build_field(corpus, cfg, window(cfg, 0, TITLE_W))
    ex = StemExpander(body, title)

    shipped = evaluate(body, [(title, K1, B, TITLE_LAMBDA)], qs, qrels, cfg)
    print(f"\nSHIPPED  {float(np.mean(list(shipped.values()))):.4f}\n")

    def run(gamma, weighting):
        return {qid: ndcg_at_k(ex.rank(qt, gamma, weighting), qrels[qid], K)
                for qid, qt in qs}

    # Sanity: gamma=0 must reproduce the shipped system exactly.
    zero = run(0.0, "flat")
    drift = max(abs(zero[q] - shipped[q]) for q in zero)
    print(f"gamma=0 vs shipped: max per-topic drift {drift:.2e} "
          f"({'OK' if drift < 1e-9 else 'MISMATCH -- expansion path differs'})\n")

    cache, results = {}, {}

    def cached(g, wt):
        if (g, wt) not in cache:
            cache[(g, wt)] = run(g, wt)
        return cache[(g, wt)]

    print("in-sample sweep (delta vs SHIPPED, win/loss/tie)")
    grid = [(g, wt) for wt in ("flat", "idf")
            for g in (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)]
    rows = []
    for g, wt in grid:
        sc = cached(g, wt)
        m = float(np.mean(list(sc.values())))
        st = paired_bootstrap(sc, shipped)
        d = m - float(np.mean(list(shipped.values())))
        w = sum(1 for q in sc if sc[q] > shipped[q])
        l = sum(1 for q in sc if sc[q] < shipped[q])
        rows.append(((g, wt), m, d, st["p_value"], w, l))
        print(f"  gamma={g:<5} {wt:<5} {m:.4f}  {d:+.4f}  p={st['p_value']:.3f}  "
              f"{w}/{l}/{len(sc)-w-l}")
        results[f"{wt}:{g}"] = {"ndcg": m, "delta": d, "p": st["p_value"],
                               "w": w, "l": l}

    rows.sort(key=lambda r: -r[1])
    print(f"\nbest in-sample: gamma={rows[0][0][0]} {rows[0][0][1]} "
          f"({rows[0][1]:.4f}) -- selection not yet charged")

    # Honest: choose gamma and weighting inside each fold.
    held, picks = {}, []
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in test]
        best, bm = None, -1.0
        for g, wt in grid:
            sc = cached(g, wt)
            m = float(np.mean([sc[q] for q in train]))
            if m > bm:
                best, bm = (g, wt), m
        picks.append(best)
        sc = cached(*best)
        for q in test:
            held[q] = sc[q]

    hm = float(np.mean(list(held.values())))
    st = paired_bootstrap(held, shipped)
    d = hm - float(np.mean(list(shipped.values())))
    w = sum(1 for q in held if held[q] > shipped[q])
    l = sum(1 for q in held if held[q] < shipped[q])
    print(f"\nHONEST nested CV  {hm:.4f}   {d:+.4f} vs shipped   "
          f"p={st['p_value']:.3f}   {w}/{l}/{len(held)-w-l}")
    print(f"fold picks: {picks}")

    results["honest"] = {"ndcg": hm, "delta": d, "p": st["p_value"],
                        "w": w, "l": l, "picks": [list(p) for p in picks]}
    with open(os.path.join(REPO, "experiments", "stem_expand.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote experiments/stem_expand.json")


if __name__ == "__main__":
    main()
