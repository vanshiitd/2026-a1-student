#!/usr/bin/env python
"""
experiments/cross_collection.py — do our findings hold outside TREC-COVID?

The recurring problem in this project is that 50 topics gives a standard error
of 0.039, so almost every effect we care about is smaller than the noise. More
topics is the only real fix, and the assignment's collection has 50.

Other collections have more. This script re-tests our load-bearing claims on
two additional BEIR collections and reports each one's own verdict:

    trec-covid    50 topics    171,332 docs   (ours; the control)
    nfcorpus     323 topics      3,633 docs   (medical, natural-language questions)
    scidocs    1,000 topics     25,657 docs   (scientific papers)

WHAT THIS CAN AND CANNOT ESTABLISH
----------------------------------
It does NOT shrink the error bar on our TREC-COVID estimate. A result on
nfcorpus is evidence about nfcorpus. Our held-out topics are still COVID topics
and still 50-ish.

What it tests is whether a finding is a PROPERTY OF THE TECHNIQUE or an
artefact of our particular 50 topics. A technique that helps on 1,373 topics
across three collections of different sizes and domains is one we should expect
to help on unseen COVID topics too. One that helps only here is one we got
lucky with. That distinction is exactly what our p-values cannot settle
internally, and it is the strongest argument available short of the leaderboard.

Every parameter is re-tuned PER COLLECTION before any technique is tested.
Testing our k1=4.5 on a 3,633-document corpus and finding it poor would say
nothing about the technique -- only that a parameter tuned on 171K documents
does not transfer, which is unsurprising and uninteresting. Each collection
gets its own best baseline, and techniques must beat THAT.

Hypotheses under test:
    H1  the pseudo-title field helps            (shipped, +0.0114 p=0.011 here)
    H2  RM3 helps                               (open, +0.0438 p=0.063 here)
    H3  IDF-filtering RM3's feedback HURTS      (F37, -0.0409 p=0.002 here)
    H4  this corpus wants an unusually high k1  (F9/F11, k1=4.5 here)

H3 is the one worth the compute. It is a sharp, counter-intuitive, mechanistic
claim -- that the low-IDF terms crowding RM3's expansion are load-bearing
rather than noise. If it replicates on 1,323 external topics it stops being a
50-topic curiosity.

    python experiments/cross_collection.py
    python experiments/cross_collection.py --only nfcorpus --skip-rm3
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.metrics import ndcg_at_k  # noqa: E402
from harness.trec_io import read_qrels, read_queries  # noqa: E402
from experiments.evaluate import cv_folds, get_index, paired_bootstrap  # noqa: E402
from experiments.rm3_refine import RM3  # noqa: E402
from experiments.rm3_stemmed_probe import build_forward_in_memory  # noqa: E402
from experiments.structure_probe import accumulate, build_field, window  # noqa: E402
from submission._analysis import AnalysisConfig, analyze  # noqa: E402

K = 10
TITLE_W = 10
COLLECTIONS = {
    "trec-covid": "data/full",
    "nfcorpus": "data/ext-nfcorpus",
    "scidocs": "data/ext-scidocs",
}
K1_GRID = (0.6, 0.9, 1.2, 2.0, 3.0, 4.5, 6.0, 8.0)
B_GRID = (0.2, 0.35, 0.5, 0.65, 0.8)


def load(data_dir):
    qs = read_queries(os.path.join(data_dir, "queries_dev.tsv"))
    qrels = read_qrels(os.path.join(data_dir, "qrels_dev.txt"))
    return [(q, t) for q, t in qs if q in qrels and qrels[q]], qrels


def rank(body, fields, terms_w, k1, b, k=K):
    """fields: [(index, weight)] scored with the same k1/b as the body."""
    s = np.zeros(body.N)
    touched = np.zeros(body.N, dtype=bool)
    for term, w in terms_w.items():
        accumulate(body, [term], s, touched, k1, b, w)
        for ix, fw in fields:
            accumulate(ix, [term], s, touched, k1, b, w * fw)
    cand = np.flatnonzero(touched)
    if cand.size == 0:
        return []
    v = s[cand]
    if cand.size > k:
        top = np.argpartition(-v, k - 1)[:k]
        cand, v = cand[top], v[top]
    return [body.doc_ids[int(cand[i])] for i in np.lexsort((cand, -v))]


def score_all(body, fields, qs, qrels, cfg, k1, b):
    out = {}
    for qid, text in qs:
        terms = list(dict.fromkeys(analyze(text, cfg)))
        w = {t: 1.0 for t in terms}
        out[qid] = ndcg_at_k(rank(body, fields, w, k1, b), qrels[qid], K)
    return out


def mean(d):
    return float(np.mean(list(d.values()))) if d else 0.0


def compare(label, a, b, name_b):
    st = paired_bootstrap(a, b)
    d = mean(a) - mean(b)
    w = sum(1 for q in a if a[q] > b[q])
    l = sum(1 for q in a if a[q] < b[q])
    sig = "**" if st["p_value"] < 0.05 else "  "
    print(f"    {label:<30} {mean(a):.4f}  {d:+.4f} vs {name_b}  "
          f"p={st['p_value']:.4f}{sig} {w}/{l}/{len(a)-w-l}")
    return {"ndcg": mean(a), "delta": d, "p": st["p_value"], "w": w, "l": l}


def tune_k1b(body, fields, qs, qrels, cfg):
    """Honest per-collection tuning: pick inside folds, score on held-out."""
    cache = {}
    for k1 in K1_GRID:
        for b in B_GRID:
            cache[(k1, b)] = score_all(body, fields, qs, qrels, cfg, k1, b)
    qids = [q for q, _ in qs]
    held, picks = {}, []
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in test]
        best, bm = None, -1.0
        for kb, sc in cache.items():
            m = float(np.mean([sc[q] for q in train]))
            if m > bm:
                best, bm = kb, m
        picks.append(best)
        for q in test:
            held[q] = cache[best][q]
    in_sample = max(cache, key=lambda kb: mean(cache[kb]))
    return cache, held, picks, in_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-rm3", action="store_true")
    args = ap.parse_args()

    names = args.only or list(COLLECTIONS)
    results = {}

    for name in names:
        data_dir = os.path.join(REPO, COLLECTIONS[name])
        corpus = os.path.join(data_dir, "corpus.jsonl")
        if not os.path.exists(corpus):
            print(f"\n### {name}: SKIPPED (no corpus at {corpus})")
            continue

        qs, qrels = load(data_dir)
        grades = defaultdict(int)
        for v in qrels.values():
            for g in v.values():
                grades[g] += 1
        print(f"\n{'='*78}\n### {name}   {len(qs)} topics   "
              f"grades {dict(sorted(grades.items()))}\n{'='*78}")

        cfg = AnalysisConfig()
        body = get_index(corpus, index_dir=os.path.join(REPO, f".xc-{name}"), config=cfg)
        title = build_field(corpus, cfg, window(cfg, 0, TITLE_W))
        print(f"  {body.N:,} docs, {len(body.terms):,} terms, "
              f"avg len {body.avg_doc_len:.1f}")

        R = {}

        # ---- H4: what k1/b does THIS collection want? -------------------
        print("\n  H4: per-collection k1/b (honest CV)")
        cache_nt, held_nt, picks_nt, best_nt = tune_k1b(body, [], qs, qrels, cfg)
        print(f"    body only     best in-sample k1={best_nt[0]}, b={best_nt[1]} "
              f"-> {mean(cache_nt[best_nt]):.4f};  honest CV {mean(held_nt):.4f}")
        print(f"    fold picks: {picks_nt}")
        R["k1b_body_only"] = {"in_sample_best": list(best_nt),
                             "honest": mean(held_nt),
                             "picks": [list(p) for p in picks_nt]}

        # ---- H1: does the title field help, at each one's own k1/b? -----
        print("\n  H1: pseudo-title field (both sides at their own honest k1/b)")
        cache_t, held_t, picks_t, best_t = tune_k1b(body, [(title, 0.10)], qs, qrels, cfg)
        print(f"    with title    best in-sample k1={best_t[0]}, b={best_t[1]} "
              f"-> {mean(cache_t[best_t]):.4f};  honest CV {mean(held_t):.4f}")
        R["H1"] = compare("title vs no-title (honest)", held_t, held_nt, "no-title")

        # ---- H2/H3: RM3, on the stemmed index --------------------------
        if not args.skip_rm3:
            print("\n  H2/H3: RM3 (stemmed index, own k1/b from the body-only tune)")
            pcfg = AnalysisConfig(stemmer="porter")
            pbody = get_index(corpus, index_dir=os.path.join(REPO, f".xc-{name}"),
                              config=pcfg)
            ptitle = build_field(corpus, pcfg, window(pcfg, 0, TITLE_W))
            fwd = build_forward_in_memory(pbody)
            rm3 = RM3(pbody, ptitle, fwd, {d: i for i, d in enumerate(pbody.doc_ids)})
            k1s, bs = best_nt

            base_stem = {}
            for qid, text in qs:
                w = {t: 1.0 for t in dict.fromkeys(analyze(text, pcfg))}
                base_stem[qid] = ndcg_at_k(
                    rank(pbody, [(ptitle, 0.10)], w, k1s, bs), qrels[qid], K)
            print(f"    stemmed baseline (no RM3)      {mean(base_stem):.4f}")

            def rm3_scores(rule, dfcut=1.0):
                out = {}
                for qid, text in qs:
                    res = rm3.score(text, rule=rule, dfcut=dfcut, k1=k1s, b=bs)
                    out[qid] = ndcg_at_k([d for d, _ in res], qrels[qid], K)
                return out

            raw = rm3_scores("raw")
            R["H2"] = compare("RM3 raw vs no-RM3", raw, base_stem, "no-RM3")
            idf = rm3_scores("idf")
            R["H3_idf"] = compare("RM3 idf-filtered vs raw", idf, raw, "RM3-raw")
            dfc = rm3_scores("dfcut", 0.10)
            R["H3_dfcut"] = compare("RM3 df-cut vs raw", dfc, raw, "RM3-raw")

        results[name] = R

    out = os.path.join(REPO, "experiments", "cross_collection.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
