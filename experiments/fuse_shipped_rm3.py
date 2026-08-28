#!/usr/bin/env python
"""
experiments/fuse_shipped_rm3.py — fuse the two actual shipped strategies.

Uses the REAL committed code (submission.bm25.score, submission.rm3.score) at
their exact shipped parameters, not a hand-rolled re-implementation. This is
literally "what if the harness combined both entries."

F13 established what fusion needs to work: the two runs must be DECORRELATED
(so each contributes information the other lacks) AND comparable in QUALITY
(so a naive combiner doesn't just get dragged down by the weaker one).

    bm25 vs bm25plus   overlap@10=0.894  -- too similar, fusion added nothing
    bm25 vs lmd        overlap@10=0.520  -- decorrelated, but lmd was 0.083
                                             weaker, so fusion still lost

Before fusing shipped and RM3, there's a specific reason to expect the SAME
kind of failure as bm25-vs-bm25plus, not the useful case: RM3's query is
`alpha * original_terms + (1-alpha) * expansion_terms` with alpha=0.6. Sixty
percent of RM3's final query weight is still sitting on the exact terms
shipped scores with. RM3 is architecturally close to an EXTENSION of shipped,
not an independent view of the corpus -- the opposite of what F13 found useful.

That is a hypothesis, not a conclusion. This script measures it:

  1. overlap@10 and Kendall tau between the two runs' top-10 lists -- the F13
     diagnostic, run for the first time on this specific pair
  2. RRF fusion (rank-based, matches F13/F15's method)
  3. z-normalised score fusion with an honestly cross-validated weight
     (matches F21's method, the direct answer to F13's "RRF is scale-blind"
     diagnosis)

Both indexes already exist on disk if you've run the pool-coverage or
selective-RM3 scripts; if not this builds them once (~30s total).

    python experiments/fuse_shipped_rm3.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.metrics import ndcg_at_k  # noqa: E402
from experiments.evaluate import cv_folds, load_topics, paired_bootstrap  # noqa: E402

CORPUS = os.path.join(REPO, "data", "full", "corpus.jsonl")
K = 10
DEPTH = 200  # retrieved for fusion; K is what's actually scored/returned


def mean(d):
    return float(np.mean(list(d.values()))) if d else 0.0


def get_deep_runs(qs):
    """Retrieve DEPTH results per query from both real strategies, fresh
    processes not required here since we only need the ranked lists, not a
    timed build -- that's already verified separately (F39)."""
    import submission.retrieve as R

    R.ACTIVE_STRATEGY = "shipped"
    R.build_index(CORPUS, os.path.join(REPO, ".fuse_index-shipped"))
    R.load_index(os.path.join(REPO, ".fuse_index-shipped"))
    shipped = {qid: R.retrieve(qt, DEPTH) for qid, qt in qs}

    for m in [m for m in list(sys.modules) if m.startswith("submission")]:
        del sys.modules[m]
    import submission.retrieve as R2

    R2.ACTIVE_STRATEGY = "rm3_stemmed"
    R2.build_index(CORPUS, os.path.join(REPO, ".fuse_index-rm3"))
    R2.load_index(os.path.join(REPO, ".fuse_index-rm3"))
    rm3 = {qid: R2.retrieve(qt, DEPTH) for qid, qt in qs}

    return shipped, rm3


def overlap_and_tau(a, b, k=10):
    """F13's diagnostic: at@k overlap and rank correlation on the common set."""
    ov, taus = [], []
    for qid in a:
        da = [d for d, _ in a[qid][:k]]
        db = [d for d, _ in b[qid][:k]]
        common = set(da) & set(db)
        ov.append(len(common) / k)
        if len(common) >= 2:
            ra = [da.index(d) for d in common]
            rb = [db.index(d) for d in common]
            t, _ = kendalltau(ra, rb)
            if not np.isnan(t):
                taus.append(t)
    return float(np.mean(ov)), float(np.mean(taus)) if taus else float("nan")


def rrf(runs, weights, k=60):
    """Reciprocal rank fusion, weighted. runs: list of {qid: [(doc,score),...]}."""
    out = {}
    for qid in runs[0]:
        acc = {}
        for run, w in zip(runs, weights):
            for rank, (doc, _s) in enumerate(run[qid]):
                acc[doc] = acc.get(doc, 0.0) + w / (k + rank + 1)
        ranked = sorted(acc.items(), key=lambda kv: -kv[1])[:K]
        out[qid] = [d for d, _ in ranked]
    return out


def znorm_combsum(runs, weights):
    """Per-query z-normalised score fusion, matching F21's method."""
    out = {}
    for qid in runs[0]:
        acc = {}
        for run, w in zip(runs, weights):
            scores = np.array([s for _d, s in run[qid]], dtype=np.float64)
            if scores.size < 2 or scores.std() == 0:
                z = np.zeros_like(scores)
            else:
                z = (scores - scores.mean()) / scores.std()
            for (doc, _s), zi in zip(run[qid], z):
                acc[doc] = acc.get(doc, 0.0) + w * float(zi)
        ranked = sorted(acc.items(), key=lambda kv: -kv[1])[:K]
        out[qid] = [d for d, _ in ranked]
    return out


def score_run(doc_lists, qrels):
    return {qid: ndcg_at_k(docs, qrels[qid], K) for qid, docs in doc_lists.items()}


def main():
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]
    qids = [q for q, _ in qs]

    print("retrieving depth-200 lists from both REAL shipped strategies ...",
          flush=True)
    shipped, rm3 = get_deep_runs(qs)

    shipped10 = {qid: ndcg_at_k([d for d, _ in shipped[qid][:K]], qrels[qid], K)
                for qid in qids}
    rm310 = {qid: ndcg_at_k([d for d, _ in rm3[qid][:K]], qrels[qid], K)
             for qid in qids}
    print(f"\nSHIPPED  nDCG@10  {mean(shipped10):.4f}")
    print(f"RM3      nDCG@10  {mean(rm310):.4f}\n")

    print("F13-style diagnostic: is this pair decorrelated, or too similar?")
    ov, tau = overlap_and_tau(shipped, rm3, k=10)
    print(f"  overlap@10 = {ov:.3f}   Kendall tau@10 = {tau:.3f}")
    print("  (reference: bm25 vs bm25plus 0.894/0.807 -- too similar, fusion failed")
    print("             bm25 vs lmd       0.520/0.504 -- decorrelated, but lmd")
    print("                                              was 0.083 weaker, still failed)\n")

    results = {"shipped": mean(shipped10), "rm3": mean(rm310),
              "overlap_at_10": ov, "kendall_tau_at_10": tau}

    print("RRF fusion (rank-based, F13/F15's method)")
    for w in (0.5, 0.6, 0.7, 0.8, 0.9):
        fused = rrf([shipped, rm3], [1 - w, w])
        sc = score_run(fused, qrels)
        st = paired_bootstrap(sc, rm310)
        d = mean(sc) - mean(rm310)
        print(f"  weight(rm3)={w:.1f}   {mean(sc):.4f}   {d:+.4f} vs RM3 alone   "
              f"p={st['p_value']:.3f}")
        results[f"rrf:{w}"] = {"ndcg": mean(sc), "delta_vs_rm3": d, "p": st["p_value"]}
    print()

    print("z-CombSUM (score-based, F21's method -- can down-weight the weaker run)")
    for w in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        fused = znorm_combsum([shipped, rm3], [1 - w, w])
        sc = score_run(fused, qrels)
        st = paired_bootstrap(sc, rm310)
        d = mean(sc) - mean(rm310)
        print(f"  weight(rm3)={w:.1f}   {mean(sc):.4f}   {d:+.4f} vs RM3 alone   "
              f"p={st['p_value']:.3f}")
        results[f"zcombsum:{w}"] = {"ndcg": mean(sc), "delta_vs_rm3": d, "p": st["p_value"]}

    print("\nHonest nested CV over the z-CombSUM weight (selection cost charged)")
    cache = {w: score_run(znorm_combsum([shipped, rm3], [1 - w, w]), qrels)
            for w in np.arange(0.0, 1.01, 0.1)}
    held, picks = {}, []
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in test]
        best, bm = None, -1.0
        for w, sc in cache.items():
            m = float(np.mean([sc[q] for q in train]))
            if m > bm:
                best, bm = w, m
        picks.append(round(best, 2))
        for q in test:
            held[q] = cache[best][q]

    st_ship = paired_bootstrap(held, shipped10)
    st_rm3 = paired_bootstrap(held, rm310)
    print(f"  honest CV fused    {mean(held):.4f}")
    print(f"  vs SHIPPED         {mean(held)-mean(shipped10):+.4f}   "
          f"p={st_ship['p_value']:.3f}")
    print(f"  vs RM3 alone       {mean(held)-mean(rm310):+.4f}   "
          f"p={st_rm3['p_value']:.3f}")
    print(f"  fold picks (weight on rm3): {picks}")

    results["honest_cv"] = {
        "ndcg": mean(held), "delta_vs_shipped": mean(held) - mean(shipped10),
        "p_vs_shipped": st_ship["p_value"],
        "delta_vs_rm3": mean(held) - mean(rm310), "p_vs_rm3": st_rm3["p_value"],
        "fold_picks": picks,
    }

    with open(os.path.join(REPO, "experiments", "fuse_shipped_rm3.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote experiments/fuse_shipped_rm3.json")


if __name__ == "__main__":
    main()
