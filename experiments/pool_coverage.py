#!/usr/bin/env python
"""
experiments/pool_coverage.py — how much of each strategy's top-10 is judged?

TREC relevance judgements are POOLED: only documents that participating systems
actually surfaced were assessed, and anything outside that pool scores 0 by
convention rather than by verdict. That penalty is not distributed evenly across
retrieval strategies. A strategy that retrieves *different* documents is, by
construction, the one most likely to land outside the pool -- and query
expansion is exactly such a strategy.

So the question this script answers is not "is RM3 better" (rm3_stemmed_probe.py
answers that, with cross-validation and a significance test). It is narrower and
mechanical: **does RM3's measured score understate it more than BM25's does?**

If RM3's top-10 contains materially more unjudged documents than the shipped
run's, then part of the gap between them is an artefact of pool coverage rather
than a difference in retrieval quality, and RM3's dev score is a floor rather
than an estimate.

Emits, for each strategy:
  - the grade-2 / grade-1 / judged-0 / UNJUDGED split of the top-10
  - nDCG@10 as scored (unjudged counted as 0, the trec_eval convention)
  - nDCG@10 under an optimistic bound (unjudged treated as if grade 2)
  - nDCG@10 over the CONDENSED list (unjudged documents removed entirely)

The first two are bounds and bracket the truth without locating it. The third
is the standard estimator for exactly this problem (Sakai's condensed-list
evaluation, also the idea behind bpref): rather than guessing a grade for an
unjudged document, drop it and PROMOTE whatever was below it. That answers
"how good is this ranking, restricted to documents anyone actually assessed?"
-- the fair comparison when two systems have different pool coverage.

Promotion is why this script retrieves DEPTH (=50) documents and condenses down
to 10, rather than condensing the top-10 in place. Condensing in place would
leave a list shorter than 10 and score it against a full-length ideal, which
double-penalises: the system loses the unjudged document AND the empty slot.
That is not the condensed-list measure, it is a third bound, and it would have
quietly made whichever system has worse pool coverage look worse still.

    python experiments/pool_coverage.py
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.trec_io import read_qrels, read_queries, write_run  # noqa: E402

CORPUS = "data/full/corpus.jsonl"
QUERIES = "data/full/queries_dev.tsv"
QRELS = "data/full/qrels_dev.txt"
RUN_DIR = "runs"
K = 10
DEPTH = 50   # retrieved so the condensed list can promote from below rank 10


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_grades, all_grades, k=K):
    """ranked_grades: grades of what we returned, in rank order."""
    gains = [(2 ** g) - 1 for g in ranked_grades[:k]]
    ideal = sorted(all_grades, reverse=True)[:k]
    idcg = dcg([(2 ** g) - 1 for g in ideal])
    return dcg(gains) / idcg if idcg > 0 else 0.0


def score_strategy(strategy, queries, qrels):
    """Build + load + run all queries under one ACTIVE_STRATEGY, fresh."""
    # Re-import cleanly so module-level state from a previous strategy cannot
    # leak: retrieve.py caches the loaded index in module globals.
    for mod in [m for m in list(sys.modules) if m.startswith("submission")]:
        del sys.modules[mod]
    import submission.retrieve as R
    R.ACTIVE_STRATEGY = strategy

    index_dir = f".pool_cov_index-{strategy}"
    t0 = time.perf_counter()
    R.build_index(CORPUS, index_dir)
    build_s = time.perf_counter() - t0

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(index_dir) for f in fs
    ) / 1e6

    # Fresh load, as the harness does in a separate process.
    for mod in [m for m in list(sys.modules) if m.startswith("submission")]:
        del sys.modules[mod]
    import submission.retrieve as R2
    R2.ACTIVE_STRATEGY = strategy
    R2.load_index(index_dir)

    # Two passes: the graded run is exactly what the harness would see (k=10),
    # and latency is measured on THAT, not on the deeper diagnostic pass.
    run, lat = {}, []
    for qid, qtext in queries:
        t = time.perf_counter()
        run[qid] = R2.retrieve(qtext, K)
        lat.append((time.perf_counter() - t) * 1000)
    deep = {qid: R2.retrieve(qtext, DEPTH) for qid, qtext in queries}
    return run, deep, build_s, size_mb, sum(lat) / len(lat)


def analyse(run, deep, qrels):
    g2 = g1 = g0 = unj = 0
    per_topic_unjudged = []
    nd_scored, nd_optimistic, nd_condensed = [], [], []

    for qid, results in run.items():
        judged = qrels.get(qid, {})
        all_grades = list(judged.values())
        graded_scored, graded_optimistic, graded_condensed = [], [], []
        u_here = 0
        for did, _score in results[:K]:
            if did in judged:
                g = judged[did]
                graded_scored.append(g)
                graded_optimistic.append(g)
                if g >= 2: g2 += 1
                elif g == 1: g1 += 1
                else: g0 += 1
            else:
                unj += 1
                u_here += 1
                graded_scored.append(0)        # trec_eval convention
                graded_optimistic.append(2)    # optimistic bound
        per_topic_unjudged.append((u_here, qid))
        nd_scored.append(ndcg_at_k(graded_scored, all_grades))
        nd_optimistic.append(ndcg_at_k(graded_optimistic, all_grades))
        # True condensed list: strike unjudged documents from the DEEP ranking
        # and take the top K of the survivors, so a dropped document is
        # replaced by the next judged one rather than leaving a hole.
        graded_condensed = [judged[d] for d, _ in deep.get(qid, [])
                            if d in judged][:K]
        nd_condensed.append(ndcg_at_k(graded_condensed, all_grades))

    n = len(run)
    return {
        "topics": n,
        "slots": g2 + g1 + g0 + unj,
        "grade2": g2, "grade1": g1, "judged0": g0, "unjudged": unj,
        "unjudged_pct": 100.0 * unj / max(1, g2 + g1 + g0 + unj),
        "ndcg_scored": sum(nd_scored) / n,
        "ndcg_optimistic": sum(nd_optimistic) / n,
        "ndcg_condensed": sum(nd_condensed) / n,
        "per_topic_unjudged": sorted(per_topic_unjudged, reverse=True)[:5],
        "_nd_scored": nd_scored,
    }


def main():
    if not os.path.exists(CORPUS):
        sys.exit(f"missing {CORPUS} — run scripts/download_full_corpus.py first")

    queries = read_queries(QUERIES)
    qrels = read_qrels(QRELS)
    print(f"{len(queries)} topics, {sum(len(v) for v in qrels.values())} judgements, "
          f"{sum(len(v) for v in qrels.values())/len(qrels):.0f} judged docs/topic\n")

    os.makedirs(RUN_DIR, exist_ok=True)
    out = {}
    for strategy in ("shipped", "rm3_stemmed"):
        print(f"=== {strategy} ===")
        run, deep, build_s, size_mb, lat_ms = score_strategy(strategy, queries, qrels)
        path = os.path.join(RUN_DIR, f"full_{strategy}.trec")
        write_run(path, run, run_tag=strategy)
        st = analyse(run, deep, qrels)
        st.update(build_s=build_s, index_mb=size_mb, latency_ms=lat_ms, run_file=path)
        out[strategy] = st
        print(f"  build {build_s:6.2f}s | index {size_mb:6.2f}MB | {lat_ms:6.2f}ms/query")
        print(f"  grade2 {st['grade2']:3d}  grade1 {st['grade1']:3d}  "
              f"judged0 {st['judged0']:3d}  UNJUDGED {st['unjudged']:3d} "
              f"({st['unjudged_pct']:.1f}%)")
        print(f"  nDCG@10 as scored     {st['ndcg_scored']:.4f}")
        print(f"  nDCG@10 optimistic    {st['ndcg_optimistic']:.4f}  "
              f"(gap {st['ndcg_optimistic']-st['ndcg_scored']:+.4f})")
        print(f"  nDCG@10 condensed     {st['ndcg_condensed']:.4f}  "
              f"(unjudged struck, next judged doc promoted)")
        print(f"  wrote {path}\n")

    a, b = out["shipped"], out["rm3_stemmed"]
    print("=== pool-coverage comparison ===")
    print(f"  unjudged slots:  shipped {a['unjudged']:3d}  vs  rm3 {b['unjudged']:3d}   "
          f"(diff {b['unjudged']-a['unjudged']:+d})")
    print(f"  headroom to the optimistic bound:")
    print(f"    shipped {a['ndcg_optimistic']-a['ndcg_scored']:+.4f}   "
          f"rm3 {b['ndcg_optimistic']-b['ndcg_scored']:+.4f}")
    print(f"  measured gap   rm3 - shipped = {b['ndcg_scored']-a['ndcg_scored']:+.4f}"
          f"   (unjudged scored 0 -- the grading convention)")
    print(f"  condensed gap  rm3 - shipped = "
          f"{b['ndcg_condensed']-a['ndcg_condensed']:+.4f}"
          f"   (unjudged removed -- the fair estimator)")
    print(f"  optimistic gap rm3 - shipped = "
          f"{b['ndcg_optimistic']-a['ndcg_optimistic']:+.4f}"
          f"   (unjudged all grade 2 -- worst case for rm3)")
    print()
    if b["unjudged"] > a["unjudged"]:
        print("  -> RM3 surfaces MORE out-of-pool documents, so its measured score is")
        print("     penalised harder by the pooling convention than the shipped run's.")
    elif b["unjudged"] < a["unjudged"]:
        print("  -> RM3 surfaces FEWER out-of-pool documents. Pooling does not explain")
        print("     its advantage; the gain is being measured on judged documents.")
    else:
        print("  -> Identical pool coverage. Pooling is neutral between the two.")

    with open("experiments/pool_coverage.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                   for k, v in out.items()}, f, indent=2)
    print("\nwrote experiments/pool_coverage.json")


if __name__ == "__main__":
    main()
