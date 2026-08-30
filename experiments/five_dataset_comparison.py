#!/usr/bin/env python
"""
experiments/five_dataset_comparison.py — shipped vs rm3_stemmed across 5
TREC/BEIR collections, against published BM25 as an external anchor.

The 5 chosen for relevance to THIS assignment (TREC-COVID: scientific/
biomedical abstracts, natural-language questions, moderate corpus size):

    trec-covid   the assignment's own corpus -- the anchor
    nfcorpus     medical/nutrition, natural-language queries -- closest domain
    scifact      scientific claim verification over abstracts -- closest
                 structural match (paper abstracts, similar length)
    scidocs      scientific paper retrieval
    fiqa         natural-language questions, moderate size -- the other
                 well-established BEIR NL-question benchmark; a domain
                 stretch (finance, not science) but methodologically apt

Runs the REAL submission.retrieve module (not a re-implementation) for both
ACTIVE_STRATEGY values on each, via the same harness metrics used for
grading. Published BM25 nDCG@10 (Thakur et al. 2021, Table 2) is included
as an external, independently-produced anchor -- verified against the
primary source PDF, not recalled from memory.

    python experiments/five_dataset_comparison.py
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.trec_io import read_queries, read_qrels  # noqa: E402
from harness.metrics import ndcg_at_k, average_precision, reciprocal_rank, precision_at_k  # noqa: E402

DATASETS = {
    "trec-covid": ("data/full", 0.656),
    "nfcorpus":   ("data/ext-nfcorpus", 0.325),
    "scifact":    ("data/ext-scifact", 0.665),
    "scidocs":    ("data/ext-scidocs", 0.158),
    "fiqa":       ("data/ext-fiqa", 0.291),
}


def score_strategy(strategy, corpus, queries, qrels, index_dir):
    for m in [m for m in list(sys.modules) if m.startswith("submission")]:
        del sys.modules[m]
    import submission.retrieve as R
    R.ACTIVE_STRATEGY = strategy

    t0 = time.perf_counter()
    R.build_index(corpus, index_dir)
    build_s = time.perf_counter() - t0
    size = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fs in os.walk(index_dir) for f in fs)

    for m in [m for m in list(sys.modules) if m.startswith("submission")]:
        del sys.modules[m]
    import submission.retrieve as R2
    R2.ACTIVE_STRATEGY = strategy
    t0 = time.perf_counter()
    R2.load_index(index_dir)
    load_s = time.perf_counter() - t0

    lat, ndcgs, maps, mrrs, ps = [], [], [], [], []
    for qid, qt in queries:
        t = time.perf_counter()
        res = R2.retrieve(qt, 10)
        lat.append((time.perf_counter() - t) * 1000)
        docs = [d for d, _ in res]
        ndcgs.append(ndcg_at_k(docs, qrels[qid], 10))
        maps.append(average_precision(docs, qrels[qid]))
        mrrs.append(reciprocal_rank(docs, qrels[qid]))
        ps.append(precision_at_k(docs, qrels[qid], 10))

    return {
        "ndcg10": float(np.mean(ndcgs)), "map10": float(np.mean(maps)),
        "mrr": float(np.mean(mrrs)), "p10": float(np.mean(ps)),
        "build_s": build_s, "load_s": load_s, "size_mb": size / 1e6,
        "latency_ms": float(np.mean(lat)),
    }


def main():
    results = {}
    for name, (data_dir, beir_bm25) in DATASETS.items():
        corpus = os.path.join(REPO, data_dir, "corpus.jsonl")
        if not os.path.exists(corpus):
            print(f"### {name}: SKIPPED, no corpus at {corpus}")
            continue

        queries = read_queries(os.path.join(REPO, data_dir, "queries_dev.tsv"))
        qrels = read_qrels(os.path.join(REPO, data_dir, "qrels_dev.txt"))
        qs = [(q, t) for q, t in queries if q in qrels and qrels[q]]

        print(f"\n{'='*70}\n### {name}  ({len(qs)} topics)\n{'='*70}")
        row = {"n_topics": len(qs), "beir_bm25_ndcg10": beir_bm25}
        for strategy in ("shipped", "rm3_stemmed"):
            idx_dir = os.path.join(REPO, f".fivecmp-{name}-{strategy}")
            st = score_strategy(strategy, corpus, qs, qrels, idx_dir)
            row[strategy] = st
            print(f"  {strategy:<14} nDCG@10={st['ndcg10']:.4f}  MAP@10={st['map10']:.4f}  "
                  f"MRR={st['mrr']:.4f}  P@10={st['p10']:.4f}")
            print(f"  {'':<14} build={st['build_s']:.2f}s  size={st['size_mb']:.2f}MB  "
                  f"latency={st['latency_ms']:.2f}ms")
        print(f"  {'BEIR BM25':<14} nDCG@10={beir_bm25:.4f}  (published, Thakur et al. 2021)")
        results[name] = row

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'dataset':<12}{'shipped':>10}{'rm3_stemmed':>13}{'BEIR BM25':>11}{'rm3 vs beir':>13}")
    for name, row in results.items():
        if "shipped" not in row:
            continue
        s, r, b = row["shipped"]["ndcg10"], row["rm3_stemmed"]["ndcg10"], row["beir_bm25_ndcg10"]
        print(f"{name:<12}{s:>10.4f}{r:>13.4f}{b:>11.4f}{r-b:>+13.4f}")

    with open(os.path.join(REPO, "experiments", "five_dataset_comparison.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote experiments/five_dataset_comparison.json")


if __name__ == "__main__":
    main()
