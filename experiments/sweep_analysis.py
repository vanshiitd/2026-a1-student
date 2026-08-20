#!/usr/bin/env python
"""
experiments/sweep_analysis.py — the L1 analysis-chain selection.

The analysis chain (stemming, stopwords, alphanumeric splitting) is baked into
the postings, so unlike a scoring parameter it cannot be fused over without
building a second index. It is therefore a genuine *selection*, and the most
confounded one in the project: every scorer comparison downstream is conditional
on it. Hence it is decided once, early, and then frozen.

Each candidate chain gets its own index and its own k1/b search, because the
optimal BM25 parameters shift with the chain -- stemming changes df and document
length, so comparing chains at fixed k1/b would confound the chain with a
now-mistuned scorer.

Also reports pairwise rank agreement between chains. Per finding F13 the project
needs a second run that is both strong and decorrelated; two analysis chains
disagree by *vocabulary* rather than by scoring formula, which is a different
axis of disagreement than swapping BM25 for a language model.

Usage:
    python experiments/sweep_analysis.py
    python experiments/sweep_analysis.py --fine   # dense k1/b grid on each chain
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.evaluate import (
    DEFAULT_DATA,
    QueryPostings,
    config_slug,
    cv_folds,
    get_index,
    load_topics,
    paired_bootstrap,
    smooth_surface,
    sweep,
)
from experiments.fuse import agreement
from experiments.sweep_bm25 import log_rows
from submission import _scorers
from submission._analysis import AnalysisConfig
from submission._scorers import CollectionStats

# The candidate chains. Deliberately a small, interpretable set rather than a
# full cross-product: each row isolates one decision, so a win is attributable.
CANDIDATES: Dict[str, AnalysisConfig] = {
    "plain": AnalysisConfig(),
    "porter": AnalysisConfig(stemmer="porter"),
    "nostop": AnalysisConfig(remove_stopwords=True),
    "porter+nostop": AnalysisConfig(stemmer="porter", remove_stopwords=True),
    "splitan": AnalysisConfig(split_alphanum=True),
    "porter+splitan": AnalysisConfig(stemmer="porter", split_alphanum=True),
    "porter+nostop+splitan": AnalysisConfig(
        stemmer="porter", remove_stopwords=True, split_alphanum=True),
}


def evaluate_chain(name: str, config: AnalysisConfig, corpus_path: str,
                   queries, qrels, fine: bool) -> Dict:
    index = get_index(corpus_path, config=config)
    step_k1, step_b = (0.3, 0.05) if fine else (0.6, 0.1)
    k1_values = np.round(np.arange(0.6, 12.0 + 1e-9, step_k1), 3)
    b_values = np.round(np.arange(0.0, 1.0 + 1e-9, step_b), 3)
    configs = [{"k1": float(a), "b": float(b)} for a in k1_values for b in b_values]

    rows = sweep(index, queries, qrels, "bm25", configs)
    log_rows(rows, f"analysis-{name}")

    axes = ("k1", "b")
    keyed = {tuple(round(r["params"][a], 6) for a in axes): r for r in rows}
    smoothed = smooth_surface(rows, axes)
    chosen = keyed[max(smoothed, key=lambda k: smoothed[k])]

    # Honest value of the whole procedure "tune k1/b on this chain, then use it".
    qids = sorted(rows[0]["per_query"])
    held_out = {}
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in set(test)]
        train_scores = {k: float(np.mean([r["per_query"][q] for q in train]))
                        for k, r in keyed.items()}
        sm = smooth_surface(rows, axes, train_scores)
        pick = keyed[max(sm, key=lambda k: sm[k])]
        for q in test:
            held_out[q] = pick["per_query"][q]

    index_bytes = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _sub, files in os.walk(f".index_cache-{config_slug(config)}") for f in files
    )
    return {
        "name": name, "config": config, "index": index,
        "params": chosen["params"],
        "in_sample": chosen["ndcg@10"],
        "honest": float(np.mean(list(held_out.values()))),
        "per_query": chosen["per_query"],
        "held_out": held_out,
        "vocab": len(index.terms),
        "index_mb": index_bytes / 1e6,
        "avg_doc_len": index.avg_doc_len,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--fine", action="store_true")
    args = ap.parse_args()

    corpus_path = os.path.join(args.data_dir, "corpus.jsonl")
    queries, qrels = load_topics(args.data_dir)

    results: List[Dict] = []
    for name, config in CANDIDATES.items():
        print(f"\n--- {name} ---", flush=True)
        results.append(evaluate_chain(name, config, corpus_path, queries, qrels, args.fine))
        r = results[-1]
        print(f"  vocab={r['vocab']:,}  avgdl={r['avg_doc_len']:.1f}  "
              f"index={r['index_mb']:.1f}MB  best={r['params']}")
        print(f"  in-sample {r['in_sample']:.4f}   honest CV {r['honest']:.4f}")

    results.sort(key=lambda r: -r["honest"])
    print(f"\n{'='*88}\nAnalysis chains ranked by HONEST cross-validated nDCG@10\n{'='*88}")
    print(f"{'chain':<24} {'honest':>8} {'in-sample':>10} {'vocab':>9} {'index MB':>9}  params")
    for r in results:
        print(f"{r['name']:<24} {r['honest']:>8.4f} {r['in_sample']:>10.4f} "
              f"{r['vocab']:>9,} {r['index_mb']:>9.1f}  {r['params']}")

    base = next(r for r in results if r["name"] == "plain")
    print(f"\n{'='*88}\nPaired bootstrap vs 'plain' (held-out scores)\n{'='*88}")
    for r in results:
        if r["name"] == "plain":
            continue
        st = paired_bootstrap(r["held_out"], base["held_out"])
        verdict = "significant" if st["p_value"] < 0.05 else "not significant"
        print(f"  {r['name']:<24} delta={st['delta']:+.4f}  p={st['p_value']:.4f}  "
              f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}  [{verdict}]")

    # Fusion potential: does a second chain disagree enough to be worth fusing?
    best = results[0]
    print(f"\n{'='*88}\nRank agreement with the best chain ('{best['name']}') — fusion potential"
          f"\n{'='*88}")
    print(f"{'chain':<24} {'overlap@10':>11} {'tau@10':>8} {'honest gap':>11}  verdict")
    stats_b = CollectionStats(best["index"].N, best["index"].avg_doc_len,
                              best["index"].total_tokens)
    scorer = _scorers.get("bm25")
    best_run = {qid: QueryPostings(best["index"], t).rank(
        scorer, _scorers.resolve_params("bm25", best["params"]), stats_b, k=100)
        for qid, t in queries}

    for r in results[1:]:
        st = CollectionStats(r["index"].N, r["index"].avg_doc_len, r["index"].total_tokens)
        run = {qid: QueryPostings(r["index"], t).rank(
            scorer, _scorers.resolve_params("bm25", r["params"]), st, k=100)
            for qid, t in queries}
        ag = agreement(best_run, run, 10)
        gap = best["honest"] - r["honest"]
        # Per F13: a useful fusion partner must be both decorrelated and close
        # in quality. A large quality gap means RRF would dilute, not combine.
        ok = ag["overlap"] < 0.75 and gap < 0.02
        print(f"{r['name']:<24} {ag['overlap']:>11.3f} {ag['kendall_tau']:>8.3f} "
              f"{gap:>11.4f}  {'CANDIDATE' if ok else 'no'}")

    out = os.path.join(os.path.dirname(__file__), "analysis_chains.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items()
                    if k not in ("index", "config", "per_query", "held_out")}
                   for r in results], f, indent=2)
    print(f"\nwrote {os.path.relpath(out)}")
    print(f"\nRECOMMENDED CHAIN: {results[0]['name']}  {results[0]['config']}")
    print(f"  with BM25 {results[0]['params']} -> honest {results[0]['honest']:.4f}")


if __name__ == "__main__":
    main()
