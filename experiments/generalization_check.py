#!/usr/bin/env python
"""
experiments/generalization_check.py — a single train/test split of the whole
pipeline, as a stress test of whether the shipped configuration is likely to
transfer to the private held-out topics.

Every parameter in the shipped configuration (k1, b, title width, title
lambda) was selected using ALL 50 dev topics -- there is no slice of dev this
project has not touched, so nothing here can be a true blind test. What this
script does instead: pick one arbitrary half of the 50 topics, rerun the
ENTIRE selection pipeline (k1/b grid + title lambda grid) using only that
half, and score the resulting configuration on the OTHER half, which that
selection never saw. This differs from every k-fold CV already run in the
project in two ways that matter for what it is checking:

  1. It selects BOTH parameter groups jointly in one pass, mirroring the
     actual pipeline structure, rather than validating one parameter at a
     time.
  2. It reports one clean split rather than a 5-fold average, which is closer
     in spirit to "public dev vs private held-out" than to standard CV.

The result answers a narrower, more direct question than the nested-CV work
already done: if the private topics turn out to be as different from "the
other 25 dev topics" as any two arbitrary halves of dev are from each other,
does the shipped recipe still land close to its reported number? It cannot
prove generalisation -- 25 topics is a small, single sample -- but a large gap
here would be a warning the extensive per-parameter CV could not surface,
since that CV never tests the whole recipe as one unit against a naive split.

Usage:
    python experiments/generalization_check.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import get_index, load_topics, paired_bootstrap, smooth_surface
from experiments.structure_probe import build_field, evaluate, window
from submission.indexer import InvertedIndex

REPO = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(os.path.dirname(__file__), "results.jsonl")
SHIPPED_K1, SHIPPED_B = 4.5, 0.60
SHIPPED_W, SHIPPED_LAMBDA = 10, 0.10
LAMBDA_GRID = (0.05, 0.08, 0.10, 0.12, 0.15)


def load_bm25_rows(tag="bm25-k1b-extended"):
    rows, seen = [], set()
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("tag") != tag:
                continue
            key = (round(r["params"]["k1"], 6), round(r["params"]["b"], 6))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def select_k1b(rows, topic_subset):
    """Argmax of the smoothed (k1, b) surface, scored only over `topic_subset`."""
    axes = ("k1", "b")
    scores = {tuple(round(r["params"][a], 6) for a in axes):
              float(np.mean([r["per_query"][q] for q in topic_subset if q in r["per_query"]]))
              for r in rows}
    keyed = {tuple(round(r["params"][a], 6) for a in axes): r for r in rows}
    smoothed = smooth_surface(rows, axes, scores)
    best = max(smoothed, key=lambda k: smoothed[k])
    return keyed[best]["params"]


def main():
    plain = InvertedIndex.load(os.path.join(REPO, ".index_cache"))
    cfg = plain.config
    corpus = os.path.join(REPO, "data", "full", "corpus.jsonl")
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]
    qids = sorted(qrels)

    rng = np.random.default_rng(20260821)
    shuffled = list(rng.permutation(qids))
    half_a, half_b = set(shuffled[:25]), set(shuffled[25:])
    print(f"split: {len(half_a)} / {len(half_b)} topics (seed 20260821)\n")

    bm25_rows = load_bm25_rows()
    title = build_field(corpus, cfg, window(cfg, 0, SHIPPED_W))

    def score_config(k1, b, lam, topic_ids):
        subset = [(q, t) for q, t in qs if q in topic_ids]
        sc = evaluate(plain, [(title, k1, b, lam)] if lam else [], subset, qrels, cfg)
        return sc

    def select_lambda(k1, b, topic_ids):
        subset = [(q, t) for q, t in qs if q in topic_ids]
        best_lam, best_mean = 0.0, -1.0
        for lam in LAMBDA_GRID:
            sc = evaluate(plain, [(title, k1, b, lam)], subset, qrels, cfg)
            m = float(np.mean(list(sc.values())))
            if m > best_mean:
                best_mean, best_lam = m, lam
        return best_lam

    results = {}
    for train_name, train_ids, test_name, test_ids in [
        ("A", half_a, "B", half_b),
        ("B", half_b, "A", half_a),
    ]:
        print(f"=== train on half {train_name} ({len(train_ids)} topics), "
              f"test on half {test_name} ===")
        params = select_k1b(bm25_rows, train_ids)
        k1, b = params["k1"], params["b"]
        lam = select_lambda(k1, b, train_ids)
        print(f"  selected on train: k1={k1:g}, b={b:g}, title_lambda={lam:g}")

        test_own = score_config(k1, b, lam, test_ids)
        test_shipped = score_config(SHIPPED_K1, SHIPPED_B, SHIPPED_LAMBDA, test_ids)
        m_own = float(np.mean(list(test_own.values())))
        m_shipped = float(np.mean(list(test_shipped.values())))
        st = paired_bootstrap(test_own, test_shipped)
        print(f"  on held-out half {test_name}:")
        print(f"    split-selected config -> {m_own:.4f}")
        print(f"    shipped config        -> {m_shipped:.4f}")
        print(f"    delta={st['delta']:+.4f}  p={st['p_value']:.4f}  "
              f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}")
        results[train_name] = {"params": {"k1": k1, "b": b, "title_lambda": lam},
                               "test_half": test_name,
                               "split_selected": m_own, "shipped": m_shipped}
        print()

    # Full-dev shipped number, for reference.
    shipped_full = evaluate(plain, [(title, SHIPPED_K1, SHIPPED_B, SHIPPED_LAMBDA)], qs, qrels, cfg)
    print(f"reference: shipped config on all 50 dev topics = "
          f"{np.mean(list(shipped_full.values())):.4f}")

    print("\nsummary: does a config selected on an arbitrary half of dev transfer")
    print("to the other half about as well as the shipped (all-50) config does?")
    for name, r in results.items():
        gap = r["split_selected"] - r["shipped"]
        verdict = "consistent" if abs(gap) < 0.02 else "DIVERGENT"
        print(f"  train {name} -> test {r['test_half']}: split {r['split_selected']:.4f} "
              f"vs shipped {r['shipped']:.4f}  (gap {gap:+.4f})  [{verdict}]")

    out = os.path.join(os.path.dirname(__file__), "generalization_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
