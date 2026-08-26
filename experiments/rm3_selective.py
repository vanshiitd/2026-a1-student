#!/usr/bin/env python
"""
experiments/rm3_selective.py — two follow-ups rm3_refine.py left open.

rm3_refine.py established, honestly, that filtering RM3's feedback vocabulary
does not work: IDF weighting costs -0.0409 (p=0.002) and a df cutoff -0.0545
(p<0.001) against unfiltered RM3. It left two things untested.

1. k1/b FOR THE EXPANDED QUERY.
   Both strategies inherit k1=4.5 from a sweep run against the user's original
   query. But RM3 does not score the user's query -- it scores a 20-30 term
   weighted blend. k1 controls how fast a term's contribution saturates with
   repetition, and a query carrying many low-weight terms has a different
   saturation profile from one carrying four high-weight ones. The in-sample
   grid liked k1=3.5, but that value was chosen by looking at the answer, so
   here k1/b enter the nested-CV candidate set and pay their own selection
   cost like everything else.

2. SELECTIVE EXPANSION.
   F30's objection to RM3 was never its mean -- it was the shape of the
   distribution: three topics lose 0.2-0.4 nDCG each. An average that is the
   sum of large wins and large losses is a worse bet on unseen topics than a
   smaller average made of consistent small wins, because the losses are
   evidence the mechanism sometimes fires on the wrong queries.

   So: predict which queries expansion will hurt, and skip it there. Three
   gates, each a hypothesis about when pseudo-relevance feedback is unsafe:

     margin  the top-1 score sits far above the rest -> the first pass already
             found something distinctive, and blending in 20 more terms can
             only blur it.
     spread  the feedback scores are tightly bunched -> no document is clearly
             more relevant than another, so the "assume the top 10 are
             relevant" premise is at its weakest.
     nterms  the query is already long -> it carries enough signal, and the
             expansion's share of the final weight is proportionally larger
             for short queries where it matters more.

   A gate that helps must beat plain RM3, not just the shipped baseline.

    python experiments/rm3_selective.py
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.metrics import ndcg_at_k  # noqa: E402
from experiments.evaluate import cv_folds, get_index, load_topics, paired_bootstrap  # noqa: E402
from experiments.rm3_refine import RM3, K, K1, B, TITLE_LAMBDA, TITLE_W, mean, run_all  # noqa: E402
from experiments.rm3_stemmed_probe import build_forward_in_memory  # noqa: E402
from experiments.structure_probe import build_field, evaluate, window  # noqa: E402
from submission._analysis import AnalysisConfig, analyze  # noqa: E402
from submission.indexer import InvertedIndex  # noqa: E402


def make_gate(kind, threshold):
    """Return a gate(rm3, base_weights, k1, b) -> True if RM3 should be applied."""
    def gate(rm3, base, k1, b):
        if kind == "nterms":
            return len(base) <= threshold
        docs = rm3._rank(base, k=10, depth=10, k1=k1, b=b)
        if len(docs) < 2:
            return False
        s = np.array([sc for _d, sc in docs], dtype=np.float64)
        if kind == "margin":
            # top-1 lead over the rest, relative to the spread. A big lead
            # means the first pass is confident; leave it alone.
            rest = s[1:]
            lead = (s[0] - rest.mean()) / (rest.std() + 1e-9)
            return lead <= threshold
        if kind == "spread":
            # coefficient of variation. Low = undifferentiated = risky.
            cv = s.std() / (abs(s.mean()) + 1e-9)
            return cv >= threshold
        raise ValueError(kind)
    return gate


def main():
    corpus = os.path.join(REPO, "data", "full", "corpus.jsonl")
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]
    qids = [q for q, _ in qs]

    print("building stemmed body + title + forward index ...", flush=True)
    pcfg = AnalysisConfig(stemmer="porter")
    pbody = get_index(corpus, config=pcfg)
    ptitle = build_field(corpus, pcfg, window(pcfg, 0, TITLE_W))
    fwd = build_forward_in_memory(pbody)
    rm3 = RM3(pbody, ptitle, fwd, {d: i for i, d in enumerate(pbody.doc_ids)})

    plain = InvertedIndex.load(os.path.join(REPO, ".index_cache"))
    cfg = plain.config
    title_plain = build_field(corpus, cfg, window(cfg, 0, TITLE_W))
    shipped = evaluate(plain, [(title_plain, K1, B, TITLE_LAMBDA)], qs, qrels, cfg)
    plain_rm3 = run_all(rm3, qs, qrels, rule="raw")

    print(f"\nSHIPPED                 {mean(shipped):.4f}")
    print(f"RM3 (raw, as built)     {mean(plain_rm3):.4f}"
          f"   {mean(plain_rm3)-mean(shipped):+.4f} vs shipped\n")

    cache, results = {}, {}

    def cached(**kw):
        key = tuple(sorted((k, str(v)) for k, v in kw.items()))
        if key not in cache:
            cache[key] = run_all(rm3, qs, qrels, **kw)
        return cache[key]

    def show(label, sc, ref, refname):
        st = paired_bootstrap(sc, ref)
        d = mean(sc) - mean(ref)
        w = sum(1 for q in sc if sc[q] > ref[q])
        l = sum(1 for q in sc if sc[q] < ref[q])
        print(f"  {label:<30} {mean(sc):.4f}  {d:+.4f} vs {refname}  "
              f"p={st['p_value']:.3f}  {w}/{l}/{len(sc)-w-l}")
        return {"ndcg": mean(sc), "delta": d, "p": st["p_value"], "w": w, "l": l}

    # -----------------------------------------------------------------
    # 1. k1/b for the expanded query -- in-sample view first.
    # -----------------------------------------------------------------
    print("1. k1/b for the expanded query (in-sample, selection not yet charged)")
    kb = []
    for k1 in (2.0, 2.5, 3.0, 3.5, 4.5, 5.5):
        for b in (0.45, 0.6, 0.75):
            sc = cached(rule="raw", k1=k1, b=b)
            kb.append(((k1, b), mean(sc)))
    kb.sort(key=lambda r: -r[1])
    for (k1, b), m in kb[:5]:
        results[f"kb:{k1}/{b}"] = show(f"k1={k1} b={b}", cached(rule="raw", k1=k1, b=b),
                                       plain_rm3, "RM3")
    print()

    # -----------------------------------------------------------------
    # 2. Selective expansion -- in-sample view.
    # -----------------------------------------------------------------
    print("2. selective expansion (gate must beat plain RM3, not just shipped)")
    gate_specs = []
    for thr in (0.5, 1.0, 1.5, 2.0):
        gate_specs.append(("margin", thr))
    for thr in (0.05, 0.10, 0.20, 0.35):
        gate_specs.append(("spread", thr))
    for thr in (4, 6, 8, 12):
        gate_specs.append(("nterms", thr))

    gate_scores = {}
    for kind, thr in gate_specs:
        sc = run_all(rm3, qs, qrels, rule="raw", gate=make_gate(kind, thr))
        gate_scores[(kind, thr)] = sc
    ranked = sorted(gate_scores.items(), key=lambda kv: -mean(kv[1]))
    for (kind, thr), sc in ranked[:6]:
        results[f"gate:{kind}={thr}"] = show(f"{kind} <= {thr}", sc, plain_rm3, "RM3")
    print()

    # -----------------------------------------------------------------
    # 3. HONEST nested CV over everything above, selection cost charged.
    # -----------------------------------------------------------------
    print("3. nested CV -- every knob above competes, and pays for being chosen")
    candidates = [{"rule": "raw"}]
    for (k1, b), _ in kb[:6]:
        candidates.append({"rule": "raw", "k1": k1, "b": b})
    gate_lookup = {}
    for (kind, thr), _sc in ranked[:6]:
        tag = f"{kind}:{thr}"
        gate_lookup[tag] = make_gate(kind, thr)
        candidates.append({"rule": "raw", "_gate_tag": tag})

    def eval_candidate(c):
        c = dict(c)
        tag = c.pop("_gate_tag", None)
        if tag is not None:
            key = ("gate", tag) + tuple(sorted(c.items()))
            if key not in cache:
                cache[key] = run_all(rm3, qs, qrels, gate=gate_lookup[tag], **c)
            return cache[key]
        return cached(**c)

    held, picks = {}, []
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in test]
        best_c, best_m = None, -1.0
        for c in candidates:
            sc = eval_candidate(c)
            m = float(np.mean([sc[q] for q in train]))
            if m > best_m:
                best_c, best_m = c, m
        picks.append(best_c)
        sc = eval_candidate(best_c)
        for q in test:
            held[q] = sc[q]

    print()
    results["honest_vs_shipped"] = show("HONEST nested CV", held, shipped, "shipped")
    results["honest_vs_rm3"] = show("HONEST nested CV", held, plain_rm3, "plain RM3")
    print(f"\n   fold picks: {picks}")

    results["_baselines"] = {"shipped": mean(shipped), "rm3_raw": mean(plain_rm3),
                             "honest": mean(held)}
    with open(os.path.join(REPO, "experiments", "rm3_selective.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote experiments/rm3_selective.json")


if __name__ == "__main__":
    main()
