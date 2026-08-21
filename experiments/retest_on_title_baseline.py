#!/usr/bin/env python
"""
experiments/retest_on_title_baseline.py — F28: retest SDM, RM3, cross-chain
fusion and stemming against the title-field baseline rather than the older
body-only one.

The title field (F24) raised the baseline from 0.6281 to 0.6395. Every earlier
rejection (SDM F16, RM3 F19, fusion F15/F21, stemming F14/F26) was measured
against 0.6281. This script re-measures each against 0.6395, in case a
rejection was an artefact of the weaker comparison point rather than a real
absence of signal. All four verdicts hold (see notes/findings.md F28).

Shares scoring logic with experiments/structure_probe.py (`accumulate`,
`evaluate`) so "body+title" means the same thing in both places.

Usage:
    python experiments/retest_on_title_baseline.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import cv_folds, get_index, load_topics, paired_bootstrap
from experiments.structure_probe import accumulate, build_field, evaluate, window
from submission import custom_scorer
from submission._analysis import AnalysisConfig, analyze
from submission._codecs import unpack_tf_nibbles, vbyte_decode
from submission._scorers import robertson_idf
from submission.indexer import InvertedIndex

REPO = os.path.join(os.path.dirname(__file__), "..")
K1, B = 4.5, 0.60
TITLE_W, TITLE_LAMBDA = 10, 0.10


def build_forward_in_memory(index):
    """Transpose postings into doc -> (term_ids, tfs). Used by the RM3 retest
    only; mirrors experiments/rm3_probe.py's approach."""
    n_terms = len(index.terms)
    total = int(index.df.sum())
    gaps = vbyte_decode(index._docid_buf, total)
    tfs = unpack_tf_nibbles(index._tf_packed, 0, total, index._tf_exc_idx, index._tf_exc_val)
    term_of = np.repeat(np.arange(n_terms, dtype=np.int64), index.df)
    starts = index._term_start
    running = np.cumsum(gaps)
    base = np.zeros(n_terms, dtype=np.int64)
    base[1:] = running[starts[1:] - 1]
    doc_of = running - np.repeat(base, index.df)
    order = np.lexsort((term_of, doc_of))
    doc_of, term_of, tfs = doc_of[order], term_of[order], tfs[order]
    counts = np.bincount(doc_of, minlength=index.N)
    offsets = np.zeros(index.N + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return term_of, tfs, offsets


def field_weighted_rank(body, title, terms_weighted, k=10, depth=None):
    """Score a weighted term set over body+title, exactly as retrieve.py does
    for the shipped configuration."""
    s = np.zeros(body.N)
    touched = np.zeros(body.N, dtype=bool)
    for term, w in terms_weighted.items():
        accumulate(body, [term], s, touched, K1, B, w)
        accumulate(title, [term], s, touched, K1, B, w * TITLE_LAMBDA)
    cand = np.flatnonzero(touched)
    if cand.size == 0:
        return []
    limit = depth or k
    v = s[cand]
    if cand.size > limit:
        top = np.argpartition(-v, limit - 1)[:limit]
        cand, v = cand[top], v[top]
    order = np.lexsort((cand, -v))
    return [(body.doc_ids[int(cand[i])], float(v[i])) for i in order]


def main():
    corpus = os.path.join(REPO, "data", "full", "corpus.jsonl")
    plain = InvertedIndex.load(os.path.join(REPO, ".index_cache"))
    cfg = plain.config
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]
    title = build_field(corpus, cfg, window(cfg, 0, TITLE_W))

    shipped = evaluate(plain, [(title, K1, B, TITLE_LAMBDA)], qs, qrels, cfg)
    sm = float(np.mean(list(shipped.values())))
    print(f"SHIPPED (body+title): {sm:.4f}\n")

    # --- 1. SDM combined with the title field ---
    print("1. SDM (proximity) + title field")
    positional_dir = os.path.join(REPO, ".index_cache-positional")
    if not os.path.exists(os.path.join(positional_dir, "meta.json")):
        print("  building positional index (one-off, ~20s) ...", flush=True)
        pos = InvertedIndex(store_positions=True)
        pos.build_from_jsonl(corpus)
        pos.save(positional_dir)
    else:
        pos = InvertedIndex.load(positional_dir)
    assert pos.doc_ids == plain.doc_ids
    custom_scorer.build(pos)

    for frac, lo, lu in [(0.02, 0.10, 0.05), (0.02, 0.25, 0.10),
                         (0.05, 0.10, 0.05), (0.05, 0.25, 0.10)]:
        sc = {}
        for q, t in qs:
            r = custom_scorer.score(t, 1000, k1=K1, b=B, lambda_o=lo, lambda_u=lu,
                                    pair_max_df_frac=frac, candidates=1000)
            rd = dict(r)
            terms = list(dict.fromkeys(analyze(t, cfg)))
            ts = np.zeros(title.N)
            touched = np.zeros(title.N, dtype=bool)
            accumulate(title, terms, ts, touched, K1, B, TITLE_LAMBDA)
            for di in np.flatnonzero(touched):
                doc_id = plain.doc_ids[int(di)]
                rd[doc_id] = rd.get(doc_id, 0.0) + float(ts[di])
            top = sorted(rd.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            sc[q] = ndcg_at_k([d for d, _ in top], qrels[q], k=10)
        m = float(np.mean(list(sc.values())))
        st = paired_bootstrap(sc, shipped)
        print(f"   frac={frac} lo={lo} lu={lu}: {m:.4f}  {m - sm:+.4f}  p={st['p_value']:.4f}")

    # --- 2. RM3 combined with the title field ---
    print("\n2. RM3 (pseudo-relevance feedback) + title field")
    fwd = build_forward_in_memory(plain)

    def rm3_combo(query, fb_docs, fb_terms, alpha):
        base_w = {t: 1.0 for t in dict.fromkeys(analyze(query, cfg))}
        if not base_w:
            return []
        tot = sum(base_w.values())
        base_w = {t: v / tot for t, v in base_w.items()}
        docs_scores = field_weighted_rank(plain, title, base_w, k=fb_docs, depth=fb_docs)
        if not docs_scores:
            return []
        ext_to_int = {d: i for i, d in enumerate(plain.doc_ids)}
        w_doc = np.array([s for _d, s in docs_scores])
        w_doc = w_doc - w_doc.min()
        w_doc = w_doc / w_doc.sum() if w_doc.sum() > 0 else np.ones(len(w_doc)) / len(w_doc)
        term_of, tfs_all, offsets = fwd
        rm = defaultdict(float)
        for (doc_id, _s), wd in zip(docs_scores, w_doc):
            di = ext_to_int[doc_id]
            lo, hi = offsets[di], offsets[di + 1]
            dl = max(int(plain.doc_len[di]), 1)
            for tid, tf in zip(term_of[lo:hi], tfs_all[lo:hi]):
                rm[int(tid)] += wd * (tf / dl)
        top = sorted(rm.items(), key=lambda kv: -kv[1])[:fb_terms]
        mass = sum(v for _t, v in top) or 1.0
        final = defaultdict(float)
        for t, w in base_w.items():
            final[t] += alpha * w
        for tid, v in top:
            final[plain.terms[tid]] += (1 - alpha) * (v / mass)
        return field_weighted_rank(plain, title, dict(final), k=10)

    results = {}
    for fb_docs in (5, 10):
        for fb_terms in (10, 20):
            for alpha in (0.6, 0.8):
                sc = {q: ndcg_at_k([d for d, _ in rm3_combo(t, fb_docs, fb_terms, alpha)],
                                   qrels[q], k=10) for q, t in qs}
                results[(fb_docs, fb_terms, alpha)] = sc
                m = np.mean(list(sc.values()))
                st = paired_bootstrap(sc, shipped)
                print(f"   F={fb_docs} m={fb_terms} alpha={alpha}: {m:.4f}  "
                      f"{m - sm:+.4f}  p={st['p_value']:.4f}")

    qids = sorted(shipped)
    held = {}
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in set(test)]
        pick = max(results, key=lambda k: np.mean([results[k][q] for q in train]))
        for q in test:
            held[q] = results[pick][q]
    st = paired_bootstrap(held, shipped)
    print(f"   HONEST CV: {np.mean(list(held.values())):.4f}  "
          f"delta={st['delta']:+.4f} p={st['p_value']:.4f}")

    # --- 3. Cross-chain fusion + title field, and the stemming reliability check ---
    print("\n3. Porter-stemmed body+title vs plain body+title (does stemming's "
          "reliability improve when stacked with the title field?)")
    pcfg = AnalysisConfig(stemmer="porter")
    pbody = get_index(corpus, config=pcfg)
    ptitle = build_field(corpus, pcfg, window(pcfg, 0, TITLE_W))
    porter_full = evaluate(pbody, [(ptitle, K1, B, TITLE_LAMBDA)], qs, qrels, pcfg)
    st = paired_bootstrap(porter_full, shipped)
    print(f"   plain+title  {sm:.4f}")
    print(f"   porter+title {np.mean(list(porter_full.values())):.4f}  "
          f"delta={st['delta']:+.4f} p={st['p_value']:.4f} "
          f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}")
    print("   (compare to stemming alone, no title field: F26 found 24/19/7, p=0.63 --"
          " same coin-flip signature expected here if the effect is unreliable)")


if __name__ == "__main__":
    main()
