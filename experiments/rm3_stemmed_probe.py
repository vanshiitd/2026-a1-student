#!/usr/bin/env python
"""
experiments/rm3_stemmed_probe.py — F30: does RM3 pay off when its feedback
terms are drawn from a stemmed vocabulary?

Motivation (see notes/findings.md F19, F26, F28, and the literature survey
that produced F30): stemming alone is rejected (a coin-flip effect, W/L/T
close to 50/50 at fixed parameters). RM3 alone is rejected (honest CV never
clears +0.01, even combined with the title field). Neither has been tried
*combined with the other*. The mechanism worth testing: PRF quality depends on
the feedback term distribution being clean. Porter conflates morphological
variants ("infection"/"infections"/"infected" -> one bucket), so even if
stemming does not move base BM25 much, it could still make RM3's expansion
terms less fragmented -- a document mentioning "infections" would otherwise
contribute a different feedback term than one mentioning "infection", diluting
the signal RM3 draws on.

External grounding: Anserini's official TREC-COVID RM3 baseline (stemmed,
100 feedback terms, richer multi-field query) gained +0.11 nDCG@10 over its
own BM25 baseline -- a genuinely large effect, though confounded with query
richness we do not have access to under this assignment's single-string
query interface. SLEDGE's own from-scratch RM3 grid (fb_terms up to 20,
unstemmed) did not find a comparable effect. This probe isolates the one
variable neither of those matches: stemming x RM3, at fixed, pre-committed
BM25 parameters, with the title field already in place as the production
baseline.

Decision rule, stated before looking at results: ship only if honest
cross-validated gain over the shipped (plain body + title, 0.6395) baseline
is >= +0.01 AND the paired win/loss ratio is not a coin flip (roughly 60/40 or
better) -- the second condition is here specifically because F26/F28 showed a
positive mean alone is not sufficient evidence on this dataset.

Cost if shipped: a second (stemmed) index plus a forward (doc->terms) index,
which this project has consistently avoided building until a probe justifies
the disk cost. This script builds the forward index in memory only.

Usage:
    python experiments/rm3_stemmed_probe.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import ndcg_at_k
from experiments.evaluate import cv_folds, get_index, load_topics, paired_bootstrap
from experiments.structure_probe import accumulate, build_field, evaluate, window
from submission._analysis import AnalysisConfig, analyze
from submission._codecs import unpack_tf_nibbles, vbyte_decode
from submission.indexer import InvertedIndex

REPO = os.path.join(os.path.dirname(__file__), "..")
K1, B = 4.5, 0.60
TITLE_W, TITLE_LAMBDA = 10, 0.10

# fb_terms widened past the 10-40 range every earlier RM3 probe used (F19,
# F28), since Anserini's headline RM3 result used 100 -- an untested regime
# for us. fb_docs and alpha stay in the range earlier probes already covered,
# since nothing there suggested those were the limiting factor.
FB_DOCS_GRID = (5, 10, 20)
FB_TERMS_GRID = (20, 40, 60, 100)
ALPHA_GRID = (0.6, 0.7, 0.8)


def build_forward_in_memory(index):
    """Transpose postings into doc -> (term_ids, tfs). Experiment-only; the
    persistent structure this would require is exactly what the decision rule
    gates paying for."""
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

    print("loading stemmed index + building stemmed title field + forward index ...", flush=True)
    pcfg = AnalysisConfig(stemmer="porter")
    pbody = get_index(corpus, config=pcfg)
    ptitle = build_field(corpus, pcfg, window(pcfg, 0, TITLE_W))
    fwd = build_forward_in_memory(pbody)
    ext_to_int = {d: i for i, d in enumerate(pbody.doc_ids)}
    print(f"  stemmed vocab: {len(pbody.terms):,} terms (plain: {len(plain.terms):,})\n")

    # Reference points.
    title_plain = build_field(corpus, cfg, window(cfg, 0, TITLE_W))
    shipped = evaluate(plain, [(title_plain, K1, B, TITLE_LAMBDA)], qs, qrels, cfg)
    sm = float(np.mean(list(shipped.values())))
    stemmed_base = evaluate(pbody, [(ptitle, K1, B, TITLE_LAMBDA)], qs, qrels, pcfg)
    stm = float(np.mean(list(stemmed_base.values())))
    print(f"SHIPPED  (plain body + title)   : {sm:.4f}")
    print(f"stemmed body + title, no RM3    : {stm:.4f}  "
          f"(reference point for F26's coin-flip effect)\n")

    def rm3_combo(query, fb_docs, fb_terms, alpha):
        base_w = {t: 1.0 for t in dict.fromkeys(analyze(query, pcfg))}
        if not base_w:
            return []
        tot = sum(base_w.values())
        base_w = {t: v / tot for t, v in base_w.items()}
        docs_scores = field_weighted_rank(pbody, ptitle, base_w, k=fb_docs, depth=fb_docs)
        if not docs_scores:
            return []
        w_doc = np.array([s for _d, s in docs_scores])
        w_doc = w_doc - w_doc.min()
        w_doc = w_doc / w_doc.sum() if w_doc.sum() > 0 else np.ones(len(w_doc)) / len(w_doc)
        term_of, tfs_all, offsets = fwd
        rm = defaultdict(float)
        for (doc_id, _s), wd in zip(docs_scores, w_doc):
            di = ext_to_int[doc_id]
            lo, hi = offsets[di], offsets[di + 1]
            dl = max(int(pbody.doc_len[di]), 1)
            for tid, tf in zip(term_of[lo:hi], tfs_all[lo:hi]):
                rm[int(tid)] += wd * (tf / dl)
        top = sorted(rm.items(), key=lambda kv: -kv[1])[:fb_terms]
        mass = sum(v for _t, v in top) or 1.0
        final = defaultdict(float)
        for t, w in base_w.items():
            final[t] += alpha * w
        for tid, v in top:
            final[pbody.terms[tid]] += (1 - alpha) * (v / mass)
        return field_weighted_rank(pbody, ptitle, dict(final), k=10)

    print(f"{'fb_docs':>8}{'fb_terms':>9}{'alpha':>7}{'nDCG':>9}{'vs shipped':>12}{'p':>8}")
    results = {}
    for fb_docs in FB_DOCS_GRID:
        for fb_terms in FB_TERMS_GRID:
            for alpha in ALPHA_GRID:
                sc = {q: ndcg_at_k([d for d, _ in rm3_combo(t, fb_docs, fb_terms, alpha)],
                                   qrels[q], k=10) for q, t in qs}
                results[(fb_docs, fb_terms, alpha)] = sc
                m = float(np.mean(list(sc.values())))
                st = paired_bootstrap(sc, shipped)
                flag = "  <--" if m > sm else ""
                print(f"{fb_docs:>8}{fb_terms:>9}{alpha:>7.1f}{m:>9.4f}{m - sm:>+12.4f}"
                      f"{st['p_value']:>8.4f}{flag}", flush=True)

    best_key = max(results, key=lambda k: np.mean(list(results[k].values())))
    best = results[best_key]
    st = paired_bootstrap(best, shipped)
    print(f"\nBEST in-sample: {best_key} -> {np.mean(list(best.values())):.4f}  "
          f"delta={st['delta']:+.4f} p={st['p_value']:.4f} "
          f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}")

    # Honest value: select (fb_docs, fb_terms, alpha) per training fold, score
    # on the held-out fold. Same discipline as every other technique tested.
    qids = sorted(shipped)
    held = {}
    picks = []
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in set(test)]
        pick = max(results, key=lambda k: np.mean([results[k][q] for q in train]))
        picks.append(pick)
        for q in test:
            held[q] = results[pick][q]
    st2 = paired_bootstrap(held, shipped)
    honest = float(np.mean(list(held.values())))
    print(f"\nper-fold picks: {picks}")
    print(f"HONEST CV vs SHIPPED: {honest:.4f} vs {sm:.4f}   "
          f"delta={st2['delta']:+.4f}  p={st2['p_value']:.4f}  "
          f"W/L/T={st2['wins']}/{st2['losses']}/{st2['ties']}")

    print(f"\nDECISION RULE (pre-committed): ship iff delta >= +0.01 AND "
          f"win/loss ratio is not a coin flip (roughly 60/40 or better).")
    ratio = st2['wins'] / max(st2['wins'] + st2['losses'], 1)
    verdict = "SHIP" if (st2['delta'] >= 0.01 and ratio >= 0.58) else "REJECT"
    print(f"  delta={st2['delta']:+.4f}  win ratio={ratio:.2f} "
          f"({st2['wins']}W/{st2['losses']}L)  -> {verdict}")


if __name__ == "__main__":
    main()
