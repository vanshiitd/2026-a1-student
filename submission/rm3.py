"""
submission/rm3.py — pseudo-relevance feedback over a stemmed analysis chain,
plus a stemmed pseudo-title field. Selected by retrieve.py's ACTIVE_STRATEGY.

RM3 (Lavrenko & Croft 2001; Abdul-Jaleel et al. 2004):

    p(w|R)  =  sum over top-F feedback docs of  p(w|d) * p(q|d)     (RM1)
    q'      =  alpha * q_original  +  (1 - alpha) * top-m of RM1     (RM3)

p(q|d) is the first-pass score, min-max normalised into a distribution over
the feedback set; p(w|d) is tf/|d|. Both passes score the same way: BM25 over
the stemmed body + title, at this module's own tuned k1/b and title weight
(not bm25.py's -- different analysis chain, never assumed to share params).

fb_docs=10, fb_terms=20, alpha=0.6: majority pick across a 5-fold CV of the
(fb_docs, fb_terms, alpha) grid, not a single argmax.
"""
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from submission import bm25 as _bm25mod
from submission._analysis import analyze
from submission._scorers import robertson_idf
from submission.indexer import InvertedIndex

_BODY: Optional[InvertedIndex] = None
_TITLE: Optional[InvertedIndex] = None
_FORWARD = None  # submission._forward.ForwardIndex, bound to _BODY

K1 = 4.5
B = 0.60
TITLE_WEIGHT = 0.10
FB_DOCS = 10
FB_TERMS = 20
ALPHA = 0.6


def build(body_index: InvertedIndex, title_index: InvertedIndex) -> None:
    """Bind the stemmed body and title indexes this scorer ranks over.

    `body_index.forward` must be set (built with `store_forward=True`) --
    RM3's feedback step reads it directly. Called from retrieve.load_index().
    """
    global _BODY, _TITLE, _FORWARD
    if body_index.forward is None:
        raise RuntimeError(
            "rm3.build() requires body_index.store_forward=True at build time; "
            "the feedback step has no term vectors to read otherwise"
        )
    if title_index.N != body_index.N:
        # _field_score() indexes the title field with _BODY's internal doc
        # ids -- only valid if both come from the same corpus build.
        raise RuntimeError(
            f"body and title indexes disagree on document count "
            f"({body_index.N} vs {title_index.N}); they must come from the "
            f"same corpus build"
        )
    _BODY, _TITLE, _FORWARD = body_index, title_index, body_index.forward
    # Warm the same query-time caches submission/bm25.py uses, for the same
    # reason: load time is not a scored metric, per-query latency is.
    if _bm25mod.HAVE_FAST:
        _bm25mod._expanded(_BODY)
        _bm25mod._length_norm(_BODY, K1, B)
        _bm25mod._expanded(_TITLE)
        _bm25mod._length_norm(_TITLE, K1, B)


def _field_score(terms_weighted: dict, k: int, depth: Optional[int] = None
                 ) -> List[Tuple[str, float]]:
    """Rank by BM25 over stemmed body + stemmed title, for a weighted term
    dict (term -> weight). Used for both the first-pass feedback ranking and
    the final reranking -- the only difference is which term weights and
    which k/depth are passed in."""
    scores = np.zeros(_BODY.N, dtype=np.float64)
    touched = np.zeros(_BODY.N, dtype=np.uint8 if _bm25mod.HAVE_FAST else bool)

    for term, weight in terms_weighted.items():
        if _bm25mod.HAVE_FAST:
            _bm25mod._accumulate(_BODY, [term], scores, touched, K1, B, weight)
            _bm25mod._accumulate(_TITLE, [term], scores, touched, K1, B,
                                 weight * TITLE_WEIGHT)
        else:
            _accumulate_numpy_single(_BODY, term, weight, scores, touched)
            _accumulate_numpy_single(_TITLE, term, weight * TITLE_WEIGHT, scores, touched)

    limit = depth or k
    if _bm25mod.HAVE_FAST:
        candidates, values = _bm25mod._fast.select_top_k(scores, touched, limit)
        if candidates.size == 0:
            return []
        return [(_BODY.doc_ids[int(candidates[i])], float(values[i]))
                for i in range(candidates.size)]

    candidates = np.flatnonzero(touched)
    if candidates.size == 0:
        return []
    values = scores[candidates]
    if candidates.size > limit:
        top = np.argpartition(-values, limit - 1)[:limit]
        candidates, values = candidates[top], values[top]
    order = np.lexsort((candidates, -values))
    return [(_BODY.doc_ids[int(candidates[i])], float(values[i])) for i in order]


def _accumulate_numpy_single(index: InvertedIndex, term: str, weight: float,
                             scores: np.ndarray, touched: np.ndarray) -> None:
    """One term's contribution via pure NumPy (no C extension available)."""
    tid = index.term_id(term)
    if tid < 0:
        return
    doc_ids, tfs = index.postings_by_id(tid)
    if doc_ids.size == 0:
        return
    doc_lens = index.doc_len[doc_ids].astype(np.float64)
    avgdl = index.avg_doc_len or 1.0
    idf = robertson_idf(int(index.df[tid]), index.N)
    norm = K1 * (1.0 - B + B * (doc_lens / avgdl))
    contrib = weight * idf * (tfs.astype(np.float64) * (K1 + 1.0)) / (tfs + norm)
    scores[doc_ids] += contrib
    touched[doc_ids] = True


def score(query: str, k: int, fb_docs: int = FB_DOCS, fb_terms: int = FB_TERMS,
         alpha: float = ALPHA) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, RM3-reranked."""
    if _BODY is None:
        raise RuntimeError("rm3.build(body_index, title_index) must be called first")
    if k <= 0:
        return []

    query_terms = list(dict.fromkeys(analyze(query, _BODY.config)))
    if not query_terms:
        return []
    base_weight = {t: 1.0 / len(query_terms) for t in query_terms}

    feedback_docs = _field_score(base_weight, k=fb_docs, depth=fb_docs)
    if not feedback_docs:
        return []

    # p(q|d): min-max normalise the first-pass scores over the feedback set,
    # so the weakest feedback document gets exactly zero influence.
    raw_scores = np.array([s for _d, s in feedback_docs], dtype=np.float64)
    shifted = raw_scores - raw_scores.min()
    doc_weights = (shifted / shifted.sum() if shifted.sum() > 0
                  else np.full(len(feedback_docs), 1.0 / len(feedback_docs)))

    doc_id_to_internal = {ext_id: i for i, ext_id in enumerate(_BODY.doc_ids)}
    feedback_term_mass: dict = defaultdict(float)
    for (doc_id, _score), doc_weight in zip(feedback_docs, doc_weights):
        internal_id = doc_id_to_internal[doc_id]
        term_ids, tfs = _FORWARD.terms_and_tfs(internal_id)
        doc_len = max(int(_BODY.doc_len[internal_id]), 1)
        for term_id, tf in zip(term_ids, tfs):
            feedback_term_mass[int(term_id)] += doc_weight * (tf / doc_len)

    top_feedback = sorted(feedback_term_mass.items(), key=lambda kv: -kv[1])[:fb_terms]
    feedback_mass_sum = sum(v for _t, v in top_feedback) or 1.0

    expanded_query: dict = defaultdict(float)
    for term, weight in base_weight.items():
        expanded_query[term] += alpha * weight
    for term_id, mass in top_feedback:
        expanded_query[_BODY.terms[term_id]] += (1.0 - alpha) * (mass / feedback_mass_sum)

    return _field_score(dict(expanded_query), k=k)
