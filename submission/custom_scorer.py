"""
submission/custom_scorer.py -- Sequential Dependence Model (SDM)

    score(D,Q) = lambda_T * sum f_T(qi, D)              unigrams
               + lambda_O * sum f_O(qi, qi+1, D)         ordered
               + lambda_U * sum f_U(qi, qi+1, D)         unordered

only adjacent pairs, not full dependence. everything bm25-saturated so
the 3 feature classes are on the same scale. proximity counts from
_proximity.py, only computed over top-N unigram candidates
"""
from typing import List, Optional, Tuple

import numpy as np

from submission import _proximity, _scorers
from submission._scorers import CollectionStats, robertson_idf
from submission._traverse import query_terms
from submission.indexer import InvertedIndex

_INDEX: Optional[InvertedIndex] = None
_STATS: Optional[CollectionStats] = None

DEFAULT_LAMBDA_O = 0.10
DEFAULT_LAMBDA_U = 0.05
DEFAULT_UW_WIDTH = 8
DEFAULT_CANDIDATES = 1000


def build(index: InvertedIndex) -> None:
    global _INDEX, _STATS
    _INDEX = index
    _STATS = CollectionStats(index.N, index.avg_doc_len, index.total_tokens)


def _saturate(counts: np.ndarray, doc_lens: np.ndarray, idf: float,
              k1: float, b: float, avgdl: float) -> np.ndarray:
    """bm25-style saturation applied to a proximity count"""
    norm = k1 * (1.0 - b + b * (doc_lens / avgdl))
    return idf * (counts * (k1 + 1.0)) / (counts + norm)


def score(query: str, k: int,
          k1: float = 4.5, b: float = 0.60,
          lambda_o: float = DEFAULT_LAMBDA_O,
          lambda_u: float = DEFAULT_LAMBDA_U,
          uw_width: int = DEFAULT_UW_WIDTH,
          candidates: int = DEFAULT_CANDIDATES,
          pair_max_df_frac: float = 1.0) -> List[Tuple[str, float]]:
    if _INDEX is None or _STATS is None:
        raise RuntimeError("custom_scorer.build(index) must be called first")
    index, stats = _INDEX, _STATS
    if k <= 0:
        return []

    unigram_scores, touched, terms = _unigram_pass(index, stats, query, k1, b)
    if terms is None or not touched.any():
        return []

    lambda_t = 1.0 - lambda_o - lambda_u
    scores = lambda_t * unigram_scores

    # proximity stage, only over top-N unigram candidates
    if (lambda_o or lambda_u) and index.store_positions and len(terms) > 1:
        cand = _top_candidates(unigram_scores, touched, candidates)
        if cand.size:
            mask = np.zeros(index.N, dtype=bool)
            mask[cand] = True
            doc_lens = index.doc_len.astype(np.float64)
            avgdl = stats.avg_doc_len or 1.0

            df_ceiling = pair_max_df_frac * stats.N
            for (term_a, tid_a), (term_b, tid_b) in zip(terms, terms[1:]):
                if tid_a < 0 or tid_b < 0 or tid_a == tid_b:
                    continue
                # skip pairs where either term is too common (basically a
                # stopword) -- e.g. "what is" from nl questions, pure noise
                if max(int(index.df[tid_a]), int(index.df[tid_b])) > df_ceiling:
                    continue
                idf = _proximity.bigram_idf(
                    robertson_idf(int(index.df[tid_a]), stats.N),
                    robertson_idf(int(index.df[tid_b]), stats.N),
                )
                ordered, unordered = _proximity.pair_counts(
                    index, tid_a, tid_b, index.N, mask, uw_width)
                if lambda_o:
                    hit = ordered > 0
                    if hit.any():
                        scores[hit] += lambda_o * _saturate(
                            ordered[hit], doc_lens[hit], idf, k1, b, avgdl)
                if lambda_u:
                    hit = unordered > 0
                    if hit.any():
                        scores[hit] += lambda_u * _saturate(
                            unordered[hit], doc_lens[hit], idf, k1, b, avgdl)

    return _top_k(index, scores, touched, k)


def _unigram_pass(index, stats, query: str, k1: float, b: float):
    """returns bm25 scores + ordered (term, tid) pairs in query order
    (needed to form adjacent pairs correctly, not just dedup order)"""
    scorer = _scorers.get("bm25")
    tokens = query_terms(index, query)
    if not tokens:
        return np.zeros(index.N), np.zeros(index.N, dtype=bool), None

    from submission._analysis import analyze
    ordered_terms = [(t, index.term_id(t)) for t in analyze(query, index.config)]

    scores = np.zeros(index.N, dtype=np.float64)
    touched = np.zeros(index.N, dtype=bool)
    for term, qtf in tokens:
        tid = index.term_id(term)
        if tid < 0:
            continue
        doc_ids, tfs = index.postings_by_id(tid)
        if doc_ids.size == 0:
            continue
        doc_lens = index.doc_len[doc_ids].astype(np.float64)
        scores[doc_ids] += scorer.term_contribution(
            tfs, doc_lens, int(index.df[tid]), int(index.cf[tid]), qtf, stats, k1=k1, b=b)
        touched[doc_ids] = True
    return scores, touched, ordered_terms


def _top_candidates(scores: np.ndarray, touched: np.ndarray, n: int) -> np.ndarray:
    cand = np.flatnonzero(touched)
    if cand.size <= n:
        return cand
    part = np.argpartition(-scores[cand], n - 1)[:n]
    return cand[part]


def _top_k(index, scores: np.ndarray, touched: np.ndarray, k: int):
    cand = np.flatnonzero(touched)
    if cand.size == 0:
        return []
    vals = scores[cand]
    if cand.size > k:
        top = np.argpartition(-vals, k - 1)[:k]
        cand, vals = cand[top], vals[top]
    order = np.lexsort((cand, -vals))
    return [(index.doc_ids[int(cand[i])], float(vals[i])) for i in order]
