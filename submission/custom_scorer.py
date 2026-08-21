"""
submission/custom_scorer.py — the Sequential Dependence Model (SDM).

Called out in the assignment (Section 4.1) as "where separation in the
leaderboard tends to happen": a combination of signals beyond a single
term-independent scorer.

BM25, VSM and query-likelihood all assume query terms are independent. They
cannot tell "rapid testing" from a document that mentions "rapid" in one
sentence and "testing" three paragraphs later. SDM (Metzler & Croft 2005) adds
that missing evidence as two extra feature classes over *adjacent* query terms:

    score(D,Q) = lambda_T * sum_i      f_T(q_i, D)              unigrams
               + lambda_O * sum_i      f_O(q_i, q_i+1, D)       ordered  (#1)
               + lambda_U * sum_i      f_U(q_i, q_i+1, D)       unordered (#uw8)

Only adjacent pairs are used -- that is what makes it *sequential* dependence
rather than full dependence, which would need all 2^n term subsets.

Every f is BM25-saturated so the three feature classes share a scale and the
lambdas stay interpretable. Ordered and unordered counts come from
submission/_proximity.py and are computed only over the top-N candidates from
the unigram pass, since proximity can rerank documents but cannot rescue a
document that contains no query terms at all.

Wire this in from submission/retrieve.py's retrieve() instead of calling a
single scorer directly.
"""
from typing import List, Optional, Tuple

import numpy as np

from submission import _proximity, _scorers
from submission._scorers import CollectionStats, robertson_idf
from submission._traverse import query_terms
from submission.indexer import InvertedIndex

_INDEX: Optional[InvertedIndex] = None
_STATS: Optional[CollectionStats] = None

# Metzler & Croft's reported defaults. Tuned per collection in practice.
DEFAULT_LAMBDA_O = 0.10
DEFAULT_LAMBDA_U = 0.05
DEFAULT_UW_WIDTH = 8
DEFAULT_CANDIDATES = 1000


def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(), not retrieve.build_index() — the
    harness runs those two in separate processes."""
    global _INDEX, _STATS
    _INDEX = index
    _STATS = CollectionStats(index.N, index.avg_doc_len, index.total_tokens)


def _saturate(counts: np.ndarray, doc_lens: np.ndarray, idf: float,
              k1: float, b: float, avgdl: float) -> np.ndarray:
    """BM25 saturation applied to a proximity count.

    Using the same functional form as the unigram features keeps all three SDM
    components on a comparable scale, which is what makes the lambda weights
    mean anything.
    """
    norm = k1 * (1.0 - b + b * (doc_lens / avgdl))
    return idf * (counts * (k1 + 1.0)) / (counts + norm)


def score(query: str, k: int,
          k1: float = 4.5, b: float = 0.60,
          lambda_o: float = DEFAULT_LAMBDA_O,
          lambda_u: float = DEFAULT_LAMBDA_U,
          uw_width: int = DEFAULT_UW_WIDTH,
          candidates: int = DEFAULT_CANDIDATES,
          pair_max_df_frac: float = 1.0) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs ranked by SDM, best first."""
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

    # Stage 1: proximity, over the top-N unigram candidates only.
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
                # Skip dependencies involving a term common enough to be
                # effectively a stopword. These queries are natural-language
                # questions, so raw adjacency produces pairs like "what is" and
                # "is the"; treating those as evidence adds noise in proportion
                # to the lambda weights. Standard SDM is applied to stopped
                # queries for exactly this reason. Filtering on the MORE common
                # term means a pair survives only if both halves are specific.
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
    """BM25 over the query terms; also returns resolved (term, term_id) pairs
    in query order, which the proximity stage needs to form adjacent pairs."""
    scorer = _scorers.get("bm25")
    tokens = query_terms(index, query)
    if not tokens:
        return np.zeros(index.N), np.zeros(index.N, dtype=bool), None

    # Adjacency must follow the ORIGINAL query order, not the deduplicated
    # (term, count) order, or "rapid testing" could be paired wrongly.
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
