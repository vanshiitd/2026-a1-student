"""
submission/boolean_vsm.py -- boolean AND/OR + tf-idf cosine VSM

    w(t, d)   = tf(t, d) * log( N / df(t) )
    sim(q, d) = (q . d) / (||q|| * ||d||)

doc norms are expensive (need every term not just query terms) so cached
and computed lazily on first vsm_score() call
"""
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

from submission._analysis import analyze
from submission._codecs import unpack_tf_nibbles, vbyte_decode
from submission.indexer import InvertedIndex

_INDEX: Optional[InvertedIndex] = None
_DOC_NORMS: Optional[np.ndarray] = None


def build(index: InvertedIndex) -> None:
    global _INDEX, _DOC_NORMS
    _INDEX = index
    _DOC_NORMS = None


def _require_index() -> InvertedIndex:
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build(index) must be called first")
    return _INDEX


def _idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    return np.log(n_docs / np.maximum(df, 1))


def _decode_all_postings(index: InvertedIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """decode whole postings file at once (not term by term, too slow).
    doc-id gaps restart per term so need to subtract each term's running
    offset after the global cumsum"""
    n_terms = len(index.terms)
    total = int(index.df.sum())
    if total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    gaps = vbyte_decode(index._docid_buf, total)
    tfs = unpack_tf_nibbles(index._tf_packed, 0, total,
                            index._tf_exc_idx, index._tf_exc_val)

    term_of_posting = np.repeat(np.arange(n_terms, dtype=np.int64), index.df)

    starts = np.empty(n_terms, dtype=np.int64)
    starts[0] = 0
    np.cumsum(index.df[:-1], out=starts[1:])

    running = np.cumsum(gaps)
    base = np.zeros(n_terms, dtype=np.int64)
    base[1:] = running[starts[1:] - 1]
    doc_ids = running - np.repeat(base, index.df)

    return term_of_posting, doc_ids, tfs


def _document_norms() -> np.ndarray:
    global _DOC_NORMS
    index = _require_index()
    if _DOC_NORMS is not None:
        return _DOC_NORMS

    term_of_posting, doc_ids, tfs = _decode_all_postings(index)
    if doc_ids.size == 0:
        _DOC_NORMS = np.zeros(index.N, dtype=np.float64)
        return _DOC_NORMS

    weights = tfs.astype(np.float64) * _idf(index.df, index.N)[term_of_posting]
    norms_sq = np.bincount(doc_ids, weights=weights * weights, minlength=index.N)
    _DOC_NORMS = np.sqrt(norms_sq)
    return _DOC_NORMS


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """unranked doc_ids matching the query as AND/OR of its terms"""
    index = _require_index()
    mode = mode.lower()
    if mode not in ("and", "or"):
        raise ValueError(f"mode must be 'and' or 'or', got {mode!r}")

    terms = list(dict.fromkeys(analyze(query, index.config)))
    if not terms:
        return []

    result: Optional[np.ndarray] = None
    for term in terms:
        doc_ids, _tfs = index.postings(term)
        if mode == "and":
            if doc_ids.size == 0:
                return []
            result = doc_ids if result is None else np.intersect1d(result, doc_ids, assume_unique=True)
            if result.size == 0:
                return []
        else:
            result = doc_ids if result is None else np.union1d(result, doc_ids)

    if result is None or result.size == 0:
        return []
    return [index.doc_ids[int(d)] for d in result]


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    index = _require_index()
    if k <= 0:
        return []

    query_tf = Counter(analyze(query, index.config))
    if not query_tf:
        return []

    scores = np.zeros(index.N, dtype=np.float64)
    touched = np.zeros(index.N, dtype=bool)
    query_norm_sq = 0.0

    for term, qtf in query_tf.items():
        tid = index.term_id(term)
        if tid < 0:
            continue
        df = int(index.df[tid])
        idf = float(np.log(index.N / max(df, 1)))
        q_weight = qtf * idf
        query_norm_sq += q_weight * q_weight

        doc_ids, tfs = index.postings_by_id(tid)
        scores[doc_ids] += q_weight * (tfs.astype(np.float64) * idf)
        touched[doc_ids] = True

    candidates = np.flatnonzero(touched)
    if candidates.size == 0 or query_norm_sq <= 0.0:
        return []

    doc_norms = _document_norms()[candidates]
    denom = np.sqrt(query_norm_sq) * doc_norms
    cand_scores = np.divide(
        scores[candidates], denom,
        out=np.zeros(candidates.size, dtype=np.float64),
        where=denom > 0,
    )

    # tie break on doc id so argpartition ties don't get arbitrary order
    order = np.lexsort((candidates, -cand_scores))[:k]
    return [(index.doc_ids[int(candidates[i])], float(cand_scores[i])) for i in order]
