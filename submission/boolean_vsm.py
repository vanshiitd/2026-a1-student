"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces:

1. Boolean retrieval: treat the query as an AND (conjunctive) or OR
   (disjunctive) combination of its terms and return the matching document set
   -- no ranking, just set membership.

2. Vector-space ranking: represent query and documents as TF-IDF weighted
   vectors and rank by cosine similarity. The weighting used here is the
   textbook one from the assignment docstring:

       w(t, d) = tf(t, d) * log( N / df(t) )

   with natural log, applied to both the document and the query side, and

       sim(q, d) = (q . d) / (||q|| * ||d||)

Both read the same InvertedIndex built in indexer.py.

Implementation note on ||d||: the document norm needs every term in a document,
not just the query terms, so it cannot be derived from a query-time traversal.
It is computed in one vectorised pass over the whole postings file and then
cached -- see `_document_norms()`. That pass is deferred until the first
`vsm_score()` call rather than run in `build()`, so a harness run that only uses
BM25 never pays for it (index load time is a measured efficiency metric).
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
    """Bind the index, and drop any cached norms from a previous index.

    Called from retrieve.load_index(), not retrieve.build_index() -- the harness
    runs those in separate processes.
    """
    global _INDEX, _DOC_NORMS
    _INDEX = index
    _DOC_NORMS = None


def _require_index() -> InvertedIndex:
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build(index) must be called first")
    return _INDEX


def _idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """w(t,d)'s IDF factor: log(N / df).

    df >= 1 for every term in the dictionary, so this is finite; a term matching
    every document scores log(1) = 0, which is the intended behaviour.
    """
    return np.log(n_docs / np.maximum(df, 1))


def _decode_all_postings(index: InvertedIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the entire postings file in one vectorised pass.

    Returns (term_id, doc_id, tf) parallel arrays covering every posting.

    Doing this term-by-term would mean ~200K NumPy calls; instead both buffers
    are decoded whole. The only subtlety is that document-id gaps restart at
    each term boundary, so a single global cumsum overshoots -- corrected by
    subtracting each term's running offset, computed vectorised below.
    """
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
    # Value of the running sum just before each term's first posting.
    base = np.zeros(n_terms, dtype=np.int64)
    base[1:] = running[starts[1:] - 1]
    doc_ids = running - np.repeat(base, index.df)

    return term_of_posting, doc_ids, tfs


def _document_norms() -> np.ndarray:
    """||d|| for every document, computed once and cached."""
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
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its
    terms.

    Results are returned in ascending internal document order (i.e. the order
    build_index() saw them), which is arbitrary but deterministic. An unknown
    term makes an AND query empty and contributes nothing to an OR query.
    """
    index = _require_index()
    mode = mode.lower()
    if mode not in ("and", "or"):
        raise ValueError(f"mode must be 'and' or 'or', got {mode!r}")

    terms = list(dict.fromkeys(analyze(query, index.config)))  # dedup, keep order
    if not terms:
        return []

    result: Optional[np.ndarray] = None
    for term in terms:
        doc_ids, _tfs = index.postings(term)
        if mode == "and":
            if doc_ids.size == 0:
                return []  # a missing term empties the conjunction
            result = doc_ids if result is None else np.intersect1d(result, doc_ids, assume_unique=True)
            if result.size == 0:
                return []
        else:
            result = doc_ids if result is None else np.union1d(result, doc_ids)

    if result is None or result.size == 0:
        return []
    return [index.doc_ids[int(d)] for d in result]


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    TF-IDF cosine similarity, highest score first."""
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
            continue  # out-of-vocabulary: contributes to neither vector
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

    if candidates.size > k:
        top = np.argpartition(-cand_scores, k - 1)[:k]
        candidates = candidates[top]
        cand_scores = cand_scores[top]
    # Deterministic tie-break on ascending internal document id.
    order = np.lexsort((candidates, -cand_scores))
    return [(index.doc_ids[int(candidates[i])], float(cand_scores[i])) for i in order]
