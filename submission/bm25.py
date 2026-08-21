"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." See the assignment background (Section 3) for the
Robertson & Walker / Robertson & Zaragoza references this is based on.

BM25 score for a query Q = q1...qn against document D:

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )

A standard IDF variant (Robertson-Sparck Jones, +1-smoothed so it stays
non-negative even for terms occurring in more than half the corpus):

    IDF(qi) = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

where:
    N        = number of documents in the corpus
    df(qi)   = number of documents containing qi
    tf(qi,D) = term frequency of qi in D
    |D|      = length of D in tokens
    avgdl    = average document length across the corpus

k1 (typically 1.2-2.0) controls term-frequency saturation; b (in [0, 1])
controls document-length normalisation strength. Both are exposed as real
parameters -- never captured constants -- because the assignment requires
sweeping them (Section 8, "parameter search procedure for k1, b") and the oral
defense perturbs exactly these.

The arithmetic itself lives in submission/_scorers.py (`bm25_contribution`) so
that one postings traversal can feed several rankers; this module is the
assignment-facing entrypoint for it.
"""
from typing import List, Optional, Tuple

import numpy as np

from submission import _traverse
from submission._analysis import analyze
from submission._scorers import robertson_idf
from submission.indexer import InvertedIndex

# Optional C extension (submission/_fast.pyx), built at image-build time by
# setup.py. It fuses VByte decoding with BM25 scoring into a single pass;
# profiling put ~90% of query time in the phases it replaces.
#
# Imported behind try/except on purpose: if it was not compiled for any reason,
# scoring silently falls back to the pure-NumPy path below and the submission
# still runs correctly, just slower. Speed is never allowed to become a
# correctness dependency.
try:
    from submission import _fast
    HAVE_FAST = True
except ImportError:  # pragma: no cover - exercised by the fallback test
    _fast = None
    HAVE_FAST = False

_INDEX: Optional[InvertedIndex] = None


def build(index: InvertedIndex) -> None:
    """Bind the index BM25 will score against.

    Called from retrieve.load_index(), not retrieve.build_index() -- the harness
    runs those in separate processes. Nothing is precomputed here: BM25 needs
    only df, tf, document lengths and avgdl, all of which the index already
    holds after `InvertedIndex.load()`.
    """
    global _INDEX
    _INDEX = index


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""
    if _INDEX is None:
        raise RuntimeError("bm25.build(index) must be called before bm25.score()")
    if HAVE_FAST:
        return _score_fast(_INDEX, query, k, k1, b)
    return _traverse.score_single(_INDEX, query, "bm25", k, k1=k1, b=b)


def _score_fast(index, query: str, k: int, k1: float, b: float) -> List[Tuple[str, float]]:
    """BM25 via the fused C kernel.

    Produces bit-identical scores to the NumPy path -- the kernel performs the
    same operations in the same order, and the extension is compiled without
    -ffast-math so the compiler may not reassociate them. Verified over the full
    dev set by tests/test_fast_equivalence.py.
    """
    if k <= 0:
        return []
    # Insertion order, NOT set order. Float addition is not associative, so
    # accumulating a document's per-term contributions in a different sequence
    # yields a different float64 result -- a 2-ULP divergence that the
    # equivalence test caught. _traverse uses Counter(...).items(), which is
    # insertion-ordered, so this must match it exactly.
    terms = list(dict.fromkeys(analyze(query, index.config)))
    if not terms:
        return []

    scores = np.zeros(index.N, dtype=np.float64)
    touched = np.zeros(index.N, dtype=np.uint8)
    avgdl = index.avg_doc_len or 1.0
    hit = False

    for term in terms:
        tid = index.term_id(term)
        if tid < 0:
            continue
        count = int(index.df[tid])
        if count == 0:
            continue
        hit = True
        start = int(index._term_start[tid])
        # Exceptions are stored in ascending posting order, so this term's are a
        # contiguous slice; the kernel then consumes them in sequence.
        lo = int(np.searchsorted(index._tf_exc_idx, start))
        hi = int(np.searchsorted(index._tf_exc_idx, start + count))
        _fast.score_bm25_term_packed(
            index._docid_buf[index._docid_off[tid]:index._docid_off[tid + 1]],
            index._tf_packed,
            start,
            count,
            np.ascontiguousarray(index._tf_exc_val[lo:hi]),
            index.doc_len,
            scores,
            touched,
            robertson_idf(count, index.N),
            k1, b, avgdl,
        )
    if not hit:
        return []

    # Single-pass top-k in C. Avoids flatnonzero + gather + argpartition over a
    # candidate set that is typically ~89% of the collection.
    candidates, values = _fast.select_top_k(scores, touched, k)
    if candidates.size == 0:
        return []
    return [(index.doc_ids[int(candidates[i])], float(values[i]))
            for i in range(candidates.size)]
