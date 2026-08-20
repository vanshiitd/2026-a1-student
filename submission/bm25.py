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

from submission import _traverse
from submission.indexer import InvertedIndex

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
    return _traverse.score_single(_INDEX, query, "bm25", k, k1=k1, b=b)
