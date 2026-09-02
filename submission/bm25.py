"""
submission/bm25.py -- Okapi BM25

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )
    IDF(qi)     = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

k1, b tunable. actual traversal logic in _scorers.py, this file is the
entrypoint + query time caches
"""
from typing import List, Optional, Tuple

import numpy as np

from submission import _traverse
from submission._analysis import analyze
from submission._codecs import unpack_tf_nibbles, vbyte_decode
from submission._scorers import robertson_idf
from submission.indexer import InvertedIndex

# fast C path, falls back to numpy if the extension didn't build
try:
    from submission import _fast
    HAVE_FAST = True
except ImportError:  # pragma: no cover - exercised by the fallback test
    _fast = None
    HAVE_FAST = False

_INDEX: Optional[InvertedIndex] = None
_TITLE: Optional[InvertedIndex] = None
_TITLE_WEIGHT: float = 0.0

BM25_DEFAULT_K1 = 4.5
BM25_DEFAULT_B = 0.60

_EXPANDED = {}            # id(index) -> (docids int32, tfs uint16)
_NORM_CACHE = {}          # (k1, b) -> length norm array


def build(index: InvertedIndex, title_index: Optional[InvertedIndex] = None,
          title_weight: float = 0.0) -> None:
    """bind the index, warm caches here since load time isn't scored but
    query latency is"""
    global _INDEX, _EXPANDED, _NORM_CACHE, _TITLE, _TITLE_WEIGHT
    _INDEX = index
    _TITLE = title_index
    _TITLE_WEIGHT = title_weight
    _EXPANDED = {}
    _NORM_CACHE = {}
    if HAVE_FAST and index.N:
        _expanded(index)
        _length_norm(index, BM25_DEFAULT_K1, BM25_DEFAULT_B)
        if title_index is not None and title_index.N:
            _expanded(title_index)
            _length_norm(title_index, BM25_DEFAULT_K1, BM25_DEFAULT_B)


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    if _INDEX is None:
        raise RuntimeError("bm25.build(index) must be called before bm25.score()")
    if HAVE_FAST:
        return _score_fast(_INDEX, query, k, k1, b)
    return _traverse.score_single(_INDEX, query, "bm25", k, k1=k1, b=b)


def _expanded(index):
    """decode postings once into flat arrays for the C kernel to index"""
    cached = _EXPANDED.get(id(index))
    if cached is None:
        total = int(index.df.sum())
        gaps = vbyte_decode(index._docid_buf, total)
        starts = index._term_start
        # int32 not int64, doc ids bounded by index.N anyway
        running = np.cumsum(gaps, dtype=np.int32)
        del gaps
        base = np.zeros(starts.size, dtype=np.int32)
        if starts.size > 1:
            base[1:] = running[starts[1:] - 1]
        docids = running - np.repeat(base, index.df)  # already int32
        del running, base
        tfs = unpack_tf_nibbles(index._tf_packed, 0, total,
                                index._tf_exc_idx, index._tf_exc_val).astype(np.uint16)
        cached = (docids, tfs)
        _EXPANDED[id(index)] = cached
    return cached


def _length_norm(index, k1: float, b: float):
    key = (id(index), k1, b)
    cached = _NORM_CACHE.get(key)
    if cached is None:
        avgdl = index.avg_doc_len or 1.0
        cached = k1 * (1.0 - b + b * (index.doc_len.astype(np.float64) / avgdl))
        _NORM_CACHE[key] = cached
    return cached


def _accumulate(index, terms, scores, touched, k1: float, b: float,
                weight: float) -> bool:
    """add one field's contribution, weight folded into idf"""
    docids_all, tfs_all = _expanded(index)
    norm = _length_norm(index, k1, b)
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
        _fast.score_bm25_expanded(
            docids_all[start:start + count],
            tfs_all[start:start + count],
            norm, scores, touched,
            weight * robertson_idf(count, index.N), k1 + 1.0,
        )
    return hit


def _score_fast(index, query: str, k: int, k1: float, b: float) -> List[Tuple[str, float]]:
    """same as numpy path just faster, tested for bit-identical output"""
    if k <= 0:
        return []
    # keep insertion order not set order, needed for exact float match
    terms = list(dict.fromkeys(analyze(query, index.config)))
    if not terms:
        return []

    scores = np.zeros(index.N, dtype=np.float64)
    touched = np.zeros(index.N, dtype=np.uint8)
    hit = _accumulate(index, terms, scores, touched, k1, b, 1.0)
    if _TITLE is not None and _TITLE_WEIGHT and _TITLE.N == index.N:
        hit = _accumulate(_TITLE, terms, scores, touched, k1, b, _TITLE_WEIGHT) or hit

    if not hit:
        return []

    candidates, values = _fast.select_top_k(scores, touched, k)
    if candidates.size == 0:
        return []
    return [(index.doc_ids[int(candidates[i])], float(values[i]))
            for i in range(candidates.size)]
