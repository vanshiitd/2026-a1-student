"""
submission/_proximity.py — ordered and unordered window counting.

Supplies the term-dependence evidence the Sequential Dependence Model needs
(Metzler & Croft 2005): for each adjacent query-term pair, how often the two
terms occur next to each other (ordered) or near each other (unordered) in a
document.

The whole computation is vectorised through one trick: positions are folded into
a single global key

    key = doc_id * _DOC_STRIDE + position

with `_DOC_STRIDE` larger than any document length in the collection. Because the
stride exceeds every position, keys from different documents can never fall
within a window of each other, so a window query over the flat key array is
automatically confined to within-document matches. That turns "for each shared
document, merge two position lists" -- a Python loop over documents -- into two
`searchsorted` calls over flat arrays.

Counting is restricted to a candidate set (the top-N of the unigram pass) rather
than the whole collection. Proximity is a reranking signal: a document with no
query terms at all cannot be rescued by term dependence, so there is nothing to
gain from scoring beyond the candidates, and a great deal of time to lose.
"""
from typing import Dict, Optional, Tuple

import numpy as np

# Must exceed the longest document in the collection (measured max: 19,411
# tokens). 2^20 leaves ample headroom while keeping keys well inside int64.
_DOC_STRIDE = 1 << 20


def _keys_for_term(index, tid: int, candidate_mask: Optional[np.ndarray]) -> np.ndarray:
    """Global (doc, position) keys for one term, optionally filtered to
    candidate documents."""
    doc_ids, tfs, positions, offsets = index.postings_with_positions(tid)
    if doc_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    if candidate_mask is not None:
        keep = candidate_mask[doc_ids]
        if not keep.any():
            return np.empty(0, dtype=np.int64)
        if not keep.all():
            # Expand the per-document mask to per-position before filtering.
            per_position = np.repeat(keep, tfs)
            doc_per_position = np.repeat(doc_ids, tfs)
            return (doc_per_position[per_position] * _DOC_STRIDE
                    + positions[per_position])

    return np.repeat(doc_ids, tfs) * _DOC_STRIDE + positions


def ordered_counts(keys_a: np.ndarray, keys_b: np.ndarray, n_docs: int) -> np.ndarray:
    """Per-document count of `a` immediately followed by `b` (Indri's #1).

    A match is exactly a key in `a` whose successor position exists in `b`, so
    the whole operation is one set intersection on shifted keys.
    """
    out = np.zeros(n_docs, dtype=np.float64)
    if keys_a.size == 0 or keys_b.size == 0:
        return out
    matches = np.intersect1d(keys_a + 1, keys_b, assume_unique=True)
    if matches.size:
        docs = matches // _DOC_STRIDE
        np.add.at(out, docs, 1.0)
    return out


def unordered_counts(keys_a: np.ndarray, keys_b: np.ndarray, n_docs: int,
                     width: int = 8) -> np.ndarray:
    """Per-document count of `a` and `b` co-occurring within `width` positions,
    in either order (Indri's #uwN, counted as matching pairs).

    Two `searchsorted` calls give, for every occurrence of `a`, how many
    occurrences of `b` lie inside its window. The document stride guarantees no
    window can straddle a document boundary.
    """
    out = np.zeros(n_docs, dtype=np.float64)
    if keys_a.size == 0 or keys_b.size == 0:
        return out
    left = np.searchsorted(keys_b, keys_a - width, side="left")
    right = np.searchsorted(keys_b, keys_a + width, side="right")
    counts = (right - left).astype(np.float64)
    hit = counts > 0
    if hit.any():
        np.add.at(out, keys_a[hit] // _DOC_STRIDE, counts[hit])
    return out


def pair_counts(index, tid_a: int, tid_b: int, n_docs: int,
                candidate_mask: Optional[np.ndarray] = None,
                width: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Ordered and unordered co-occurrence counts for one adjacent term pair."""
    keys_a = _keys_for_term(index, tid_a, candidate_mask)
    keys_b = _keys_for_term(index, tid_b, candidate_mask)
    return (ordered_counts(keys_a, keys_b, n_docs),
            unordered_counts(keys_a, keys_b, n_docs, width))


def bigram_idf(idf_a: float, idf_b: float) -> float:
    """IDF weight for a term pair, approximated as the rarer term's IDF.

    Exact bigram collection statistics would require intersecting two *full*
    postings lists per pair -- for common terms that is millions of positions,
    seconds per query, and quite unaffordable at query time.

    `min(df_a, df_b)` bounds how often the pair can occur, so `max(idf_a, idf_b)`
    is a lower bound on the true pair IDF. Two things make the approximation
    benign: a pair's IDF is *constant across documents*, so it cannot reorder
    documents within a single pair's contribution -- it only sets the relative
    weight between pairs; and the SDM lambda weights are tuned on top of it,
    absorbing any systematic scale error.
    """
    return max(idf_a, idf_b)
