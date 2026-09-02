"""
submission/_proximity.py -- ordered/unordered window counts for SDM.
fold (doc, pos) into one key = doc_id*_DOC_STRIDE + position so two
searchsorted calls do what would otherwise be a per-doc merge loop
"""
from typing import Dict, Optional, Tuple

import numpy as np

_DOC_STRIDE = 1 << 20  # bigger than the longest doc in the corpus


def _keys_for_term(index, tid: int, candidate_mask: Optional[np.ndarray]) -> np.ndarray:
    doc_ids, tfs, positions, offsets = index.postings_with_positions(tid)
    if doc_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    if candidate_mask is not None:
        keep = candidate_mask[doc_ids]
        if not keep.any():
            return np.empty(0, dtype=np.int64)
        if not keep.all():
            per_position = np.repeat(keep, tfs)
            doc_per_position = np.repeat(doc_ids, tfs)
            return (doc_per_position[per_position] * _DOC_STRIDE
                    + positions[per_position])

    return np.repeat(doc_ids, tfs) * _DOC_STRIDE + positions


def ordered_counts(keys_a: np.ndarray, keys_b: np.ndarray, n_docs: int) -> np.ndarray:
    """count of a immediately followed by b, per doc"""
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
    """count of a,b within `width` positions either order, per doc"""
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
    keys_a = _keys_for_term(index, tid_a, candidate_mask)
    keys_b = _keys_for_term(index, tid_b, candidate_mask)
    return (ordered_counts(keys_a, keys_b, n_docs),
            unordered_counts(keys_a, keys_b, n_docs, width))


def bigram_idf(idf_a: float, idf_b: float) -> float:
    """approx pair idf as the rarer term's idf, exact would need
    intersecting full postings lists which is too slow at query time"""
    return max(idf_a, idf_b)
