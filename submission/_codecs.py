"""
submission/_codecs.py — integer compression primitives for the postings file.

Implements delta (gap) encoding and byte-aligned variable-length integers
(VByte / LEB128-style): the standard way to store an inverted index compactly,
and the basis of the index-size leaderboard component (assignment Section 7,
plan.md Section 6).

Why VByte and not "just pickle the arrays": document ids within a postings list
are sorted and dense, so their *gaps* are small integers. A 4-byte int32 spends
4 bytes on a gap of 3. VByte spends 1. On a real collection this is the
difference between a multi-hundred-megabyte index and a small one, and it costs
one cheap decode pass.

Format (one byte at a time, little-endian 7-bit groups):
    - low 7 bits of each byte carry payload
    - high bit (0x80) set => "more bytes follow"
    - so 0..127 is one byte, 128..16383 is two, etc.

Everything here is vectorised with NumPy rather than written as a Python loop.
That is not premature optimisation: indexing touches every posting in the
collection, and `build_index()` wall-clock time is a graded efficiency metric
(assignment Section 7). A per-posting Python loop would dominate build time.

All functions are pure and round-trip exact for non-negative integers; see
tests/test_codecs.py for the property tests that pin that down.
"""
from typing import Sequence

import numpy as np

# 32-bit values need at most ceil(32/7) = 5 VByte bytes.
_MAX_VBYTE_BYTES = 5
_SHIFT_THRESHOLDS = ((7, 2), (14, 3), (21, 4), (28, 5))


def delta_encode(sorted_values: np.ndarray) -> np.ndarray:
    """Gap-encode a sorted, strictly increasing array.

    Returns [v0, v1-v0, v2-v1, ...]. Inverse of `delta_decode`.
    """
    arr = np.ascontiguousarray(sorted_values, dtype=np.int64)
    if arr.size == 0:
        return arr.astype(np.uint64)
    gaps = np.empty(arr.size, dtype=np.int64)
    gaps[0] = arr[0]
    np.subtract(arr[1:], arr[:-1], out=gaps[1:])
    if gaps.size and gaps.min() < 0:
        raise ValueError("delta_encode() requires a non-decreasing input array")
    return gaps.astype(np.uint64)


def delta_decode(gaps: np.ndarray) -> np.ndarray:
    """Reconstruct absolute values from gaps. Inverse of `delta_encode`."""
    arr = np.ascontiguousarray(gaps, dtype=np.int64)
    if arr.size == 0:
        return arr
    return np.cumsum(arr, dtype=np.int64)


def vbyte_widths(values: Sequence[int]) -> np.ndarray:
    """Bytes each value will occupy under `vbyte_encode`, without encoding it.

    Exists so the index can VByte-encode the *entire* collection's postings in
    one vectorised call and still recover per-term byte offsets afterwards
    (group-sum these widths). The alternative -- calling `vbyte_encode` once per
    term -- pays NumPy's per-call overhead ~10^6 times during a build and would
    dominate the graded index-build time.
    """
    v = np.ascontiguousarray(values, dtype=np.uint64)
    widths = np.ones(v.size, dtype=np.int64)
    for shift, count in _SHIFT_THRESHOLDS:
        widths[v >= (np.uint64(1) << np.uint64(shift))] = count
    return widths


def vbyte_encode(values: Sequence[int]) -> np.ndarray:
    """Encode non-negative integers as a VByte uint8 array.

    Vectorised: at most `_MAX_VBYTE_BYTES` passes over the data regardless of
    how many values there are.
    """
    v = np.ascontiguousarray(values, dtype=np.uint64)
    if v.size == 0:
        return np.empty(0, dtype=np.uint8)
    if np.any(v.astype(np.int64) < 0):
        raise ValueError("vbyte_encode() requires non-negative values")

    # How many bytes each value needs.
    nbytes = np.ones(v.size, dtype=np.int64)
    for shift, count in _SHIFT_THRESHOLDS:
        nbytes[v >= (np.uint64(1) << np.uint64(shift))] = count

    ends = np.cumsum(nbytes)
    starts = ends - nbytes
    out = np.zeros(int(ends[-1]), dtype=np.uint8)

    for j in range(_MAX_VBYTE_BYTES):
        active = nbytes > j
        if not active.any():
            break
        payload = ((v[active] >> np.uint64(7 * j)) & np.uint64(0x7F)).astype(np.uint8)
        # Continuation bit on every byte except each value's last.
        is_last = (nbytes[active] - 1) == j
        out[starts[active] + j] = np.where(is_last, payload, payload | 0x80)

    return out


def vbyte_decode(buf: np.ndarray, count: int = -1) -> np.ndarray:
    """Decode a VByte buffer produced by `vbyte_encode`.

    `count` is the expected number of values; pass -1 to infer it. When given,
    it is verified -- a mismatch means the buffer or an offset is wrong, which
    is exactly the kind of corruption that would otherwise show up as a
    mysteriously bad ranking rather than an error.
    """
    b = np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else buf
    b = np.ascontiguousarray(b, dtype=np.uint8)
    if b.size == 0:
        if count > 0:
            raise ValueError(f"empty buffer but {count} values expected")
        return np.empty(0, dtype=np.int64)

    is_last = (b & 0x80) == 0
    # Check truncation first: it is the more fundamental corruption, and it
    # also explains a count mismatch, so reporting it first gives the better
    # diagnostic when an offset into the postings file is wrong.
    if not is_last[-1]:
        raise ValueError("truncated VByte buffer: last byte has continuation bit set")
    n_values = int(is_last.sum())
    if count >= 0 and n_values != count:
        raise ValueError(f"VByte buffer holds {n_values} values, expected {count}")

    # Which value each byte belongs to, and its position within that value.
    value_index = np.cumsum(is_last) - is_last
    starts_a_value = np.empty(b.size, dtype=bool)
    starts_a_value[0] = True
    starts_a_value[1:] = is_last[:-1]
    value_start = np.flatnonzero(starts_a_value)
    position = np.arange(b.size, dtype=np.int64) - value_start[value_index]

    out = np.zeros(n_values, dtype=np.uint64)
    payload = (b & 0x7F).astype(np.uint64)
    for p in range(_MAX_VBYTE_BYTES):
        at_p = position == p
        if not at_p.any():
            continue
        out[value_index[at_p]] |= payload[at_p] << np.uint64(7 * p)

    return out.astype(np.int64)


def encode_sorted_ids(sorted_ids: np.ndarray) -> np.ndarray:
    """delta + VByte, the standard postings-list docid encoding."""
    return vbyte_encode(delta_encode(sorted_ids))


def decode_sorted_ids(buf: np.ndarray, count: int = -1) -> np.ndarray:
    """Inverse of `encode_sorted_ids`."""
    return delta_decode(vbyte_decode(buf, count))
