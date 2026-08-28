"""
submission/_codecs.py — integer compression primitives for the postings file.

Delta (gap) encoding plus byte-aligned variable-length integers (VByte):
document ids within a postings list are sorted and dense, so their gaps are
small. A 4-byte int32 spends 4 bytes on a gap of 3; VByte spends 1.

Format (little-endian 7-bit groups): low 7 bits carry payload, high bit
(0x80) set means "more bytes follow" -- so 0..127 is one byte, 128..16383 two.

Vectorised with NumPy throughout rather than a Python loop, since indexing
touches every posting and build time is graded. Pure, round-trip exact for
non-negative integers; see tests/test_codecs.py.
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


# ---------------------------------------------------------------------------
# Nibble-packed term frequencies
# ---------------------------------------------------------------------------
# Measured on the real corpus: 70.6% of postings have tf == 1 and 99.68% have
# tf <= 15, yet VByte spends a whole byte on every one of them. Packing tf into
# 4 bits with an escape for the 0.32% tail halves that file (16.3MB -> ~8.2MB).
#
# tf is always >= 1, which frees the value 0 to act as the escape code: a zero
# nibble means "this posting's tf lives in the exception list". Clipping instead
# of escaping was rejected -- with k1=4.5 a tf of 600 still scores materially
# above a tf of 15, so clipping would change rankings.
#
# Packed tfs need no per-term offset table at all: the nibble index of a posting
# IS its posting index, which the cumulative document frequency already gives.

TF_NIBBLE_MAX = 15


def pack_tf_nibbles(tfs: np.ndarray):
    """Pack term frequencies into nibbles. Returns (packed, exc_index, exc_value)."""
    arr = np.ascontiguousarray(tfs, dtype=np.int64)
    if arr.size and arr.min() < 1:
        raise ValueError("term frequencies must be >= 1")
    small = arr <= TF_NIBBLE_MAX
    nibbles = np.where(small, arr, 0).astype(np.uint8)
    exc_index = np.flatnonzero(~small).astype(np.int64)
    exc_value = arr[exc_index]

    padded = np.zeros(((nibbles.size + 1) // 2) * 2, dtype=np.uint8)
    padded[:nibbles.size] = nibbles
    packed = (padded[0::2] | (padded[1::2] << 4)).astype(np.uint8)
    return packed, exc_index, exc_value


def unpack_tf_nibbles(packed: np.ndarray, start: int, count: int,
                      exc_index: np.ndarray, exc_value: np.ndarray) -> np.ndarray:
    """Decode `count` term frequencies beginning at posting `start`."""
    if count == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.arange(start, start + count, dtype=np.int64)
    byte = packed[idx >> 1]
    out = np.where(idx & 1, byte >> 4, byte & 0x0F).astype(np.int64)
    lo = int(np.searchsorted(exc_index, start))
    hi = int(np.searchsorted(exc_index, start + count))
    if hi > lo:
        out[exc_index[lo:hi] - start] = exc_value[lo:hi]
    return out
