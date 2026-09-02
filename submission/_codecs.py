"""
submission/_codecs.py -- delta + VByte encoding for postings, plus nibble
packing for tf. all vectorised w/ numpy, no python loops (too slow otherwise)
"""
from typing import Sequence

import numpy as np

# max 5 bytes for a 32-bit value
_MAX_VBYTE_BYTES = 5
_SHIFT_THRESHOLDS = ((7, 2), (14, 3), (21, 4), (28, 5))


def delta_encode(sorted_values: np.ndarray) -> np.ndarray:
    """gap encode a sorted array. [v0, v1-v0, v2-v1, ...]"""
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
    """undo delta_encode via cumsum"""
    arr = np.ascontiguousarray(gaps, dtype=np.int64)
    if arr.size == 0:
        return arr
    return np.cumsum(arr, dtype=np.int64)


def vbyte_widths(values: Sequence[int]) -> np.ndarray:
    """byte width of each value under vbyte, without actually encoding.
    used to get per-term offsets after encoding everything in one shot"""
    v = np.ascontiguousarray(values, dtype=np.uint64)
    widths = np.ones(v.size, dtype=np.int64)
    for shift, count in _SHIFT_THRESHOLDS:
        widths[v >= (np.uint64(1) << np.uint64(shift))] = count
    return widths


def vbyte_encode(values: Sequence[int]) -> np.ndarray:
    """standard vbyte, 7 bits payload + continuation bit"""
    v = np.ascontiguousarray(values, dtype=np.uint64)
    if v.size == 0:
        return np.empty(0, dtype=np.uint8)
    if np.any(v.astype(np.int64) < 0):
        raise ValueError("vbyte_encode() requires non-negative values")

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
        is_last = (nbytes[active] - 1) == j
        out[starts[active] + j] = np.where(is_last, payload, payload | 0x80)

    return out


def vbyte_decode(buf: np.ndarray, count: int = -1) -> np.ndarray:
    """inverse of vbyte_encode. pass count if you know it, gets checked"""
    b = np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else buf
    b = np.ascontiguousarray(b, dtype=np.uint8)
    if b.size == 0:
        if count > 0:
            raise ValueError(f"empty buffer but {count} values expected")
        return np.empty(0, dtype=np.int64)

    is_last = (b & 0x80) == 0
    if not is_last[-1]:
        raise ValueError("truncated VByte buffer: last byte has continuation bit set")
    n_values = int(is_last.sum())
    if count >= 0 and n_values != count:
        raise ValueError(f"VByte buffer holds {n_values} values, expected {count}")

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
    """delta + vbyte together, for postings docids"""
    return vbyte_encode(delta_encode(sorted_ids))


def decode_sorted_ids(buf: np.ndarray, count: int = -1) -> np.ndarray:
    return delta_decode(vbyte_decode(buf, count))


# tf nibble packing -- most tfs are tiny (checked on the corpus, most are 1),
# so 4 bits per tf + an exception list for the rare big ones saves a lot vs vbyte
TF_NIBBLE_MAX = 15


def pack_tf_nibbles(tfs: np.ndarray):
    """returns (packed, exc_index, exc_value)"""
    arr = np.ascontiguousarray(tfs, dtype=np.int64)
    if arr.size and arr.min() < 1:
        raise ValueError("term frequencies must be >= 1")
    small = arr <= TF_NIBBLE_MAX
    nibbles = np.where(small, arr, 0).astype(np.uint8)  # 0 = check exception list
    exc_index = np.flatnonzero(~small).astype(np.int64)
    exc_value = arr[exc_index]

    padded = np.zeros(((nibbles.size + 1) // 2) * 2, dtype=np.uint8)
    padded[:nibbles.size] = nibbles
    packed = (padded[0::2] | (padded[1::2] << 4)).astype(np.uint8)
    return packed, exc_index, exc_value


def unpack_tf_nibbles(packed: np.ndarray, start: int, count: int,
                      exc_index: np.ndarray, exc_value: np.ndarray) -> np.ndarray:
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
