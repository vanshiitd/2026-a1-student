"""
tests/test_codecs.py — property tests for submission/_codecs.py.

The postings file is written once and read back in a different process, so a
codec bug does not look like a crash: it looks like a quietly wrong ranking.
These tests exist to make that class of bug impossible to ship. Boundary values
(the 2^7 / 2^14 / 2^21 / 2^28 VByte width transitions) are tested explicitly
because that is where an off-by-one in the width table would hide.
"""
import numpy as np
import pytest

from submission._codecs import (
    decode_sorted_ids,
    delta_decode,
    delta_encode,
    encode_sorted_ids,
    vbyte_decode,
    vbyte_encode,
)

# The exact widths where VByte adds a byte.
BOUNDARIES = [0, 1, 126, 127, 128, 129, 16382, 16383, 16384, 16385,
              2097151, 2097152, 268435455, 268435456, 2**32 - 1]


def test_vbyte_round_trip_on_width_boundaries():
    values = np.array(BOUNDARIES, dtype=np.int64)
    decoded = vbyte_decode(vbyte_encode(values), len(values))
    np.testing.assert_array_equal(decoded, values)


@pytest.mark.parametrize("seed", range(8))
def test_vbyte_round_trip_random(seed):
    rng = np.random.default_rng(seed)
    # Mixed magnitudes so every VByte width is exercised in one buffer.
    values = np.concatenate([
        rng.integers(0, 127, size=200),
        rng.integers(0, 16383, size=200),
        rng.integers(0, 2**28, size=200),
        rng.integers(0, 2**32 - 1, size=50),
    ]).astype(np.int64)
    decoded = vbyte_decode(vbyte_encode(values), len(values))
    np.testing.assert_array_equal(decoded, values)


def test_vbyte_widths_are_minimal():
    # 1 byte per value below 128, 2 below 16384 -- if this regresses, the index
    # silently grows and the index-size component (Section 7) suffers.
    assert vbyte_encode(np.array([127] * 10)).size == 10
    assert vbyte_encode(np.array([128] * 10)).size == 20
    assert vbyte_encode(np.array([16383] * 10)).size == 20
    assert vbyte_encode(np.array([16384] * 10)).size == 30


def test_vbyte_empty_round_trip():
    assert vbyte_encode(np.array([], dtype=np.int64)).size == 0
    assert vbyte_decode(np.array([], dtype=np.uint8), 0).size == 0


def test_vbyte_rejects_negative_values():
    with pytest.raises(ValueError):
        vbyte_encode(np.array([1, -2, 3], dtype=np.int64))


def test_vbyte_detects_wrong_expected_count():
    buf = vbyte_encode(np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="expected"):
        vbyte_decode(buf, 4)


def test_vbyte_detects_truncated_buffer():
    buf = vbyte_encode(np.array([300], dtype=np.int64))  # 2 bytes
    with pytest.raises(ValueError, match="truncated"):
        vbyte_decode(buf[:1], 1)


def test_delta_round_trip():
    ids = np.array([0, 1, 5, 5, 100, 10_000, 10_001], dtype=np.int64)
    np.testing.assert_array_equal(delta_decode(delta_encode(ids)), ids)


def test_delta_rejects_unsorted_input():
    with pytest.raises(ValueError):
        delta_encode(np.array([5, 3, 9], dtype=np.int64))


@pytest.mark.parametrize("seed", range(5))
def test_sorted_id_round_trip_random(seed):
    rng = np.random.default_rng(seed)
    ids = np.unique(rng.integers(0, 2_000_000, size=5000)).astype(np.int64)
    np.testing.assert_array_equal(decode_sorted_ids(encode_sorted_ids(ids), ids.size), ids)


def test_gap_encoding_actually_shrinks_dense_postings():
    # The whole justification for delta+VByte: a dense postings list should cost
    # ~1 byte per posting, not 4. If this fails the compression story is broken.
    ids = np.arange(0, 100_000, 3, dtype=np.int64)  # gaps of 3
    encoded = encode_sorted_ids(ids)
    assert encoded.size < ids.size * 1.1, "dense gaps should cost ~1 byte each"
    assert encoded.size < ids.nbytes / 3, "should be far smaller than raw int64"
