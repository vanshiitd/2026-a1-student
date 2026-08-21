# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
submission/_fast.pyx — fused VByte decode + BM25 scoring in C.

Profiling the pure-NumPy query path showed 90% of per-query time going to four
phases that each walk the *same* ~480,000 postings: VByte decode (63%), the
gap cumsum (7%), the document-length gather (3%), and the BM25 arithmetic (17%).
The NumPy decoder alone makes five passes over the buffer and allocates four
intermediate arrays the size of the postings list.

This module collapses all four into a single pass with no intermediate
allocation. Document ids are accumulated in a running register as gaps are
decoded, term frequencies are read from their parallel buffer in lockstep, and
the BM25 contribution is added straight into the caller's score array.

The arithmetic is deliberately written to match submission/_scorers.py's
`bm25_contribution` operation-for-operation and in the same order, so both paths
produce bit-identical float64 results rather than merely similar ones. That is
what tests/test_fast_equivalence.py asserts -- a fast wrong answer is worth less
than a slow right one.

This is an optimisation, not a fallback-free dependency: every caller imports it
behind a try/except and keeps a working pure-Python path.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, int64_t, uint64_t

cnp.import_array()


cdef inline uint64_t _read_vbyte(const uint8_t[::1] buf, Py_ssize_t *pos) noexcept nogil:
    """Decode one VByte value, advancing *pos past it.

    Low 7 bits carry payload; the high bit means "another byte follows" --
    identical to the format written by submission/_codecs.py.
    """
    cdef uint64_t value = 0
    cdef int shift = 0
    cdef uint8_t byte
    while True:
        byte = buf[pos[0]]
        pos[0] += 1
        value |= (<uint64_t>(byte & 0x7F)) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return value



def score_bm25_term(const uint8_t[::1] docid_buf,
                    const uint8_t[::1] tf_buf,
                    Py_ssize_t count,
                    const int64_t[::1] doc_len,
                    double[::1] scores,
                    uint8_t[::1] touched,
                    double idf,
                    double k1,
                    double b,
                    double avgdl):
    """Decode one term's postings and accumulate its BM25 contribution.

        contribution = idf * (tf * (k1 + 1)) / (tf + k1*(1 - b + b*dl/avgdl))

    `docid_buf` holds delta-encoded document ids, `tf_buf` the parallel term
    frequencies, both VByte-packed. `scores` and `touched` are accumulated into
    in place, indexed by internal document id.
    """
    cdef Py_ssize_t dpos = 0, fpos = 0, n
    cdef int64_t docid = 0
    cdef double tf, dl, norm
    cdef double k1_plus_1 = k1 + 1.0
    cdef double one_minus_b = 1.0 - b

    with nogil:
        for n in range(count):
            docid += <int64_t>_read_vbyte(docid_buf, &dpos)
            tf = <double>_read_vbyte(tf_buf, &fpos)
            dl = <double>doc_len[docid]
            # Same operation order as the NumPy path, so rounding matches.
            norm = k1 * (one_minus_b + b * (dl / avgdl))
            scores[docid] += idf * (tf * k1_plus_1) / (tf + norm)
            touched[docid] = 1


def decode_postings(const uint8_t[::1] docid_buf,
                    const uint8_t[::1] tf_buf,
                    Py_ssize_t count):
    """Decode a postings list to (doc_ids, tfs) arrays.

    Kept for the paths that genuinely need materialised arrays (Boolean search,
    the experiment harness). Still a single pass per buffer rather than NumPy's
    five, but it does allocate its outputs, so the fused scorer above is the one
    that matters for query latency.
    """
    cdef cnp.ndarray[int64_t, ndim=1] docs = np.empty(count, dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] tfs = np.empty(count, dtype=np.int64)
    cdef int64_t[::1] dv = docs
    cdef int64_t[::1] fv = tfs
    cdef Py_ssize_t dpos = 0, fpos = 0, n
    cdef int64_t docid = 0

    with nogil:
        for n in range(count):
            docid += <int64_t>_read_vbyte(docid_buf, &dpos)
            dv[n] = docid
            fv[n] = <int64_t>_read_vbyte(tf_buf, &fpos)
    return docs, tfs


def score_bm25_term_packed(const uint8_t[::1] docid_buf,
                           const uint8_t[::1] tf_packed,
                           Py_ssize_t tf_start,
                           Py_ssize_t count,
                           const int64_t[::1] exc_val,
                           const int64_t[::1] doc_len,
                           double[::1] scores,
                           uint8_t[::1] touched,
                           double idf,
                           double k1,
                           double b,
                           double avgdl):
    """As `score_bm25_term`, reading nibble-packed term frequencies.

    A posting's nibble index is its posting index, so `tf_start` (the term's
    first posting) is all the offset information needed -- there is no per-term
    tf offset table. A zero nibble is the escape code meaning "this tf is in the
    exception list"; exceptions are visited in order, so `exc_val` is simply the
    slice of exceptions belonging to this term.
    """
    cdef Py_ssize_t dpos = 0, n, j, exc_i = 0
    cdef int64_t docid = 0
    cdef double tf, dl, norm
    cdef double k1_plus_1 = k1 + 1.0
    cdef double one_minus_b = 1.0 - b
    cdef uint8_t byte, nib

    with nogil:
        for n in range(count):
            docid += <int64_t>_read_vbyte(docid_buf, &dpos)
            j = tf_start + n
            byte = tf_packed[j >> 1]
            nib = (byte >> 4) if (j & 1) else (byte & 0x0F)
            if nib == 0:
                tf = <double>exc_val[exc_i]
                exc_i += 1
            else:
                tf = <double>nib
            dl = <double>doc_len[docid]
            norm = k1 * (one_minus_b + b * (dl / avgdl))
            scores[docid] += idf * (tf * k1_plus_1) / (tf + norm)
            touched[docid] = 1


def select_top_k(const double[::1] scores,
                 const uint8_t[::1] touched,
                 int k):
    """Top-k in one pass, without materialising the candidate set.

    The NumPy route -- flatnonzero(touched), gather scores, argpartition -- makes
    three passes over the candidates and allocates two arrays the size of the
    candidate set. On this collection a typical query touches ~152,000 of
    171,000 documents, so that dominated query time (1.18ms of 1.78ms) purely to
    find ten results.

    This keeps a sorted array of the best k seen so far, ascending by score. For
    k=10 linear insertion beats a heap: the branch is only taken when a document
    beats the current k-th best, which after the first few hundred documents is
    rare, and ten shifts is cheaper than sift-down bookkeeping.

    Ties break on ascending document id, matching the NumPy path: documents are
    scanned in id order and a tie does NOT displace the incumbent, so the
    earlier id survives.
    """
    cdef Py_ssize_t n = scores.shape[0]
    cdef Py_ssize_t i, j
    cdef int filled = 0
    cdef double s

    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    cdef cnp.ndarray[double, ndim=1] best_s = np.empty(k, dtype=np.float64)
    cdef cnp.ndarray[int64_t, ndim=1] best_i = np.empty(k, dtype=np.int64)
    cdef double[::1] bs = best_s
    cdef int64_t[::1] bi = best_i

    with nogil:
        for i in range(n):
            if touched[i] == 0:
                continue
            s = scores[i]
            if filled < k:
                # Insert into the ascending-by-score prefix.
                j = filled
                # >= not > : the array is ascending and gets reversed on return,
                # so a new equal score must be placed BEFORE the incumbent here
                # to end up AFTER it in the result. Documents are scanned in
                # ascending id order, so this is what makes ties resolve to the
                # smaller id, matching np.lexsort((cand, -values)).
                while j > 0 and bs[j - 1] >= s:
                    bs[j] = bs[j - 1]
                    bi[j] = bi[j - 1]
                    j -= 1
                bs[j] = s
                bi[j] = i
                filled += 1
            elif s > bs[0]:
                # Strictly greater, so an equal score never evicts an earlier id.
                j = 0
                while j + 1 < k and bs[j + 1] < s:
                    bs[j] = bs[j + 1]
                    bi[j] = bi[j + 1]
                    j += 1
                bs[j] = s
                bi[j] = i

    # Stored ascending; the caller wants best first.
    return best_i[:filled][::-1].copy(), best_s[:filled][::-1].copy()
