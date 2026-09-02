# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
submission/_fast.pyx -- fused vbyte decode + bm25 in C, avoids numpy's
multiple passes/allocations per query. matches _scorers.py bit for bit,
checked in tests/test_fast_equivalence.py. pure speedup, has a python
fallback if this doesn't compile
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, uint16_t, int32_t, int64_t, uint64_t

cnp.import_array()


cdef inline uint64_t _read_vbyte(const uint8_t[::1] buf, Py_ssize_t *pos) noexcept nogil:
    """same format as _codecs.py's vbyte"""
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
    """decode one term's postings + accumulate bm25 score in one pass"""
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
            norm = k1 * (one_minus_b + b * (dl / avgdl))
            scores[docid] += idf * (tf * k1_plus_1) / (tf + norm)
            touched[docid] = 1


def decode_postings(const uint8_t[::1] docid_buf,
                    const uint8_t[::1] tf_buf,
                    Py_ssize_t count):
    """decode to (doc_ids, tfs) arrays, for paths that need materialised arrays"""
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
    """same as score_bm25_term but tf is nibble packed. nibble index ==
    posting index so no offset table needed. 0 nibble = read from exc_val"""
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
    """top-k in one pass instead of numpy's flatnonzero+gather+argpartition.
    keeps a sorted best-k-so-far array, linear insertion, fine for small k"""
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
                j = filled
                # >= not > here, matters for tie-break to match np.lexsort
                while j > 0 and bs[j - 1] >= s:
                    bs[j] = bs[j - 1]
                    bi[j] = bi[j - 1]
                    j -= 1
                bs[j] = s
                bi[j] = i
                filled += 1
            elif s > bs[0]:
                j = 0
                while j + 1 < k and bs[j + 1] < s:
                    bs[j] = bs[j + 1]
                    bi[j] = bi[j + 1]
                    j += 1
                bs[j] = s
                bi[j] = i

    return best_i[:filled][::-1].copy(), best_s[:filled][::-1].copy()


def score_bm25_expanded(const int32_t[::1] docids,
                        const uint16_t[::1] tfs,
                        const double[::1] norm,
                        double[::1] scores,
                        uint8_t[::1] touched,
                        double idf,
                        double k1_plus_1):
    """bm25 over already-decoded postings + precomputed length norm, for
    the hot query path after load-time expansion"""
    cdef Py_ssize_t n, count = docids.shape[0]
    cdef int32_t d
    cdef double tf

    with nogil:
        for n in range(count):
            d = docids[n]
            tf = <double>tfs[n]
            scores[d] += idf * (tf * k1_plus_1) / (tf + norm[d])
            touched[d] = 1
