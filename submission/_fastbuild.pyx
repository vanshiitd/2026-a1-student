# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
submission/_fastbuild.pyx — tokenisation and posting emission in C++.

Phase-timing the build put 69% of it in three pure-Python phases: the
per-token tokenisation loop (37%), `Counter` plus ~16.3M list appends (25%),
and converting those lists to NumPy arrays (6%). The remaining 31% -- lexsort
and VByte encoding -- is already NumPy and has little headroom.

The expensive part is not the scanning, it is that every one of ~29M tokens
becomes a Python `str` object purely to be looked up in the vocabulary dict.
This module avoids that entirely: tokens stay as raw bytes and are interned
through a C++ `unordered_map<string, int>`, so no Python object is created per
token. Short tokens fit in std::string's small-string optimisation and do not
allocate at all.

Per-document term frequencies are counted in an O(1) scratch vector indexed by
term id, with a touched-list to reset only the entries actually used -- rather
than building a `Counter` per document.

Exactness
---------
`text.lower()` is still done in Python (it is already C-speed and handles the
full Unicode case mapping), then encoded to UTF-8 and scanned here for
`[a-z0-9]+` runs. UTF-8 continuation bytes are all >= 0x80 and so can never be
mistaken for ASCII alphanumerics, which makes the byte scan produce exactly the
tokens Python's `re` pattern would.

Only the default analysis chain is supported -- no stemming, stopwords or
alphanumeric splitting. `Builder.supports()` reports that, and the caller falls
back to the Python path for any other configuration rather than silently
producing a different index.
"""

import numpy as np
cimport numpy as cnp

from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t

cnp.import_array()


cdef inline bint _is_token_byte(unsigned char c) noexcept nogil:
    return (c >= b'a' and c <= b'z') or (c >= b'0' and c <= b'9')


cdef class Builder:
    """Accumulates (term_id, doc_id, tf) triples across the whole corpus.

    Holding the output in C++ vectors rather than Python lists is a large part
    of the win: the previous path performed three Python-level `append` calls per
    posting, ~48M in total.
    """
    cdef unordered_map[string, int] vocab
    cdef vector[string] term_bytes          # term id -> its bytes, for the dictionary
    cdef vector[int] scratch_tf             # term id -> tf within the current document
    cdef vector[int] touched                # term ids used by the current document
    # Postings held per term rather than in one document-ordered stream. This
    # is what removes the global sort: documents are processed in ascending id
    # order, so each term's doc list is built already ascending, and grouping by
    # term is exactly what a per-term vector is. The previous layout needed a
    # 16.3M-element np.lexsort afterwards to recover both properties -- 3.06s of
    # a 5.78s build.
    cdef vector[vector[int32_t]] post_docs
    cdef vector[vector[int32_t]] post_tfs
    cdef Py_ssize_t max_token_len
    cdef Py_ssize_t min_token_len

    def __cinit__(self, Py_ssize_t min_token_len=1, Py_ssize_t max_token_len=32):
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len

    @staticmethod
    def supports(config) -> bool:
        """True only for the analysis chain this kernel reproduces exactly."""
        d = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        return (d.get("lowercase", True)
                and not d.get("remove_stopwords", False)
                and d.get("stemmer") is None
                and not d.get("split_alphanum", False))

    def add_document(self, bytes lowered_utf8, int doc_id, Py_ssize_t prefix_tokens=-1):
        """Tokenise one document and emit its postings. Returns the token count.

        `prefix_tokens >= 0` stops after that many surviving tokens, which is how
        the pseudo-title field is built without a second pass over the text.
        """
        cdef const unsigned char* buf = <const unsigned char*>lowered_utf8
        cdef Py_ssize_t n = len(lowered_utf8)
        cdef Py_ssize_t i = 0, start
        cdef Py_ssize_t length
        cdef Py_ssize_t n_tokens = 0
        cdef int tid
        cdef string key
        cdef unordered_map[string, int].iterator it
        cdef Py_ssize_t j
        cdef int t

        self.touched.clear()

        while i < n:
            if not _is_token_byte(buf[i]):
                i += 1
                continue
            start = i
            while i < n and _is_token_byte(buf[i]):
                i += 1
            length = i - start
            if length < self.min_token_len or length > self.max_token_len:
                continue
            # Document length counts tokens that SURVIVE the filter -- the
            # Python analyzer appends to its output list only after filtering,
            # and doc_len feeds BM25's length normalisation, so an off-by-any
            # here would silently change every score.
            n_tokens += 1

            key = string(<const char*>(buf + start), length)
            it = self.vocab.find(key)
            if it == self.vocab.end():
                tid = <int>self.term_bytes.size()
                self.vocab[key] = tid
                self.term_bytes.push_back(key)
                self.scratch_tf.push_back(0)
                self.post_docs.push_back(vector[int32_t]())
                self.post_tfs.push_back(vector[int32_t]())
            else:
                tid = deref(it).second

            if self.scratch_tf[tid] == 0:
                self.touched.push_back(tid)
            self.scratch_tf[tid] += 1

            if prefix_tokens >= 0 and n_tokens >= prefix_tokens:
                break

        # Flush this document's postings, resetting only what was touched.
        for j in range(<Py_ssize_t>self.touched.size()):
            t = self.touched[j]
            self.post_docs[t].push_back(<int32_t>doc_id)
            self.post_tfs[t].push_back(<int32_t>self.scratch_tf[t])
            self.scratch_tf[t] = 0
        return n_tokens

    def terms(self):
        """Vocabulary in first-seen order, as Python strings."""
        cdef Py_ssize_t v = <Py_ssize_t>self.term_bytes.size()
        cdef Py_ssize_t i
        out = [None] * v
        for i in range(v):
            out[i] = self.term_bytes[i].decode("utf-8")
        return out

    def finish_sorted(self, const int32_t[::1] order):
        """Concatenate postings in sorted-term order. Returns (docs, tfs, df).

        `order[i]` is the original term id of the i-th alphabetically-sorted
        term. Walking terms in that order and copying each one's vector produces
        exactly the layout the encoder wants -- grouped by term, ascending by
        doc id within a term -- in a single pass, with no sort anywhere.
        """
        cdef Py_ssize_t n_terms = order.shape[0]
        cdef Py_ssize_t i, j, t, total = 0, pos = 0

        for i in range(n_terms):
            total += <Py_ssize_t>self.post_docs[order[i]].size()

        docs_arr = np.empty(total, dtype=np.int32)
        tfs_arr = np.empty(total, dtype=np.int32)
        df_arr = np.empty(n_terms, dtype=np.int64)
        cdef int32_t[::1] dv = docs_arr
        cdef int32_t[::1] fv = tfs_arr
        cdef long long[::1] df = df_arr
        cdef Py_ssize_t m

        with nogil:
            for i in range(n_terms):
                t = order[i]
                m = <Py_ssize_t>self.post_docs[t].size()
                df[i] = m
                for j in range(m):
                    dv[pos] = self.post_docs[t][j]
                    fv[pos] = self.post_tfs[t][j]
                    pos += 1
        return docs_arr, tfs_arr, df_arr
