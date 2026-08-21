"""
submission/indexer.py — the inverted index (assignment Section 4.1).

Built from scratch: no Lucene/Elasticsearch/Pyserini/Whoosh. Only NumPy (for
array manipulation) and the standard library.

Representation
--------------
Columnar, not a dict-of-dicts. The obvious `Dict[str, Dict[str, int]]` shape
costs a Python object per posting; at 16.3M postings (measured, see
notes/corpus_profile.md) that is several GB of interpreter overhead alone and
would not fit the 8GB grading machine. Instead every per-term and per-posting
quantity lives in a flat NumPy array:

    terms[t]                 term string, sorted ascending
    df[t], cf[t]             document frequency, collection frequency
    docid_off[t:t+2]         byte slice of this term's docids in `_docid_buf`
    term_start[t]            index of this term's first posting; also its
                             nibble offset into the packed tf array
    doc_len[d]               document length in tokens
    doc_ids[d]               external doc_id string for internal id d

Postings are delta+VByte encoded (see submission/_codecs.py). Both the encode
and decode paths are vectorised, and the *whole collection* is encoded in a
single `vbyte_encode` call rather than once per term -- per-term calls would pay
NumPy's fixed overhead ~200K times and dominate the graded build time.

Persistence
-----------
`save()`/`load()` round-trip through plain files with no pickling, so an index
written by one process is readable by a fresh one with no shared state (which is
exactly what the harness verifies). The on-disk size is a graded leaderboard
component (assignment Section 7), so nothing is persisted that `retrieve()` does
not need -- in particular the raw document text is deliberately NOT stored.
"""
import json
import os
import zlib
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from submission._analysis import AnalysisConfig, make_analyzer
from submission._codecs import (pack_tf_nibbles, unpack_tf_nibbles, vbyte_decode,
                                vbyte_encode, vbyte_widths)

# Optional C++ kernel for tokenising and posting emission (see
# submission/_fastbuild.pyx). Imported behind try/except: without it the build
# falls back to the pure-Python path and produces the identical index, just
# more slowly.
try:
    from submission import _fastbuild as _FASTBUILD
except ImportError:  # pragma: no cover - exercised by the fallback test
    _FASTBUILD = None

FORMAT_VERSION = 3   # 3: index files zlib-compressed on disk

# Every index file is deflated before it hits disk. The graded metric is the
# on-disk byte size (assignment Section 7), and the interface contract
# explicitly suggests compressing what is persisted.
#
# Level 4 is chosen from a measured curve, not by default. Compression runs
# inside save(), hence inside build_index(), so it is charged against the
# index-build-time metric; decompression runs in load(), whose time
# harness/leaderboard.py does NOT score. Measured on the real index:
#     level 1 -> 22.20 MB, 0.40s compress
#     level 4 -> 21.57 MB, 0.60s compress
#     level 9 -> 21.22 MB, 7.27s compress   (1MB more for 7s -- rejected)
# Decompression is ~0.05s at any level.
_ZLIB_LEVEL = 4

# Postings accumulated in Python lists before being flushed to NumPy arrays.
# Bounds peak interpreter memory during the build without making the flush
# itself frequent enough to matter.
_FLUSH_EVERY = 2_000_000

_META = "meta.json"
_TERMS = "terms.txt"
_DOCIDS = "docids.txt"
_DF = "df.bin"
_CF = "cf.bin"
_DOCLEN = "doclen.bin"
_DOCID_LEN = "docid_len.bin"
_TF_NIB = "tf_nib.bin"
_TF_EXC_I = "tf_exc_idx.bin"
_TF_EXC_V = "tf_exc_val.bin"
_POSTINGS_D = "postings_d.bin"
_POS_LEN = "pos_len.bin"
_POSITIONS = "positions.bin"


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-only tokenization.

    Kept as a module-level function with the shipped name and behaviour. The
    configurable chain used by the index itself lives in submission/_analysis.py;
    this delegates to it with default settings so the two can never drift.
    """
    from submission._analysis import analyze
    return analyze(text)


def _iter_jsonl(path: str) -> Iterator[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["doc_id"], obj["text"]


class InvertedIndex:
    """Inverted index with delta+VByte postings and on-disk persistence."""

    def __init__(self, config: Optional[AnalysisConfig] = None,
                 store_positions: bool = False):
        self.config = config or AnalysisConfig()
        # Positions cost roughly one VByte per token occurrence and are only
        # needed by proximity/phrase scoring, so they are opt-in: an index built
        # without them is byte-for-byte what it was before this existed.
        self.store_positions = store_positions
        self._pos_buf = np.zeros(0, dtype=np.uint8)
        self._pos_off = np.zeros(1, dtype=np.int64)
        self.terms: List[str] = []
        self.term_lookup: Dict[str, int] = {}
        self.df = np.zeros(0, dtype=np.int64)
        self.cf = np.zeros(0, dtype=np.int64)
        self.doc_ids: List[str] = []
        self.doc_len = np.zeros(0, dtype=np.int64)
        self.N: int = 0
        self.total_tokens: int = 0
        self.avg_doc_len: float = 0.0
        self._docid_buf = np.zeros(0, dtype=np.uint8)
        self._docid_off = np.zeros(1, dtype=np.int64)
        # Term frequencies are nibble-packed (see submission/_codecs.py). No
        # per-term offset table is needed: a posting's nibble index is its
        # posting index, which `_term_start` (cumulative df) already gives.
        self._tf_packed = np.zeros(0, dtype=np.uint8)
        self._tf_exc_idx = np.zeros(0, dtype=np.int64)
        self._tf_exc_val = np.zeros(0, dtype=np.int64)
        self._term_start = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """Build from a list of (doc_id, text) pairs.

        This is the signature the starter shipped. For a large corpus prefer
        `build_from_jsonl()`, which streams instead of materialising every
        document string in memory first.
        """
        self._build(iter(corpus))

    def build_from_jsonl(self, corpus_path: str) -> None:
        """Build by streaming a corpus.jsonl file -- the memory-safe path."""
        self._build(_iter_jsonl(corpus_path))

    def _build(self, docs: Iterable[Tuple[str, str]]) -> None:
        if self.store_positions:
            return self._build_positional(docs)
        if _FASTBUILD is not None and _FASTBUILD.Builder.supports(self.config):
            return self._build_counts_fast(docs)
        return self._build_counts(docs)

    def _build_counts_fast(self, docs: Iterable[Tuple[str, str]]) -> None:
        """Same index as `_build_counts`, with tokenising and posting emission
        done in C++ (see submission/_fastbuild.pyx).

        Used only when the analysis chain is the default one the kernel
        reproduces exactly; any other configuration falls back to Python rather
        than risking a silently different index.
        """
        builder = _FASTBUILD.Builder(self.config.min_token_len, self.config.max_token_len)
        doc_ids: List[str] = []
        doc_lens: List[int] = []
        for internal_id, (ext_id, text) in enumerate(docs):
            doc_ids.append(ext_id)
            # str.lower() stays in Python: it is already C-speed and applies the
            # full Unicode case mapping the byte scanner cannot.
            doc_lens.append(builder.add_document(text.lower().encode("utf-8"), internal_id))

        self.doc_ids = doc_ids
        self.doc_len = np.array(doc_lens, dtype=np.int64)
        self.N = len(doc_ids)
        self.total_tokens = int(self.doc_len.sum()) if self.N else 0
        self.avg_doc_len = (self.total_tokens / self.N) if self.N else 0.0

        terms_unsorted = builder.terms()
        if not terms_unsorted:
            self._finalise_empty({})
            return

        # Sort the vocabulary, then have the kernel concatenate postings in that
        # order. This replaces the 16.3M-element np.lexsort the document-ordered
        # layout required -- 3.06s of a 5.78s build -- with a single copy pass.
        order = sorted(range(len(terms_unsorted)), key=terms_unsorted.__getitem__)
        sorted_terms = [terms_unsorted[i] for i in order]
        docs_arr, tfs_arr, df = builder.finish_sorted(np.asarray(order, dtype=np.int32))

        self.terms = sorted_terms
        self.term_lookup = {term: i for i, term in enumerate(sorted_terms)}
        self.df = df
        self.cf = np.zeros(len(sorted_terms), dtype=np.int64)
        np.add.reduceat(tfs_arr.astype(np.int64), self._term_starts(df), out=self.cf)
        self._encode_postings(docs_arr.astype(np.int64), tfs_arr.astype(np.int64))

    @staticmethod
    def _term_starts(df: np.ndarray) -> np.ndarray:
        """Index of each term's first posting. Every term has df >= 1, so the
        cumulative document frequency gives this directly."""
        starts = np.empty(df.size, dtype=np.int64)
        starts[0] = 0
        np.cumsum(df[:-1], out=starts[1:])
        return starts

    def _encode_postings(self, post_doc: np.ndarray, post_tf: np.ndarray) -> None:
        """Delta+VByte encode postings that are already grouped by term and
        ascending by doc id within each term."""
        term_start = self._term_starts(self.df)
        gaps = np.empty(post_doc.size, dtype=np.int64)
        gaps[0] = post_doc[0]
        np.subtract(post_doc[1:], post_doc[:-1], out=gaps[1:])
        gaps[term_start] = post_doc[term_start]

        docid_bytes = np.add.reduceat(vbyte_widths(gaps), term_start)
        self._docid_buf = vbyte_encode(gaps)
        self._docid_off = np.concatenate(([0], np.cumsum(docid_bytes))).astype(np.int64)
        self._term_start = term_start
        self._tf_packed, self._tf_exc_idx, self._tf_exc_val = pack_tf_nibbles(post_tf)

    def _build_positional(self, docs: Iterable[Tuple[str, str]]) -> None:
        """Build with term positions retained, for proximity/phrase scoring.

        Emits one (term, doc, position) triple per token occurrence rather than
        one (term, doc, tf) triple per distinct term, then recovers tf by
        grouping. That means term frequency never has to be counted separately:
        a posting's tf is exactly how many positions it owns, which also means
        the positions file needs no per-posting offset table -- the existing tf
        values already delimit it.
        """
        analyzer = make_analyzer(self.config)
        provisional: Dict[str, int] = {}
        doc_ids: List[str] = []
        doc_lens: List[int] = []

        buf_t: List[int] = []
        buf_d: List[int] = []
        buf_p: List[int] = []
        chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def flush() -> None:
            if not buf_t:
                return
            chunks.append((
                np.array(buf_t, dtype=np.int32),
                np.array(buf_d, dtype=np.int32),
                np.array(buf_p, dtype=np.int32),
            ))
            buf_t.clear()
            buf_d.clear()
            buf_p.clear()

        for internal_id, (ext_id, text) in enumerate(docs):
            tokens = analyzer(text)
            doc_ids.append(ext_id)
            doc_lens.append(len(tokens))
            for position, term in enumerate(tokens):
                tid = provisional.get(term)
                if tid is None:
                    tid = len(provisional)
                    provisional[term] = tid
                buf_t.append(tid)
                buf_d.append(internal_id)
                buf_p.append(position)
            if len(buf_t) >= _FLUSH_EVERY:
                flush()
        flush()

        self.doc_ids = doc_ids
        self.doc_len = np.array(doc_lens, dtype=np.int64)
        self.N = len(doc_ids)
        self.total_tokens = int(self.doc_len.sum()) if self.N else 0
        self.avg_doc_len = (self.total_tokens / self.N) if self.N else 0.0

        if not chunks:
            self._finalise_empty(provisional)
            return

        term_ids = np.concatenate([c[0] for c in chunks]).astype(np.int64)
        docs_arr = np.concatenate([c[1] for c in chunks]).astype(np.int64)
        positions = np.concatenate([c[2] for c in chunks]).astype(np.int64)
        del chunks

        sorted_terms = sorted(provisional)
        remap = np.empty(len(sorted_terms), dtype=np.int64)
        for new_id, term in enumerate(sorted_terms):
            remap[provisional[term]] = new_id
        term_ids = remap[term_ids]

        # Group by term, then doc, then position ascending.
        order = np.lexsort((positions, docs_arr, term_ids))
        term_ids = term_ids[order]
        docs_arr = docs_arr[order]
        positions = positions[order]
        del order

        n_tokens = term_ids.size
        # A new posting begins wherever (term, doc) changes.
        is_new = np.empty(n_tokens, dtype=bool)
        is_new[0] = True
        np.logical_or(term_ids[1:] != term_ids[:-1], docs_arr[1:] != docs_arr[:-1], out=is_new[1:])
        posting_start = np.flatnonzero(is_new)

        post_term = term_ids[posting_start]
        post_doc = docs_arr[posting_start]
        post_tf = np.diff(np.append(posting_start, n_tokens))

        n_terms = len(sorted_terms)
        self.terms = sorted_terms
        self.term_lookup = {term: i for i, term in enumerate(sorted_terms)}
        self.df = np.bincount(post_term, minlength=n_terms).astype(np.int64)
        self.cf = np.bincount(post_term, weights=post_tf, minlength=n_terms).astype(np.int64)

        term_start = np.empty(n_terms, dtype=np.int64)
        term_start[0] = 0
        np.cumsum(self.df[:-1], out=term_start[1:])

        gaps = np.empty(post_doc.size, dtype=np.int64)
        gaps[0] = post_doc[0]
        np.subtract(post_doc[1:], post_doc[:-1], out=gaps[1:])
        gaps[term_start] = post_doc[term_start]

        docid_bytes = np.add.reduceat(vbyte_widths(gaps), term_start)
        self._docid_buf = vbyte_encode(gaps)
        self._docid_off = np.concatenate(([0], np.cumsum(docid_bytes))).astype(np.int64)
        self._term_start = term_start
        self._tf_packed, self._tf_exc_idx, self._tf_exc_val = pack_tf_nibbles(post_tf)

        # Positions delta-encoded within each posting (they are ascending there),
        # restarting the chain at every posting boundary.
        pos_gaps = np.empty(n_tokens, dtype=np.int64)
        pos_gaps[0] = positions[0]
        np.subtract(positions[1:], positions[:-1], out=pos_gaps[1:])
        pos_gaps[posting_start] = positions[posting_start]

        # Each term's slice of the position buffer starts at its first posting.
        pos_term_start = posting_start[term_start]
        pos_bytes = np.add.reduceat(vbyte_widths(pos_gaps), pos_term_start)
        self._pos_buf = vbyte_encode(pos_gaps)
        self._pos_off = np.concatenate(([0], np.cumsum(pos_bytes))).astype(np.int64)

    def _build_counts(self, docs: Iterable[Tuple[str, str]]) -> None:
        analyzer = make_analyzer(self.config)  # holds the stem cache across docs

        provisional: Dict[str, int] = {}     # term -> first-seen id
        doc_ids: List[str] = []
        doc_lens: List[int] = []

        buf_t: List[int] = []
        buf_d: List[int] = []
        buf_f: List[int] = []
        chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def flush() -> None:
            if not buf_t:
                return
            chunks.append((
                np.array(buf_t, dtype=np.int64),
                np.array(buf_d, dtype=np.int64),
                np.array(buf_f, dtype=np.int64),
            ))
            buf_t.clear()
            buf_d.clear()
            buf_f.clear()

        for internal_id, (ext_id, text) in enumerate(docs):
            tokens = analyzer(text)
            doc_ids.append(ext_id)
            doc_lens.append(len(tokens))
            # Counter preserves first-seen order, so this loop is deterministic.
            for term, tf in Counter(tokens).items():
                tid = provisional.get(term)
                if tid is None:
                    tid = len(provisional)
                    provisional[term] = tid
                buf_t.append(tid)
                buf_d.append(internal_id)
                buf_f.append(tf)
            if len(buf_t) >= _FLUSH_EVERY:
                flush()
        flush()

        self.doc_ids = doc_ids
        self.doc_len = np.array(doc_lens, dtype=np.int64)
        self.N = len(doc_ids)
        self.total_tokens = int(self.doc_len.sum()) if self.N else 0
        self.avg_doc_len = (self.total_tokens / self.N) if self.N else 0.0

        if not chunks:
            self._finalise_empty(provisional)
            return

        term_ids = np.concatenate([c[0] for c in chunks])
        docs_arr = np.concatenate([c[1] for c in chunks])
        tfs_arr = np.concatenate([c[2] for c in chunks])
        del chunks

        # Reassign term ids so they follow alphabetical order. Sorting the terms
        # makes the dictionary compressible (shared prefixes end up adjacent)
        # and makes the build deterministic regardless of corpus order.
        self._finalise_postings(list(provisional), term_ids, docs_arr, tfs_arr,
                                first_seen=provisional)

    def _finalise_postings(self, seen_terms, term_ids, docs_arr, tfs_arr,
                           first_seen=None) -> None:
        """Sort, group and encode postings. Shared by the Python and C++ paths
        so both produce a byte-identical index."""
        provisional = first_seen if first_seen is not None else {
            t: i for i, t in enumerate(seen_terms)}
        sorted_terms = sorted(provisional)
        remap = np.empty(len(sorted_terms), dtype=np.int64)
        for new_id, term in enumerate(sorted_terms):
            remap[provisional[term]] = new_id
        term_ids = remap[term_ids]

        # Group by term, ascending docid within each term. lexsort's LAST key is
        # primary, so this is "sort by term, then by doc".
        order = np.lexsort((docs_arr, term_ids))
        term_ids = term_ids[order]
        docs_arr = docs_arr[order]
        tfs_arr = tfs_arr[order]
        del order

        n_terms = len(sorted_terms)
        self.terms = sorted_terms
        self.term_lookup = {term: i for i, term in enumerate(sorted_terms)}
        self.df = np.bincount(term_ids, minlength=n_terms).astype(np.int64)
        self.cf = np.bincount(term_ids, weights=tfs_arr, minlength=n_terms).astype(np.int64)

        # Every term has df >= 1, so cumulative df gives each term's first
        # posting index directly.
        starts = np.empty(n_terms, dtype=np.int64)
        starts[0] = 0
        np.cumsum(self.df[:-1], out=starts[1:])

        # Delta-encode docids, restarting the gap chain at each term boundary.
        gaps = np.empty(docs_arr.size, dtype=np.int64)
        gaps[0] = docs_arr[0]
        np.subtract(docs_arr[1:], docs_arr[:-1], out=gaps[1:])
        gaps[starts] = docs_arr[starts]

        # Per-term byte lengths, computed without encoding term-by-term.
        docid_bytes = np.add.reduceat(vbyte_widths(gaps), starts)
        self._docid_buf = vbyte_encode(gaps)
        self._docid_off = np.concatenate(([0], np.cumsum(docid_bytes))).astype(np.int64)
        self._term_start = starts
        self._tf_packed, self._tf_exc_idx, self._tf_exc_val = pack_tf_nibbles(tfs_arr)

    def _finalise_empty(self, provisional: Dict[str, int]) -> None:
        self.terms = sorted(provisional)
        self.term_lookup = {t: i for i, t in enumerate(self.terms)}
        n = len(self.terms)
        self.df = np.zeros(n, dtype=np.int64)
        self.cf = np.zeros(n, dtype=np.int64)
        self._docid_off = np.zeros(n + 1, dtype=np.int64)
        self._term_start = np.zeros(n, dtype=np.int64)
        self._pos_off = np.zeros(n + 1, dtype=np.int64)

    # ------------------------------------------------------------------
    # Query-time accessors
    # ------------------------------------------------------------------
    def term_id(self, term: str) -> int:
        """Internal id for `term`, or -1 if it is not in the vocabulary."""
        return self.term_lookup.get(term, -1)

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        tid = self.term_lookup.get(term)
        return int(self.df[tid]) if tid is not None else 0

    def collection_frequency(self, term: str) -> int:
        """Total occurrences of `term` across the whole collection."""
        tid = self.term_lookup.get(term)
        return int(self.cf[tid]) if tid is not None else 0

    def postings(self, term: str) -> Tuple[np.ndarray, np.ndarray]:
        """Decode `term`'s postings list.

        Returns (doc_ids, term_frequencies) as parallel arrays with doc_ids
        ascending -- internal integer ids, not external strings. Returns empty
        arrays for an unknown term.
        """
        tid = self.term_lookup.get(term)
        if tid is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return self.postings_by_id(tid)

    def postings_by_id(self, tid: int) -> Tuple[np.ndarray, np.ndarray]:
        """`postings()` for an already-resolved term id."""
        count = int(self.df[tid])
        gaps = vbyte_decode(self._docid_buf[self._docid_off[tid]:self._docid_off[tid + 1]], count)
        tfs = unpack_tf_nibbles(self._tf_packed, int(self._term_start[tid]), count,
                                self._tf_exc_idx, self._tf_exc_val)
        return np.cumsum(gaps), tfs

    def postings_with_positions(self, tid: int):
        """Decode a term's postings together with its term positions.

        Returns (doc_ids, tfs, positions, offsets) where `positions` is the flat
        concatenation of every posting's position list and `offsets[i]` is where
        document i's positions begin. tf doubles as the per-posting length, so
        no separate offset table is stored on disk.
        """
        if not self.store_positions:
            raise RuntimeError("this index was built without positions")
        doc_ids, tfs = self.postings_by_id(tid)
        total = int(self.cf[tid])
        gaps = vbyte_decode(self._pos_buf[self._pos_off[tid]:self._pos_off[tid + 1]], total)

        offsets = np.empty(tfs.size, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(tfs[:-1], out=offsets[1:])
        # Undo the per-posting delta chains: cumulative sum, minus each
        # posting's running base.
        running = np.cumsum(gaps)
        base = np.zeros(tfs.size, dtype=np.int64)
        base[1:] = running[offsets[1:] - 1]
        positions = running - np.repeat(base, tfs)
        return doc_ids, tfs, positions, offsets

    def external_id(self, internal_id: int) -> str:
        return self.doc_ids[internal_id]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _write_blob(path: str, payload: bytes) -> None:
        with open(path, "wb") as f:
            f.write(zlib.compress(payload, _ZLIB_LEVEL))

    @staticmethod
    def _read_blob(path: str) -> bytes:
        with open(path, "rb") as f:
            return zlib.decompress(f.read())
    def save(self, index_dir: str) -> None:
        """Persist everything `retrieve()` needs, and nothing else.

        Deliberately omits raw document text: BM25, VSM and the LM/DFR scorers
        need only term-frequency and length statistics, and storing 189MB of
        text would wreck the index-size component for zero query-time benefit.
        """
        os.makedirs(index_dir, exist_ok=True)

        meta = {
            "format_version": FORMAT_VERSION,
            "n_docs": self.N,
            "n_terms": len(self.terms),
            "total_tokens": self.total_tokens,
            "avg_doc_len": self.avg_doc_len,
            "analysis": self.config.to_dict(),
            "store_positions": self.store_positions,
            "n_tf_exceptions": int(self._tf_exc_idx.size),
        }
        with open(os.path.join(index_dir, _META), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # The tokenizer emits [a-z0-9]+ only, and external doc_ids in this
        # collection carry no newlines, so newline framing is unambiguous.
        self._write_blob(os.path.join(index_dir, _TERMS),
                         "\n".join(self.terms).encode("utf-8"))
        self._write_blob(os.path.join(index_dir, _DOCIDS),
                         "\n".join(self.doc_ids).encode("utf-8"))

        # Per-term byte lengths rather than absolute offsets: the lengths are
        # small integers that VByte to ~1 byte, while absolute offsets grow to
        # 4-5 bytes each. Reconstructed by cumsum at load.
        docid_len = np.diff(self._docid_off)

        for name, arr in (
            (_DF, self.df),
            (_CF, self.cf),
            (_DOCLEN, self.doc_len),
            (_DOCID_LEN, docid_len),
        ):
            self._write_blob(os.path.join(index_dir, name), vbyte_encode(arr).tobytes())

        self._write_blob(os.path.join(index_dir, _POSTINGS_D), self._docid_buf.tobytes())
        self._write_blob(os.path.join(index_dir, _TF_NIB), self._tf_packed.tobytes())
        # Exception positions are ascending, so gap-encode them like doc ids.
        exc_gaps = np.diff(self._tf_exc_idx, prepend=0) if self._tf_exc_idx.size else self._tf_exc_idx
        self._write_blob(os.path.join(index_dir, _TF_EXC_I), vbyte_encode(exc_gaps).tobytes())
        self._write_blob(os.path.join(index_dir, _TF_EXC_V), vbyte_encode(self._tf_exc_val).tobytes())

        if self.store_positions:
            self._write_blob(os.path.join(index_dir, _POS_LEN),
                             vbyte_encode(np.diff(self._pos_off)).tobytes())
            self._write_blob(os.path.join(index_dir, _POSITIONS), self._pos_buf.tobytes())

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct an index from `index_dir` alone, in a fresh process."""
        with open(os.path.join(index_dir, _META), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"index format version {meta.get('format_version')} != "
                f"{FORMAT_VERSION}; rebuild the index"
            )

        index = cls(AnalysisConfig.from_dict(meta.get("analysis", {})),
                    store_positions=bool(meta.get("store_positions", False)))
        index.N = int(meta["n_docs"])
        index.total_tokens = int(meta["total_tokens"])
        index.avg_doc_len = float(meta["avg_doc_len"])
        n_terms = int(meta["n_terms"])

        def read_text(name: str) -> List[str]:
            blob = cls._read_blob(os.path.join(index_dir, name)).decode("utf-8")
            return blob.split("\n") if blob else []

        def read_raw(name: str) -> np.ndarray:
            return np.frombuffer(cls._read_blob(os.path.join(index_dir, name)),
                                 dtype=np.uint8)

        def read_vbyte(name: str, count: int) -> np.ndarray:
            return vbyte_decode(read_raw(name), count)

        index.terms = read_text(_TERMS)
        index.doc_ids = read_text(_DOCIDS)
        index.term_lookup = {term: i for i, term in enumerate(index.terms)}

        index.df = read_vbyte(_DF, n_terms)
        index.cf = read_vbyte(_CF, n_terms)
        index.doc_len = read_vbyte(_DOCLEN, index.N)

        docid_len = read_vbyte(_DOCID_LEN, n_terms)
        index._docid_off = np.concatenate(([0], np.cumsum(docid_len))).astype(np.int64)
        index._term_start = cls._term_starts(index.df) if n_terms else np.zeros(0, dtype=np.int64)

        index._docid_buf = read_raw(_POSTINGS_D)
        index._tf_packed = read_raw(_TF_NIB)
        n_exc = int(meta.get("n_tf_exceptions", 0))
        index._tf_exc_idx = np.cumsum(read_vbyte(_TF_EXC_I, n_exc)) if n_exc else np.zeros(0, dtype=np.int64)
        index._tf_exc_val = read_vbyte(_TF_EXC_V, n_exc) if n_exc else np.zeros(0, dtype=np.int64)

        if index.store_positions:
            pos_len = read_vbyte(_POS_LEN, n_terms)
            index._pos_off = np.concatenate(([0], np.cumsum(pos_len))).astype(np.int64)
            index._pos_buf = read_raw(_POSITIONS)
        return index
