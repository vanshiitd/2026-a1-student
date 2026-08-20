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
    tf_off[t:t+2]            byte slice of this term's tfs in `_tf_buf`
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
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from submission._analysis import AnalysisConfig, make_analyzer
from submission._codecs import vbyte_decode, vbyte_encode, vbyte_widths

FORMAT_VERSION = 1

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
_TF_LEN = "tf_len.bin"
_POSTINGS_D = "postings_d.bin"
_POSTINGS_F = "postings_f.bin"


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

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
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
        self._tf_buf = np.zeros(0, dtype=np.uint8)
        self._docid_off = np.zeros(1, dtype=np.int64)
        self._tf_off = np.zeros(1, dtype=np.int64)

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
        tf_bytes = np.add.reduceat(vbyte_widths(tfs_arr), starts)

        self._docid_buf = vbyte_encode(gaps)
        self._tf_buf = vbyte_encode(tfs_arr)
        self._docid_off = np.concatenate(([0], np.cumsum(docid_bytes))).astype(np.int64)
        self._tf_off = np.concatenate(([0], np.cumsum(tf_bytes))).astype(np.int64)

    def _finalise_empty(self, provisional: Dict[str, int]) -> None:
        self.terms = sorted(provisional)
        self.term_lookup = {t: i for i, t in enumerate(self.terms)}
        n = len(self.terms)
        self.df = np.zeros(n, dtype=np.int64)
        self.cf = np.zeros(n, dtype=np.int64)
        self._docid_off = np.zeros(n + 1, dtype=np.int64)
        self._tf_off = np.zeros(n + 1, dtype=np.int64)

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
        tfs = vbyte_decode(self._tf_buf[self._tf_off[tid]:self._tf_off[tid + 1]], count)
        return np.cumsum(gaps), tfs

    def external_id(self, internal_id: int) -> str:
        return self.doc_ids[internal_id]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
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
        }
        with open(os.path.join(index_dir, _META), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # The tokenizer emits [a-z0-9]+ only, and external doc_ids in this
        # collection carry no newlines, so newline framing is unambiguous.
        with open(os.path.join(index_dir, _TERMS), "w", encoding="utf-8") as f:
            f.write("\n".join(self.terms))
        with open(os.path.join(index_dir, _DOCIDS), "w", encoding="utf-8") as f:
            f.write("\n".join(self.doc_ids))

        # Per-term byte lengths rather than absolute offsets: the lengths are
        # small integers that VByte to ~1 byte, while absolute offsets grow to
        # 4-5 bytes each. Reconstructed by cumsum at load.
        docid_len = np.diff(self._docid_off)
        tf_len = np.diff(self._tf_off)

        for name, arr in (
            (_DF, self.df),
            (_CF, self.cf),
            (_DOCLEN, self.doc_len),
            (_DOCID_LEN, docid_len),
            (_TF_LEN, tf_len),
        ):
            vbyte_encode(arr).tofile(os.path.join(index_dir, name))

        self._docid_buf.tofile(os.path.join(index_dir, _POSTINGS_D))
        self._tf_buf.tofile(os.path.join(index_dir, _POSTINGS_F))

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

        index = cls(AnalysisConfig.from_dict(meta.get("analysis", {})))
        index.N = int(meta["n_docs"])
        index.total_tokens = int(meta["total_tokens"])
        index.avg_doc_len = float(meta["avg_doc_len"])
        n_terms = int(meta["n_terms"])

        def read_text(name: str) -> List[str]:
            with open(os.path.join(index_dir, name), "r", encoding="utf-8") as fh:
                blob = fh.read()
            return blob.split("\n") if blob else []

        def read_vbyte(name: str, count: int) -> np.ndarray:
            return vbyte_decode(np.fromfile(os.path.join(index_dir, name), dtype=np.uint8), count)

        index.terms = read_text(_TERMS)
        index.doc_ids = read_text(_DOCIDS)
        index.term_lookup = {term: i for i, term in enumerate(index.terms)}

        index.df = read_vbyte(_DF, n_terms)
        index.cf = read_vbyte(_CF, n_terms)
        index.doc_len = read_vbyte(_DOCLEN, index.N)

        docid_len = read_vbyte(_DOCID_LEN, n_terms)
        tf_len = read_vbyte(_TF_LEN, n_terms)
        index._docid_off = np.concatenate(([0], np.cumsum(docid_len))).astype(np.int64)
        index._tf_off = np.concatenate(([0], np.cumsum(tf_len))).astype(np.int64)

        index._docid_buf = np.fromfile(os.path.join(index_dir, _POSTINGS_D), dtype=np.uint8)
        index._tf_buf = np.fromfile(os.path.join(index_dir, _POSTINGS_F), dtype=np.uint8)
        return index
