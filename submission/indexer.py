"""
submission/indexer.py -- the inverted index.

using flat numpy arrays not dict of dicts, otherwise ~16M postings just
blow up interpreter memory way past 8gb

    terms[t]           term string, sorted
    df[t], cf[t]        doc / collection freq
    docid_off[t:t+2]    byte slice in _docid_buf for this term
    term_start[t]       first posting index for term t
    doc_len[d]          doc length in tokens
    doc_ids[d]          external doc_id for internal id d

postings are delta + vbyte encoded, whole collection at once not per term.
save/load are just plain files, no pickling. raw doc text never stored.
"""
import json
import multiprocessing
import os
import zlib
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from submission._analysis import AnalysisConfig, make_analyzer
from submission._codecs import (pack_tf_nibbles, unpack_tf_nibbles, vbyte_decode,
                                vbyte_encode, vbyte_widths)

# below this doc count parallel build isn't worth the process overhead
_PARALLEL_MIN_DOCS = 20_000

# c++ tokenizer, optional -- falls back to python if it didn't compile
try:
    from submission import _fastbuild as _FASTBUILD
except ImportError:  # pragma: no cover - exercised by the fallback test
    _FASTBUILD = None

FORMAT_VERSION = 3

_ZLIB_LEVEL = 4  # measured, higher levels barely shrink but cost build time

_FLUSH_EVERY = 2_000_000  # flush python lists to numpy every this many postings

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
    """kept for interface compat, delegates to _analysis so they don't drift"""
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


def _split_byte_ranges(corpus_path: str, n_workers: int):
    """newline aligned byte offsets to split the file into n_workers pieces,
    also figures out each piece's starting doc id (needs a line count pass)"""
    size = os.path.getsize(corpus_path)
    target = size // n_workers
    ranges = []
    doc_id_start = 0
    with open(corpus_path, "rb") as f:
        pos = 0
        for w in range(n_workers):
            start = pos
            if w == n_workers - 1:
                end = size
            else:
                seek_to = min(start + target, size)
                f.seek(seek_to)
                f.readline()  # land on a line boundary
                end = f.tell()
            f.seek(start)
            n_lines = 0
            while f.tell() < end:
                line = f.readline()
                if line.strip():
                    n_lines += 1
            ranges.append((start, end, doc_id_start))
            doc_id_start += n_lines
            pos = end
    return ranges


def _detected_worker_count() -> int:
    """os.cpu_count() can report the HOST's cpu count inside a container,
    not what the container is actually limited to -- hit this for real on
    the grading machine (it spawned way too many workers). use
    sched_getaffinity where available since it respects the real limit,
    and cap at 8 regardless since more doesn't help anyway"""
    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, 8))


def _parallel_build_worker(args):
    """runs in its own process, tokenises one byte range. module level fn
    not a method since multiprocessing needs to pickle it by reference"""
    (corpus_path, byte_start, byte_end, doc_id_start, min_token_len,
     max_token_len, stem_tokens, prefix_tokens) = args
    from submission import _fastbuild as fb  # fresh process, re-import

    builder = fb.Builder(min_token_len, max_token_len, stem_tokens)
    doc_ids: List[str] = []
    doc_lens: List[int] = []
    with open(corpus_path, "rb") as f:
        f.seek(byte_start)
        internal_id = doc_id_start
        while f.tell() < byte_end:
            raw = f.readline()
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_ids.append(obj["doc_id"])
            doc_lens.append(builder.add_document(
                obj["text"].lower().encode("utf-8"), internal_id, prefix_tokens))
            internal_id += 1

    terms_unsorted = builder.terms()
    if not terms_unsorted:
        # keep int32 here to match finish_sorted()'s dtype, else concatenate
        # later silently upcasts everything to int64
        return doc_ids, doc_lens, [], np.zeros(0, dtype=np.int32), \
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)

    # identity order (unsorted, first seen) -- real sort happens once at merge
    identity = np.arange(len(terms_unsorted), dtype=np.int32)
    docs_arr, tfs_arr, df = builder.finish_sorted(identity)
    return doc_ids, doc_lens, terms_unsorted, docs_arr, tfs_arr, df


class InvertedIndex:
    """inverted index, delta+vbyte postings, on disk persistence"""

    def __init__(self, config: Optional[AnalysisConfig] = None,
                 store_positions: bool = False):
        self.config = config or AnalysisConfig()
        self.store_positions = store_positions  # opt in, only needed for proximity
        self._prefix_tokens = -1
        self.store_doc_ids = True  # title index shares main's doc order, doesn't need own copy
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
        self._tf_packed = np.zeros(0, dtype=np.uint8)
        self._tf_exc_idx = np.zeros(0, dtype=np.int64)
        self._tf_exc_val = np.zeros(0, dtype=np.int64)
        self._term_start = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """starter signature, prefer build_from_jsonl for big corpora"""
        self._build(iter(corpus))

    def build_from_jsonl(self, corpus_path: str, prefix_tokens: int = -1) -> None:
        """streams the corpus, memory safe. prefix_tokens for the title field"""
        self._prefix_tokens = prefix_tokens
        self._build(_iter_jsonl(corpus_path))

    def _build(self, docs: Iterable[Tuple[str, str]]) -> None:
        if self.store_positions:
            return self._build_positional(docs)
        if _FASTBUILD is not None and _FASTBUILD.Builder.supports(self.config):
            return self._build_counts_fast(docs)
        return self._build_counts(docs)

    def build_from_jsonl_parallel(self, corpus_path: str, prefix_tokens: int = -1,
                                  n_workers: Optional[int] = None,
                                  min_docs: int = _PARALLEL_MIN_DOCS) -> bool:
        """splits tokenising across n_workers processes, falls back to
        build_from_jsonl() if it can't (corpus too small, config not
        supported by the c++ builder etc). only tokenising is parallel,
        postings assembly stays serial so don't expect a clean 4x"""
        self._prefix_tokens = prefix_tokens
        if _FASTBUILD is None or not _FASTBUILD.Builder.supports(self.config):
            return False
        if self.store_positions:
            return False

        n_workers = n_workers or _detected_worker_count()
        size = os.path.getsize(corpus_path)
        approx_docs = size // 200  # rough estimate, just for the gate check
        if n_workers < 2 or approx_docs < min_docs:
            return False

        ranges = _split_byte_ranges(corpus_path, n_workers)
        tasks = [(corpus_path, start, end, doc_id_start,
                 self.config.min_token_len, self.config.max_token_len,
                 self.config.stemmer == "porter", prefix_tokens)
                for start, end, doc_id_start in ranges]

        # each worker re-imports numpy fresh and its BLAS backend spins up
        # its own thread pool sized off visible cpu count -- combined with
        # n_workers processes this actually crashed on the grading machine
        # (pthread_create failing). force BLAS single threaded in workers,
        # multiprocessing is already doing the parallelism
        for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[_var] = "1"

        ctx = multiprocessing.get_context("spawn")  # fork unsafe w/ compiled ext loaded
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_parallel_build_worker, tasks)

        self._merge_parallel_results(results)
        return True

    def _merge_parallel_results(self, results) -> None:
        """union each worker's vocab, remap, concat in worker order.
        concat works (no k-way merge needed) since each worker's doc range
        is contiguous and non overlapping"""
        doc_ids: List[str] = []
        doc_lens: List[int] = []
        for w_doc_ids, w_doc_lens, _terms, _d, _t, _df in results:
            doc_ids.extend(w_doc_ids)
            doc_lens.extend(w_doc_lens)

        self.doc_ids = doc_ids
        self.doc_len = np.array(doc_lens, dtype=np.int64)
        self.N = len(doc_ids)
        self.total_tokens = int(self.doc_len.sum()) if self.N else 0
        self.avg_doc_len = (self.total_tokens / self.N) if self.N else 0.0

        all_terms = set()
        for _di, _dl, terms, _d, _t, _df in results:
            all_terms.update(terms)
        if not all_terms:
            self._finalise_empty({})
            return

        sorted_terms = sorted(all_terms)
        global_index = {t: i for i, t in enumerate(sorted_terms)}
        n_global = len(sorted_terms)

        per_worker = []
        for _di, _dl, terms, docs_arr, tfs_arr, df in results:
            local_start = self._term_starts(df) if df.size else np.zeros(0, dtype=np.int64)
            g_of_l = np.fromiter((global_index[t] for t in terms), dtype=np.int64,
                                 count=len(terms))
            per_worker.append((docs_arr, tfs_arr, df, local_start, g_of_l))

        docs_chunks: List[np.ndarray] = []
        tfs_chunks: List[np.ndarray] = []
        df_final = np.zeros(n_global, dtype=np.int64)
        inv_maps = []  # global->local per worker, -1 = worker never saw it
        for docs_arr, tfs_arr, df, local_start, g_of_l in per_worker:
            inv = np.full(n_global, -1, dtype=np.int64)
            if g_of_l.size:
                inv[g_of_l] = np.arange(g_of_l.size, dtype=np.int64)
            inv_maps.append(inv)

        for g in range(n_global):
            for (docs_arr, tfs_arr, df, local_start, _g_of_l), inv in zip(per_worker, inv_maps):
                l = inv[g]
                if l < 0:
                    continue
                s = int(local_start[l])
                e = s + int(df[l])
                docs_chunks.append(docs_arr[s:e])
                tfs_chunks.append(tfs_arr[s:e])
                df_final[g] += (e - s)

        # int32 to match finish_sorted()'s dtype
        docs_arr = (np.concatenate(docs_chunks) if docs_chunks
                   else np.zeros(0, dtype=np.int32))
        tfs_arr = (np.concatenate(tfs_chunks) if tfs_chunks
                  else np.zeros(0, dtype=np.int32))

        self.terms = sorted_terms
        self.term_lookup = global_index
        self.df = df_final
        self.cf = np.zeros(n_global, dtype=np.int64)
        if docs_arr.size:
            # reduceat's out= upcasts int32->int64 automatically, no need
            # to cast the input first (would just allocate a big extra copy)
            np.add.reduceat(tfs_arr, self._term_starts(df_final), out=self.cf)
        self._encode_postings(docs_arr, tfs_arr)

    def _build_counts_fast(self, docs: Iterable[Tuple[str, str]]) -> None:
        """same as _build_counts but tokenising done in c++"""
        builder = _FASTBUILD.Builder(self.config.min_token_len, self.config.max_token_len,
                                     self.config.stemmer == "porter")
        doc_ids: List[str] = []
        doc_lens: List[int] = []
        for internal_id, (ext_id, text) in enumerate(docs):
            doc_ids.append(ext_id)
            doc_lens.append(builder.add_document(
                text.lower().encode("utf-8"), internal_id, self._prefix_tokens))

        self.doc_ids = doc_ids
        self.doc_len = np.array(doc_lens, dtype=np.int64)
        self.N = len(doc_ids)
        self.total_tokens = int(self.doc_len.sum()) if self.N else 0
        self.avg_doc_len = (self.total_tokens / self.N) if self.N else 0.0

        terms_unsorted = builder.terms()
        if not terms_unsorted:
            self._finalise_empty({})
            return

        # sort vocab, let kernel emit postings already in that order
        order = sorted(range(len(terms_unsorted)), key=terms_unsorted.__getitem__)
        sorted_terms = [terms_unsorted[i] for i in order]
        docs_arr, tfs_arr, df = builder.finish_sorted(np.asarray(order, dtype=np.int32))

        self.terms = sorted_terms
        self.term_lookup = {term: i for i, term in enumerate(sorted_terms)}
        self.df = df
        self.cf = np.zeros(len(sorted_terms), dtype=np.int64)
        np.add.reduceat(tfs_arr, self._term_starts(df), out=self.cf)
        self._encode_postings(docs_arr, tfs_arr)

    @staticmethod
    def _term_starts(df: np.ndarray) -> np.ndarray:
        starts = np.empty(df.size, dtype=np.int64)
        starts[0] = 0
        np.cumsum(df[:-1], out=starts[1:])
        return starts

    def _encode_postings(self, post_doc: np.ndarray, post_tf: np.ndarray) -> None:
        """delta+vbyte encode postings, already grouped by term + doc asc"""
        term_start = self._term_starts(self.df)
        # int32 not int64, doc id delta bounded by self.N anyway, saves a lot
        gaps = np.empty(post_doc.size, dtype=np.int32)
        gaps[0] = post_doc[0]
        np.subtract(post_doc[1:], post_doc[:-1], out=gaps[1:])
        gaps[term_start] = post_doc[term_start]

        docid_bytes = np.add.reduceat(vbyte_widths(gaps), term_start)
        self._docid_buf = vbyte_encode(gaps)
        self._docid_off = np.concatenate(([0], np.cumsum(docid_bytes))).astype(np.int64)
        self._term_start = term_start
        self._tf_packed, self._tf_exc_idx, self._tf_exc_val = pack_tf_nibbles(post_tf)

    def _build_positional(self, docs: Iterable[Tuple[str, str]]) -> None:
        """same as _build_counts but keeps term positions for proximity/SDM"""
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

        order = np.lexsort((positions, docs_arr, term_ids))  # sort by term, doc, pos
        term_ids = term_ids[order]
        docs_arr = docs_arr[order]
        positions = positions[order]
        del order

        n_tokens = term_ids.size
        is_new = np.empty(n_tokens, dtype=bool)  # new posting where (term,doc) changes
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

        # positions delta encoded within each posting, restart at each boundary
        pos_gaps = np.empty(n_tokens, dtype=np.int64)
        pos_gaps[0] = positions[0]
        np.subtract(positions[1:], positions[:-1], out=pos_gaps[1:])
        pos_gaps[posting_start] = positions[posting_start]

        pos_term_start = posting_start[term_start]
        pos_bytes = np.add.reduceat(vbyte_widths(pos_gaps), pos_term_start)
        self._pos_buf = vbyte_encode(pos_gaps)
        self._pos_off = np.concatenate(([0], np.cumsum(pos_bytes))).astype(np.int64)

    def _build_counts(self, docs: Iterable[Tuple[str, str]]) -> None:
        analyzer = make_analyzer(self.config)  # keeps stem cache warm across docs

        provisional: Dict[str, int] = {}
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
            if self._prefix_tokens >= 0:
                tokens = tokens[:self._prefix_tokens]
            doc_ids.append(ext_id)
            doc_lens.append(len(tokens))
            for term, tf in Counter(tokens).items():  # Counter keeps first-seen order
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

        self._finalise_postings(list(provisional), term_ids, docs_arr, tfs_arr,
                                first_seen=provisional)

    def _finalise_postings(self, seen_terms, term_ids, docs_arr, tfs_arr,
                           first_seen=None) -> None:
        """sort, group, encode. shared by python + c++ paths so both give
        byte identical index"""
        provisional = first_seen if first_seen is not None else {
            t: i for i, t in enumerate(seen_terms)}
        sorted_terms = sorted(provisional)
        remap = np.empty(len(sorted_terms), dtype=np.int64)
        for new_id, term in enumerate(sorted_terms):
            remap[provisional[term]] = new_id
        term_ids = remap[term_ids]

        order = np.lexsort((docs_arr, term_ids))  # sort by term then doc
        term_ids = term_ids[order]
        docs_arr = docs_arr[order]
        tfs_arr = tfs_arr[order]
        del order

        n_terms = len(sorted_terms)
        self.terms = sorted_terms
        self.term_lookup = {term: i for i, term in enumerate(sorted_terms)}
        self.df = np.bincount(term_ids, minlength=n_terms).astype(np.int64)
        self.cf = np.bincount(term_ids, weights=tfs_arr, minlength=n_terms).astype(np.int64)

        starts = np.empty(n_terms, dtype=np.int64)
        starts[0] = 0
        np.cumsum(self.df[:-1], out=starts[1:])

        gaps = np.empty(docs_arr.size, dtype=np.int64)
        gaps[0] = docs_arr[0]
        np.subtract(docs_arr[1:], docs_arr[:-1], out=gaps[1:])
        gaps[starts] = docs_arr[starts]

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
        return self.term_lookup.get(term, -1)

    def document_frequency(self, term: str) -> int:
        tid = self.term_lookup.get(term)
        return int(self.df[tid]) if tid is not None else 0

    def collection_frequency(self, term: str) -> int:
        tid = self.term_lookup.get(term)
        return int(self.cf[tid]) if tid is not None else 0

    def postings(self, term: str) -> Tuple[np.ndarray, np.ndarray]:
        """(doc_ids, tfs) parallel arrays, doc_ids ascending, internal ids"""
        tid = self.term_lookup.get(term)
        if tid is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return self.postings_by_id(tid)

    def postings_by_id(self, tid: int) -> Tuple[np.ndarray, np.ndarray]:
        count = int(self.df[tid])
        gaps = vbyte_decode(self._docid_buf[self._docid_off[tid]:self._docid_off[tid + 1]], count)
        tfs = unpack_tf_nibbles(self._tf_packed, int(self._term_start[tid]), count,
                                self._tf_exc_idx, self._tf_exc_val)
        return np.cumsum(gaps), tfs

    def postings_with_positions(self, tid: int):
        """(doc_ids, tfs, positions, offsets). tf doubles as per-posting
        position count so no separate offset table needed on disk"""
        if not self.store_positions:
            raise RuntimeError("this index was built without positions")
        doc_ids, tfs = self.postings_by_id(tid)
        total = int(self.cf[tid])
        gaps = vbyte_decode(self._pos_buf[self._pos_off[tid]:self._pos_off[tid + 1]], total)

        offsets = np.empty(tfs.size, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(tfs[:-1], out=offsets[1:])
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
        """writes everything retrieve() needs, nothing else. no raw text"""
        os.makedirs(index_dir, exist_ok=True)

        meta = {
            "format_version": FORMAT_VERSION,
            "n_docs": self.N,
            "n_terms": len(self.terms),
            "total_tokens": self.total_tokens,
            "avg_doc_len": self.avg_doc_len,
            "analysis": self.config.to_dict(),
            "store_positions": self.store_positions,
            "store_doc_ids": self.store_doc_ids,
            "n_tf_exceptions": int(self._tf_exc_idx.size),
        }
        with open(os.path.join(index_dir, _META), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        self._write_blob(os.path.join(index_dir, _TERMS),
                         "\n".join(self.terms).encode("utf-8"))
        if self.store_doc_ids:
            self._write_blob(os.path.join(index_dir, _DOCIDS),
                             "\n".join(self.doc_ids).encode("utf-8"))

        docid_len = np.diff(self._docid_off)  # per-term lengths, not offsets, vbytes smaller

        for name, arr in (
            (_DF, self.df),
            (_CF, self.cf),
            (_DOCLEN, self.doc_len),
            (_DOCID_LEN, docid_len),
        ):
            self._write_blob(os.path.join(index_dir, name), vbyte_encode(arr).tobytes())

        self._write_blob(os.path.join(index_dir, _POSTINGS_D), self._docid_buf.tobytes())
        self._write_blob(os.path.join(index_dir, _TF_NIB), self._tf_packed.tobytes())
        exc_gaps = np.diff(self._tf_exc_idx, prepend=0) if self._tf_exc_idx.size else self._tf_exc_idx
        self._write_blob(os.path.join(index_dir, _TF_EXC_I), vbyte_encode(exc_gaps).tobytes())
        self._write_blob(os.path.join(index_dir, _TF_EXC_V), vbyte_encode(self._tf_exc_val).tobytes())

        if self.store_positions:
            self._write_blob(os.path.join(index_dir, _POS_LEN),
                             vbyte_encode(np.diff(self._pos_off)).tobytes())
            self._write_blob(os.path.join(index_dir, _POSITIONS), self._pos_buf.tobytes())

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """rebuild from index_dir alone, fresh process"""
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
        index.store_doc_ids = bool(meta.get("store_doc_ids", True))
        index.doc_ids = read_text(_DOCIDS) if index.store_doc_ids else []
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
