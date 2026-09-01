"""
submission/indexer.py — the inverted index (assignment Section 4.1).

Columnar arrays, not a dict-of-dicts: at 16.3M postings the obvious
Dict[str, Dict[str, int]] shape costs several GB of pure interpreter overhead,
which won't fit the 8GB grading machine.

    terms[t]           term string, sorted ascending
    df[t], cf[t]        document / collection frequency
    docid_off[t:t+2]    byte slice of this term's docids in `_docid_buf`
    term_start[t]       index of this term's first posting; also its nibble
                        offset into the packed tf array
    doc_len[d]           document length in tokens
    doc_ids[d]           external doc_id string for internal id d

Postings are delta+VByte encoded (submission/_codecs.py), the whole
collection in one vectorised call rather than per-term. save()/load() round-
trip through plain files with no pickling and no shared state between
processes, and store nothing retrieve() doesn't need -- raw document text is
never persisted.
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

# Below this many documents, splitting across processes costs more (spawn +
# pickle + merge) than it could ever save. The toy corpus (20 docs) and small
# test fixtures always take the serial path; this is not a graded corpus size.
_PARALLEL_MIN_DOCS = 20_000

# Optional C++ kernel for tokenising and posting emission (see
# submission/_fastbuild.pyx). Imported behind try/except: without it the build
# falls back to the pure-Python path and produces the identical index, just
# more slowly.
try:
    from submission import _fastbuild as _FASTBUILD
except ImportError:  # pragma: no cover - exercised by the fallback test
    _FASTBUILD = None

FORMAT_VERSION = 3   # 3: index files zlib-compressed on disk

# Every index file is deflated on disk. Level 4 from a measured curve, not the
# default: compression is charged against build time (level 1 -> 22.20MB/0.40s,
# level 4 -> 21.57MB/0.60s, level 9 -> 21.22MB/7.27s -- not worth it),
# decompression in load() isn't graded and is ~0.05s regardless of level.
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

    Kept as a module-level function under this name for interface
    compatibility. The configurable chain used by the index itself lives in
    submission/_analysis.py; this delegates to it with default settings so
    the two can never drift.
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


def _split_byte_ranges(corpus_path: str, n_workers: int):
    """Newline-aligned byte offsets splitting the file into `n_workers`
    roughly-equal pieces, plus each piece's starting internal doc id.

    One sequential pass over line boundaries, not the document content --
    the file is never materialised in the main process, preserving
    build_from_jsonl()'s memory-safe streaming property. Internal doc ids
    must equal file line order (many things assume this), so each worker
    needs to know exactly how many non-blank lines precede its range.
    """
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
                f.readline()  # consume the partial line; land on a boundary
                end = f.tell()
            # Count non-blank lines in [start, end) to get the NEXT worker's
            # doc_id_start -- cheap relative to JSON-decoding the same bytes.
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
    """Best-effort process-visible CPU count, capped defensively.

    os.cpu_count() reports the HOST's total CPU count, even inside a
    container whose actual allocation is far smaller -- the grading machine
    is documented as 4 cores (assignment1.tex Sec. 5), but a real grading
    run's error log showed OpenBLAS sizing ITS OWN thread pool to 28 (see
    _detected_worker_count()'s caller): the host's count, not the
    container's. os.sched_getaffinity(0), where available, respects the
    process's actual CPU affinity mask, which container CPU limits
    typically DO set correctly; os.cpu_count() is the fallback where
    sched_getaffinity doesn't exist (e.g. macOS has no such call). Clamped
    regardless: this project's own build-time measurements
    (notes/findings.md) show throughput saturating well under 8 workers, so
    trusting a much larger raw count buys nothing and only raises the odds
    of hitting a process/thread ceiling like the one below.
    """
    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, 8))


def _parallel_build_worker(args):
    """Runs in a separate process: tokenise/stem/intern one byte range of the
    corpus with its own local Builder, return everything needed to merge.

    Module-level (not a method) because multiprocessing pickles the callable
    by reference -- a bound method or closure can't cross the process
    boundary. Returns plain picklable types only (lists, bytes, ndarrays),
    never an InvertedIndex or a Cython object.
    """
    (corpus_path, byte_start, byte_end, doc_id_start, min_token_len,
     max_token_len, stem_tokens, prefix_tokens) = args
    from submission import _fastbuild as fb  # re-imported: fresh process

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
        # int32 for docs/tfs, matching finish_sorted()'s normal-case dtype
        # (_fastbuild.pyx) -- np.concatenate upcasts its whole result to
        # int64 if even one chunk in the list doesn't match, which would
        # silently defeat _merge_parallel_results()'s int32 path whenever a
        # worker's byte range happened to be empty.
        return doc_ids, doc_lens, [], np.zeros(0, dtype=np.int32), \
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)

    # Identity order: local postings come back in first-seen (unsorted) term
    # order. The merge step below does the real sort once, globally, rather
    # than sorting per worker and re-sorting again after the union.
    identity = np.arange(len(terms_unsorted), dtype=np.int32)
    docs_arr, tfs_arr, df = builder.finish_sorted(identity)
    return doc_ids, doc_lens, terms_unsorted, docs_arr, tfs_arr, df


class InvertedIndex:
    """Inverted index with delta+VByte postings and on-disk persistence."""

    def __init__(self, config: Optional[AnalysisConfig] = None,
                 store_positions: bool = False):
        self.config = config or AnalysisConfig()
        # Positions cost roughly one VByte per token occurrence and are only
        # needed by proximity/phrase scoring, so they are opt-in: an index built
        # without them is byte-for-byte what it was before this existed.
        self.store_positions = store_positions
        # A forward (doc -> terms) index, needed only by pseudo-relevance
        # feedback (submission/rm3.py). Off by default: it is real extra disk
        # and build time, and the plain BM25 strategy never needs it.
        self.store_forward = False
        self.forward = None  # submission._forward.ForwardIndex, once built/loaded
        self._prefix_tokens = -1
        # The pseudo-title index shares the main index's document order, so it
        # does not need its own copy of the external doc-id strings (1.5MB).
        self.store_doc_ids = True
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

    def build_from_jsonl(self, corpus_path: str, prefix_tokens: int = -1) -> None:
        """Build by streaming a corpus.jsonl file -- the memory-safe path.

        `prefix_tokens >= 0` indexes only each document's first N tokens, which
        is how the pseudo-title field is built (see submission/retrieve.py).
        """
        self._prefix_tokens = prefix_tokens
        self._build(_iter_jsonl(corpus_path))

    def _build(self, docs: Iterable[Tuple[str, str]]) -> None:
        if self.store_positions:
            return self._build_positional(docs)
        # The forward index is built inside _finalise_postings(), which the C++
        # fast path bypasses. Route to Python whenever it's requested -- costs
        # nothing in practice since the only chain that wants one (stemmed RM3)
        # already declines the C++ builder anyway.
        if self.store_forward:
            return self._build_counts(docs)
        if _FASTBUILD is not None and _FASTBUILD.Builder.supports(self.config):
            return self._build_counts_fast(docs)
        return self._build_counts(docs)

    def build_from_jsonl_parallel(self, corpus_path: str, prefix_tokens: int = -1,
                                  n_workers: Optional[int] = None,
                                  min_docs: int = _PARALLEL_MIN_DOCS) -> bool:
        """Split the corpus across up to `n_workers` processes for the
        per-document tokenise/stem/intern phase. Returns True if it ran (the
        parallel path was applicable), False if the caller should fall back
        to `build_from_jsonl()` -- e.g. the C++ builder can't reproduce this
        analysis chain, or the corpus is too small for splitting to pay for
        its own overhead.

        Only tokenisation is parallel; postings assembly, encoding and disk
        writes stay serial regardless (a global, whole-collection step by
        nature), so this is bounded by Amdahl's law, not a 4x build-time cut.
        """
        self._prefix_tokens = prefix_tokens
        if _FASTBUILD is None or not _FASTBUILD.Builder.supports(self.config):
            return False
        if self.store_positions or self.store_forward:
            return False  # not supported by the fast path at all, parallel or not

        n_workers = n_workers or _detected_worker_count()
        size = os.path.getsize(corpus_path)
        approx_docs = size // 200  # ~200 bytes/doc average on this corpus; a
        # cheap upper-bound estimate to gate on, not an exact count -- getting
        # this wrong only costs "did we parallelise a corpus that was too
        # small to benefit", never correctness.
        if n_workers < 2 or approx_docs < min_docs:
            return False

        ranges = _split_byte_ranges(corpus_path, n_workers)
        tasks = [(corpus_path, start, end, doc_id_start,
                 self.config.min_token_len, self.config.max_token_len,
                 self.config.stemmer == "porter", prefix_tokens)
                for start, end, doc_id_start in ranges]

        # Each spawned worker re-imports numpy fresh (spawn re-imports
        # everything), and numpy's BLAS backend sizes ITS OWN internal
        # thread pool from the visible CPU count too -- with n_workers
        # processes each also spinning up that many BLAS threads, a real
        # grading run hit `pthread_create failed ... Resource temporarily
        # unavailable` inside OpenBLAS's own init, partway through numpy's
        # import, before any of this project's code had run at all.
        # Multiprocessing is already the parallelism here; BLAS threading
        # inside each worker is pure redundant oversubscription on top of
        # it. Setting these here, in the parent, propagates through
        # process creation to each child's environment (spawn inherits the
        # environment at Process-creation time) without affecting the
        # BLAS thread pool this process itself already initialised when it
        # first imported numpy -- that's sized once, at import time, so a
        # later env var change here has no retroactive effect on it.
        for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[_var] = "1"

        ctx = multiprocessing.get_context("spawn")  # portable; fork is unsafe
        # to rely on once the parent has imported a compiled extension.
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_parallel_build_worker, tasks)

        self._merge_parallel_results(results)
        return True

    def _merge_parallel_results(self, results) -> None:
        """Union each worker's local vocabulary into one global sorted one,
        remap postings, and concatenate per term in worker-rank order.

        Concatenation (not a k-way merge) is correct because each worker's
        doc-id range is contiguous, non-overlapping, and increases with
        worker rank -- worker 0's documents all precede worker 1's, etc. --
        so a term's postings are already doc-id-ascending within one worker
        and stay ascending when workers are joined in rank order.
        """
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

        # Per worker: local term_start offsets (into its own docs/tfs arrays)
        # and local-index -> global-index, so a global term's contribution
        # from this worker can be sliced out directly.
        per_worker = []
        for _di, _dl, terms, docs_arr, tfs_arr, df in results:
            local_start = self._term_starts(df) if df.size else np.zeros(0, dtype=np.int64)
            g_of_l = np.fromiter((global_index[t] for t in terms), dtype=np.int64,
                                 count=len(terms))
            per_worker.append((docs_arr, tfs_arr, df, local_start, g_of_l))

        docs_chunks: List[np.ndarray] = []
        tfs_chunks: List[np.ndarray] = []
        df_final = np.zeros(n_global, dtype=np.int64)
        # Global-to-local lookup per worker (-1 = this worker never saw the
        # term), built once so the term loop below is O(1) per worker.
        inv_maps = []
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

        # int32, matching finish_sorted()'s own return dtype (_fastbuild.pyx):
        # docs_chunks/tfs_chunks are views into each worker's int32 arrays, so
        # concatenating them is already int32 in the common case; only the
        # dtype= here matters for the (unreachable in practice -- all_terms
        # non-empty implies at least one chunk) empty fallback, kept
        # consistent so concatenate never has to upcast the real case to
        # match a mismatched empty-array dtype.
        docs_arr = (np.concatenate(docs_chunks) if docs_chunks
                   else np.zeros(0, dtype=np.int32))
        tfs_arr = (np.concatenate(tfs_chunks) if tfs_chunks
                  else np.zeros(0, dtype=np.int32))

        self.terms = sorted_terms
        self.term_lookup = global_index
        self.df = df_final
        self.cf = np.zeros(n_global, dtype=np.int64)
        if docs_arr.size:
            # reduceat's out= safely upcasts int32 -> int64 during
            # accumulation; casting the input first would just allocate a
            # redundant total-postings-sized copy (F51, notes/findings.md).
            np.add.reduceat(tfs_arr, self._term_starts(df_final), out=self.cf)
        self._encode_postings(docs_arr, tfs_arr)

    def _build_counts_fast(self, docs: Iterable[Tuple[str, str]]) -> None:
        """Same index as `_build_counts`, with tokenising and posting emission
        done in C++ (see submission/_fastbuild.pyx).

        Used only when the analysis chain is the default one the kernel
        reproduces exactly; any other configuration falls back to Python rather
        than risking a silently different index.
        """
        builder = _FASTBUILD.Builder(self.config.min_token_len, self.config.max_token_len,
                                     self.config.stemmer == "porter")
        doc_ids: List[str] = []
        doc_lens: List[int] = []
        for internal_id, (ext_id, text) in enumerate(docs):
            doc_ids.append(ext_id)
            # str.lower() stays in Python: it is already C-speed and applies the
            # full Unicode case mapping the byte scanner cannot.
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
        # docs_arr/tfs_arr come back from finish_sorted() as int32 already
        # (submission/_fastbuild.pyx); reduceat's out= safely upcasts during
        # accumulation, so casting the input up first would only allocate a
        # redundant total-postings-sized int64 copy for no benefit (F51).
        np.add.reduceat(tfs_arr, self._term_starts(df), out=self.cf)
        self._encode_postings(docs_arr, tfs_arr)

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
        # int32, not int64: a doc-id delta is bounded by self.N, nowhere near
        # int32's ~2.1 billion ceiling even at the "larger collection" scale
        # the held-out evaluation uses (assignment1.tex Sec. 3). This is a
        # total-postings-sized array (16.3M+ at the dev corpus), so this is
        # the single biggest lever in this function (see F51, notes/findings.md).
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
        """Build with term positions retained, for proximity/phrase scoring.

        Emits one (term, doc, position) triple per token occurrence and
        recovers tf by grouping, so a posting's tf is exactly how many
        positions it owns -- no separate offset table needed for the
        positions file.
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
            if self._prefix_tokens >= 0:
                tokens = tokens[:self._prefix_tokens]
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

        if self.store_forward:
            # Built here, before the term-major sort below reorders these
            # triples away from a form ForwardIndex.build() can re-sort itself.
            from submission._forward import ForwardIndex
            self.forward = ForwardIndex.build(docs_arr, term_ids, tfs_arr, self.N)

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
        if self.store_forward:
            # No postings at all (every document tokenised to nothing, or the
            # corpus was empty), so every document trivially has zero terms.
            from submission._forward import ForwardIndex
            self.forward = ForwardIndex.build(
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64), self.N)

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
            "store_doc_ids": self.store_doc_ids,
            "store_forward": self.store_forward,
            "n_tf_exceptions": int(self._tf_exc_idx.size),
        }
        with open(os.path.join(index_dir, _META), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # The tokenizer emits [a-z0-9]+ only, and external doc_ids in this
        # collection carry no newlines, so newline framing is unambiguous.
        self._write_blob(os.path.join(index_dir, _TERMS),
                         "\n".join(self.terms).encode("utf-8"))
        if self.store_doc_ids:
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

        if self.store_forward and self.forward is not None:
            self.forward.save(index_dir)

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

        index.store_forward = bool(meta.get("store_forward", False))
        if index.store_forward:
            from submission._forward import ForwardIndex
            index.forward = ForwardIndex.load(index_dir)
        return index
