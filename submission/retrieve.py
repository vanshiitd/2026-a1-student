"""
submission/retrieve.py -- required entrypoint

    build_index(corpus_path, index_dir) -> None
    load_index(index_dir) -> None
    retrieve(query, k=10) -> List[Tuple[str, float]]

thin on purpose, actual indexing in indexer.py, scoring in bm25.py /
boolean_vsm.py.

plain bm25 + pseudo title field, no feedback pass. also tried RM3 (feedback)
which scored better on dev but lost on the actual held out topics (Day 4,
nDCG@10 0.1714 vs class avg ~0.20), so reverted back to plain bm25 and
removed the rm3 code, see report for the full story
"""
import os
import sys
from typing import List, Optional, Tuple

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex

# k1=4.5, b=0.60 (textbook is 1.2/0.75), tuned via grid search + CV
BM25_K1 = 4.5
BM25_B = 0.60

# pseudo title field: first TITLE_WIDTH tokens indexed again as a second
# field w/ small weight. no real title/abstract boundary so this is an
# approximation but it works
TITLE_WIDTH = 10
TITLE_WEIGHT = 0.10
_MAIN_DIR = "main"
_TITLE_DIR = "title"

# set by load_index(), read by retrieve(). build_index runs in its own
# process so nothing here carries over from it
_INDEX: Optional[InvertedIndex] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    """streams the corpus, don't materialise it all in memory"""
    os.makedirs(index_dir, exist_ok=True)
    index = InvertedIndex()
    _build(index, corpus_path)
    index.save(os.path.join(index_dir, _MAIN_DIR))

    title = InvertedIndex()
    title.store_doc_ids = False  # shares main's doc order
    _build(title, corpus_path, prefix_tokens=TITLE_WIDTH)
    title.save(os.path.join(index_dir, _TITLE_DIR))


def _build(index: InvertedIndex, corpus_path: str, prefix_tokens: int = -1) -> None:
    """tries parallel build first, falls back to serial if it declines"""
    if not index.build_from_jsonl_parallel(corpus_path, prefix_tokens=prefix_tokens):
        index.build_from_jsonl(corpus_path, prefix_tokens=prefix_tokens)


def load_index(index_dir: str) -> None:
    global _INDEX
    _INDEX = InvertedIndex.load(os.path.join(index_dir, _MAIN_DIR))
    title = InvertedIndex.load(os.path.join(index_dir, _TITLE_DIR))
    bm25.build(_INDEX, title_index=title, title_weight=TITLE_WEIGHT)
    boolean_vsm.build(_INDEX)  # required regardless


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    if _INDEX is None:
        raise RuntimeError(
            "retrieve() called before load_index(); the harness always "
            "calls build_index(corpus_path, index_dir) and then "
            "load_index(index_dir) — in that order, in two separate "
            "processes — before any retrieve() calls. If you're testing "
            "manually, do the same."
        )

    try:
        results = bm25.score(query, k, k1=BM25_K1, b=BM25_B)
    except Exception as exc:  # noqa: BLE001 - deliberate boundary guard
        # one bad query shouldn't kill the whole run
        print(f"WARNING: retrieve() failed for query {query!r}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return []
    return _finalise(results, k)


def _finalise(results: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    """dedup by doc_id, keep best rank, truncate to k"""
    seen = set()
    out: List[Tuple[str, float]] = []
    for doc_id, score in results:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append((doc_id, float(score)))
        if len(out) >= k:
            break
    return out
