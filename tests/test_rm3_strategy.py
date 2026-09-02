"""
tests/test_rm3_strategy.py — the rm3_stemmed strategy, and the switch that
selects between it and the plain BM25 ("shipped") alternative.

submission/retrieve.py's ACTIVE_STRATEGY is a one-line edit (see its
module-level comment) -- rm3_stemmed shipped for the 28 Aug initial
submission and through Day 4 of the competition round, then switched back to
shipped (plain BM25) after rm3_stemmed placed near the bottom of the class on
the private held-out topics; see the comment above ACTIVE_STRATEGY for the
numbers. The risk either way: a future change to
submission/rm3.py or submission/_forward.py could silently break the
non-active strategy, and nothing would notice until someone actually flips
the switch, under time pressure, with the private held-out topics on the line.

These tests exercise the rm3_stemmed path continuously regardless of which
strategy is currently committed, and pin down that ACTIVE_STRATEGY matches
what was actually intended to ship, so an accidental edit to that one line
fails CI rather than shipping silently.

Cross-process persistence itself (build_index and load_index/retrieve running
as genuinely separate processes) is already covered generally by
tests/test_forward_index.py's save/load round-trip tests and
tests/test_interface_conformance.py's subprocess harness test for the default
strategy; this file focuses on what is specific to rm3_stemmed -- correct
dispatch, and the scorer's own robustness.
"""
import os
import tempfile

import pytest

from submission import retrieve as entry

TOY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "toy")
CORPUS_PATH = os.path.join(TOY_DIR, "corpus.jsonl")


def test_committed_default_is_the_intended_strategy():
    """A guard against ACTIVE_STRATEGY drifting by accident, not a fixed
    opinion about which strategy should be active. Update the expected value
    alongside any deliberate switch -- don't just delete this."""
    assert entry.ACTIVE_STRATEGY == "shipped", (
        "ACTIVE_STRATEGY is not 'shipped' (switched back from rm3_stemmed "
        "after Day 4 of the competition round -- see the comment above "
        "ACTIVE_STRATEGY in submission/retrieve.py). If this is a deliberate "
        "switch, update this test's expected value alongside it."
    )


@pytest.fixture
def rm3_index_dir(monkeypatch, tmp_path):
    """Build and load a real rm3_stemmed index from the toy corpus, without
    touching the committed ACTIVE_STRATEGY value on disk."""
    monkeypatch.setattr(entry, "ACTIVE_STRATEGY", "rm3_stemmed")
    index_dir = str(tmp_path / "index")
    entry.build_index(CORPUS_PATH, index_dir)
    entry.load_index(index_dir)
    yield index_dir
    # load_index() leaves module state bound to whichever strategy was
    # active; restore it so later tests in the same process see the default.
    monkeypatch.undo()


def test_rm3_strategy_builds_and_loads_from_the_toy_corpus(rm3_index_dir):
    written = [os.path.join(dp, fn) for dp, _dn, fns in os.walk(rm3_index_dir) for fn in fns]
    assert written, "build_index() wrote nothing under the rm3_stemmed strategy"


def test_rm3_strategy_produces_well_formed_results(monkeypatch, rm3_index_dir):
    monkeypatch.setattr(entry, "ACTIVE_STRATEGY", "rm3_stemmed")
    from harness.trec_io import read_corpus, read_queries
    queries = read_queries(os.path.join(TOY_DIR, "queries_dev.tsv"))
    valid_doc_ids = {doc_id for doc_id, _text in read_corpus(CORPUS_PATH)}

    for qid, text in queries:
        results = entry.retrieve(text, 5)
        assert isinstance(results, list)
        assert len(results) <= 5, f"qid={qid} returned more than k results"
        doc_ids = [d for d, _s in results]
        assert len(doc_ids) == len(set(doc_ids)), f"qid={qid} returned a duplicate doc_id"
        scores = [s for _d, s in results]
        assert scores == sorted(scores, reverse=True), f"qid={qid} not sorted descending"
        for doc_id, score in results:
            assert doc_id in valid_doc_ids, f"qid={qid} returned unknown doc_id {doc_id!r}"
            assert isinstance(score, (int, float))


def test_rm3_strategy_is_deterministic(monkeypatch, rm3_index_dir):
    monkeypatch.setattr(entry, "ACTIVE_STRATEGY", "rm3_stemmed")
    a = entry.retrieve("cat", 10)
    b = entry.retrieve("cat", 10)
    assert a == b


ADVERSARIAL_QUERIES = [
    "", "   \t\n  ", "!!! ??? ---", "你好 مرحبا αβγ", "\x00\x01 null bytes",
    "a" * 50_000, " ".join(["cat"] * 5_000), "zzzqqq wwwvvv", "CAT dog!!! 100%",
]


@pytest.mark.parametrize("query", ADVERSARIAL_QUERIES)
def test_rm3_strategy_survives_adversarial_queries(monkeypatch, rm3_index_dir, query):
    """Mirrors tests/test_ranking_components.py's adversarial suite for the
    shipped strategy -- the two-tier fallback in retrieve() protects both
    strategies alike, but only if rm3.score() itself does not raise on inputs
    that legitimately produce zero query terms or zero feedback documents."""
    monkeypatch.setattr(entry, "ACTIVE_STRATEGY", "rm3_stemmed")
    results = entry.retrieve(query, 10)
    assert isinstance(results, list)
    assert len(results) <= 10
    doc_ids = [d for d, _s in results]
    assert len(doc_ids) == len(set(doc_ids))


def test_rm3_score_respects_k(rm3_index_dir):
    from submission import rm3
    assert rm3.score("cat", 0) == []
    assert len(rm3.score("cat", 2)) <= 2
    assert len(rm3.score("cat", 1000)) <= 20  # toy corpus has 20 docs


def test_rm3_raises_a_clear_error_before_build():
    from submission import rm3
    saved = (rm3._BODY, rm3._TITLE, rm3._FORWARD)
    rm3._BODY = rm3._TITLE = rm3._FORWARD = None
    try:
        with pytest.raises(RuntimeError, match="build"):
            rm3.score("cat", 10)
    finally:
        rm3._BODY, rm3._TITLE, rm3._FORWARD = saved


def test_rm3_build_rejects_a_body_index_without_a_forward_index():
    from submission._analysis import AnalysisConfig
    from submission import rm3
    from submission.indexer import InvertedIndex

    body = InvertedIndex(AnalysisConfig(stemmer="porter"))
    body.build([("d1", "alpha beta")])  # store_forward left False
    title = InvertedIndex(AnalysisConfig(stemmer="porter"))
    title.build([("d1", "alpha")])

    with pytest.raises(RuntimeError, match="store_forward"):
        rm3.build(body, title)


def test_rm3_build_rejects_mismatched_body_and_title_document_counts():
    from submission._analysis import AnalysisConfig
    from submission import rm3
    from submission.indexer import InvertedIndex

    cfg = AnalysisConfig(stemmer="porter")
    body = InvertedIndex(cfg)
    body.store_forward = True
    body.build([("d1", "alpha beta"), ("d2", "gamma delta")])
    title = InvertedIndex(cfg)
    title.build([("d1", "alpha")])  # only 1 document, body has 2

    with pytest.raises(RuntimeError, match="document count"):
        rm3.build(body, title)
