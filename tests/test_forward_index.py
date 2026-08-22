"""
tests/test_forward_index.py — correctness of submission/_forward.py.

The forward (doc -> terms) index exists only for pseudo-relevance feedback
(submission/rm3.py) and is off by default. Every check here is against a
naive per-document recount from submission._analysis.analyze(), not against
the implementation's own output — the same discipline as
tests/test_ranking_components.py.

Documents can legitimately be empty (this corpus has 8 -- notes/findings.md
F2), unlike terms which always have df >= 1. That asymmetry is exactly where
the implementation had a real bug during development: np.add.reduceat, given
an out-of-bounds start index for a trailing empty document, silently
corrupted the PRECEDING document's byte range rather than raising. Several
cases below exist specifically to keep that bug from coming back.
"""
from collections import Counter

import numpy as np
import pytest

from submission._analysis import AnalysisConfig, analyze
from submission.indexer import InvertedIndex


def _naive_forward(corpus, config):
    """doc index -> {term_id: tf}, computed independently of ForwardIndex."""
    out = {}
    ix = InvertedIndex(config)
    ix.build(corpus)
    for did, (_doc_id, text) in enumerate(corpus):
        counts = Counter(analyze(text, config))
        out[did] = {ix.term_id(t): tf for t, tf in counts.items()}
    return out


def _build_with_forward(corpus, config=None):
    cfg = config or AnalysisConfig()
    ix = InvertedIndex(cfg)
    ix.store_forward = True
    ix.build(corpus)
    return ix


def _assert_matches_naive(ix, corpus, config):
    expected = _naive_forward(corpus, config)
    for did in range(ix.N):
        term_ids, tfs = ix.forward.terms_and_tfs(did)
        got = dict(zip(term_ids.tolist(), tfs.tolist()))
        assert got == expected[did], f"doc {did} ({corpus[did][0]!r}): {got} != {expected[did]}"
        assert list(term_ids) == sorted(term_ids), f"doc {did}: term ids not ascending"


CORPUS_WITH_EMPTIES = [
    ("d1", "alpha beta gamma"),
    ("d2", "alpha alpha delta"),
    ("d3", "beta gamma gamma gamma delta epsilon"),
    ("d4", ""),                       # empty, interior
    ("d5", "alpha beta gamma delta epsilon zeta"),
    ("d6", "!!! ???"),                # tokenises to nothing, TRAILING
]


def test_matches_naive_recount_with_interior_and_trailing_empty_docs():
    ix = _build_with_forward(CORPUS_WITH_EMPTIES)
    _assert_matches_naive(ix, CORPUS_WITH_EMPTIES, ix.config)


def test_multiple_trailing_empty_documents():
    """The exact shape that exposed the reduceat boundary bug: several empty
    documents in a row at the end of the corpus."""
    corpus = [("x1", "alpha beta"), ("x2", "gamma"), ("x3", ""),
              ("x4", "!!!"), ("x5", "   ")]
    ix = _build_with_forward(corpus)
    _assert_matches_naive(ix, corpus, ix.config)


def test_all_documents_empty():
    corpus = [("a", ""), ("b", "   "), ("c", "!!!")]
    ix = _build_with_forward(corpus)
    assert ix.N == 3
    for did in range(ix.N):
        term_ids, tfs = ix.forward.terms_and_tfs(did)
        assert term_ids.size == 0 and tfs.size == 0


def test_empty_corpus():
    ix = _build_with_forward([])
    assert ix.N == 0


def test_stemmed_config():
    """Production usage: the forward index always pairs with a stemmed body
    index (submission/rm3.py), never the default chain."""
    corpus = [("d1", "running runners ran"), ("d2", "the quick fox"), ("d3", "")]
    cfg = AnalysisConfig(stemmer="porter")
    ix = _build_with_forward(corpus, cfg)
    _assert_matches_naive(ix, corpus, cfg)


@pytest.mark.parametrize("seed", range(3))
def test_random_corpus_with_scattered_empty_documents(seed):
    rng = np.random.default_rng(seed)
    vocab = [f"t{i}" for i in range(60)]
    corpus = []
    for i in range(400):
        n = int(rng.integers(0, 40))  # sometimes 0 -> an empty document
        words = rng.choice(vocab, size=n, replace=True) if n else []
        corpus.append((f"doc{i}", " ".join(words)))
    ix = _build_with_forward(corpus)
    _assert_matches_naive(ix, corpus, ix.config)


def test_save_load_round_trip_is_exact():
    import tempfile
    ix = _build_with_forward(CORPUS_WITH_EMPTIES)
    with tempfile.TemporaryDirectory() as d:
        ix.save(d)
        reloaded = InvertedIndex.load(d)
    assert reloaded.store_forward
    for did in range(ix.N):
        a = ix.forward.terms_and_tfs(did)
        b = reloaded.forward.terms_and_tfs(did)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_not_built_when_store_forward_is_false():
    ix = InvertedIndex(AnalysisConfig())
    ix.build(CORPUS_WITH_EMPTIES)
    assert ix.forward is None


def test_forward_index_forces_the_python_build_path():
    """store_forward must not silently fall through the C++ builder, which
    never populates it -- this was the first bug found during development."""
    import submission.indexer as ixmod
    if ixmod._FASTBUILD is None:
        pytest.skip("C++ extension not built; nothing to distinguish here")
    ix = InvertedIndex(AnalysisConfig())  # default config: C++ path is eligible
    ix.store_forward = True
    ix.build(CORPUS_WITH_EMPTIES)
    assert ix.forward is not None, "forward index was not built"
    _assert_matches_naive(ix, CORPUS_WITH_EMPTIES, ix.config)
