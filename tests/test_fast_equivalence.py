"""
tests/test_fast_equivalence.py — the C extension must be an optimisation only.

submission/_fast.pyx exists purely for speed. These tests pin down the two
properties that make that acceptable:

  1. **Bit-identical results.** Not "similar rankings", not "close scores" --
     the same float64 values. The kernel performs the same operations in the
     same order as the NumPy path and is compiled without -ffast-math, so exact
     equality is achievable and therefore worth asserting. Anything weaker would
     let a scoring discrepancy hide behind a rounding excuse.

  2. **A working fallback.** If the extension did not compile, the submission
     must still produce correct results. Speed must never become a correctness
     dependency at a graded boundary.

If the extension is not built, the equivalence tests skip and the fallback test
still runs.
"""
import numpy as np
import pytest

from submission import bm25, boolean_vsm
from submission._codecs import encode_sorted_ids, vbyte_encode
from submission.indexer import InvertedIndex

fast = pytest.importorskip("submission._fast",
                           reason="C extension not built; pure-Python path in use")

CORPUS = [
    ("d1", "alpha beta gamma"),
    ("d2", "alpha alpha delta"),
    ("d3", "beta gamma gamma gamma delta epsilon"),
    ("d4", "epsilon"),
    ("d5", "alpha beta gamma delta epsilon zeta"),
]


@pytest.fixture
def index():
    ix = InvertedIndex()
    ix.build(CORPUS)
    bm25.build(ix)
    boolean_vsm.build(ix)
    return ix


@pytest.mark.parametrize("query", [
    "alpha", "beta gamma", "alpha beta gamma delta epsilon zeta",
    "gamma gamma", "missing", "alpha missing", "", "!!!",
])
@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (4.5, 0.60), (0.3, 0.0), (12.0, 1.0)])
def test_c_and_python_paths_are_bit_identical(index, query, k1, b):
    """Same doc order AND the same float64 bits, across parameter extremes."""
    from submission import _traverse
    fast_results = bm25._score_fast(index, query, 10, k1, b)
    numpy_results = _traverse.score_single(index, query, "bm25", 10, k1=k1, b=b)

    assert [d for d, _ in fast_results] == [d for d, _ in numpy_results]
    for (_da, sa), (_db, sb) in zip(fast_results, numpy_results):
        assert sa == sb, f"score differs: {sa!r} vs {sb!r} (query={query!r})"


def test_decode_postings_matches_the_numpy_codec(index):
    """The C decoder must agree with submission/_codecs.py on every term."""
    for term in index.terms:
        tid = index.term_id(term)
        count = int(index.df[tid])
        py_docs, py_tfs = index.postings_by_id(tid)
        c_docs, c_tfs = fast.decode_postings(
            index._docid_buf[index._docid_off[tid]:index._docid_off[tid + 1]],
            index._tf_buf[index._tf_off[tid]:index._tf_off[tid + 1]],
            count,
        )
        np.testing.assert_array_equal(py_docs, c_docs)
        np.testing.assert_array_equal(py_tfs, c_tfs)


@pytest.mark.parametrize("seed", range(5))
def test_decode_postings_on_random_wide_gaps(seed):
    """Exercise multi-byte VByte values, which a small toy corpus never reaches."""
    rng = np.random.default_rng(seed)
    docs = np.unique(rng.integers(0, 5_000_000, size=3000)).astype(np.int64)
    tfs = rng.integers(1, 40_000, size=docs.size).astype(np.int64)
    c_docs, c_tfs = fast.decode_postings(
        encode_sorted_ids(docs), vbyte_encode(tfs), docs.size)
    np.testing.assert_array_equal(docs, c_docs)
    np.testing.assert_array_equal(tfs, c_tfs)


def test_fallback_produces_identical_results_when_extension_is_absent(index, monkeypatch):
    """Simulate a submission where the extension failed to compile."""
    with_fast = bm25.score("beta gamma", 10, k1=4.5, b=0.60)
    monkeypatch.setattr(bm25, "HAVE_FAST", False)
    without_fast = bm25.score("beta gamma", 10, k1=4.5, b=0.60)
    assert with_fast == without_fast, "fallback path diverges from the C path"
    assert without_fast, "fallback returned nothing"


def test_extension_does_not_mutate_the_index(index):
    """The kernel writes into caller-owned score buffers only."""
    before = (index._docid_buf.copy(), index._tf_buf.copy(), index.doc_len.copy())
    bm25.score("alpha beta gamma", 10, k1=4.5, b=0.60)
    np.testing.assert_array_equal(before[0], index._docid_buf)
    np.testing.assert_array_equal(before[1], index._tf_buf)
    np.testing.assert_array_equal(before[2], index.doc_len)


def _synthetic_corpus(n_docs=1500, seed=0):
    """Corpus large and varied enough to expose float divergence.

    The small fixture above did NOT catch a real bug: with -O3 the compiler
    fused `a*b + c` into a single FMA instruction, rounding once instead of
    twice, and the kernel diverged from NumPy on the full 171K-document corpus
    while passing every small-fixture test. Accumulating many contributions per
    document with widely varying lengths is what surfaces it, so the regression
    guard has to be built that way.
    """
    rng = np.random.default_rng(seed)
    vocab = [f"t{i}" for i in range(400)]
    corpus = []
    for i in range(n_docs):
        length = int(rng.integers(3, 600))          # deliberately wide spread
        words = rng.choice(vocab, size=length, replace=True)
        corpus.append((f"s{i}", " ".join(words)))
    return corpus


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (4.5, 0.60), (9.9, 0.35)])
def test_bit_identical_on_a_corpus_large_enough_to_expose_fp_contraction(k1, b):
    from submission import _traverse
    ix = InvertedIndex()
    ix.build(_synthetic_corpus())
    bm25.build(ix)
    rng = np.random.default_rng(7)
    vocab = [f"t{i}" for i in range(400)]

    for _ in range(25):
        query = " ".join(rng.choice(vocab, size=int(rng.integers(2, 15)), replace=True))
        fast_r = bm25._score_fast(ix, query, 10, k1, b)
        numpy_r = _traverse.score_single(ix, query, "bm25", 10, k1=k1, b=b)
        assert [d for d, _ in fast_r] == [d for d, _ in numpy_r], f"ranking differs for {query!r}"
        for (_a, sa), (_b2, sb) in zip(fast_r, numpy_r):
            assert sa == sb, (
                f"score bits differ ({sa!r} vs {sb!r}) for {query!r} at k1={k1}, b={b}. "
                "Check the compiler is not contracting a*b+c into an FMA "
                "(-ffp-contract=off in setup.py)."
            )


# ---------------------------------------------------------------------------
# Build-side kernel (submission/_fastbuild.pyx)
# ---------------------------------------------------------------------------

def _build_both(corpus, config=None):
    """Build the same corpus with and without the C++ kernel."""
    import submission.indexer as ixmod
    from submission._analysis import AnalysisConfig
    cfg = config or AnalysisConfig()

    saved = ixmod._FASTBUILD
    ixmod._FASTBUILD = None
    py = InvertedIndex(cfg); py.build(corpus)
    ixmod._FASTBUILD = saved
    c = InvertedIndex(cfg); c.build(corpus)
    return py, c


def _assert_identical(py, c):
    assert py.terms == c.terms
    assert py.doc_ids == c.doc_ids
    assert (py.N, py.total_tokens, py.avg_doc_len) == (c.N, c.total_tokens, c.avg_doc_len)
    for name in ("doc_len", "df", "cf", "_docid_buf", "_tf_buf", "_docid_off", "_tf_off"):
        np.testing.assert_array_equal(getattr(py, name), getattr(c, name),
                                      err_msg=f"{name} differs between build paths")


def test_cpp_builder_produces_a_byte_identical_index():
    pytest.importorskip("submission._fastbuild")
    _assert_identical(*_build_both(_synthetic_corpus(400, seed=3)))


@pytest.mark.parametrize("corpus", [
    [("d1", "")],                                     # empty document
    [("d1", "   !!!   ")],                            # nothing tokenisable
    [("d1", "a"), ("d2", "a a"), ("d3", "b")],        # minimal
    [("d1", "COVID-19 SARS-CoV-2 100% α β 你好")],     # punctuation and non-Latin
    [("d1", "x" * 40 + " ok")],                       # token beyond max_token_len
    [("d1", "Ω".join(["term"] * 50))],                # non-ASCII separators
])
def test_cpp_builder_matches_python_on_awkward_input(corpus):
    """UTF-8 continuation bytes must never be mistaken for token characters, and
    the over-long-token filter must drop exactly what the Python analyzer drops
    -- document length feeds BM25's normalisation, so any drift changes scores."""
    pytest.importorskip("submission._fastbuild")
    _assert_identical(*_build_both(corpus))


def test_cpp_builder_declines_configs_it_cannot_reproduce():
    """Stemming and stopword removal are not implemented in the kernel; it must
    say so rather than silently building a different index."""
    fb = pytest.importorskip("submission._fastbuild")
    from submission._analysis import AnalysisConfig
    assert fb.Builder.supports(AnalysisConfig())
    assert not fb.Builder.supports(AnalysisConfig(stemmer="porter"))
    assert not fb.Builder.supports(AnalysisConfig(remove_stopwords=True))
    assert not fb.Builder.supports(AnalysisConfig(split_alphanum=True))
