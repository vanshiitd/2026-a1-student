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
    before = (index._docid_buf.copy(), index._tf_packed.copy(),
              index._tf_exc_val.copy(), index.doc_len.copy())
    bm25.score("alpha beta gamma", 10, k1=4.5, b=0.60)
    np.testing.assert_array_equal(before[0], index._docid_buf)
    np.testing.assert_array_equal(before[1], index._tf_packed)
    np.testing.assert_array_equal(before[2], index._tf_exc_val)
    np.testing.assert_array_equal(before[3], index.doc_len)


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
    for name in ("doc_len", "df", "cf", "_docid_buf", "_docid_off",
                 "_tf_packed", "_tf_exc_idx", "_tf_exc_val", "_term_start"):
        np.testing.assert_array_equal(getattr(py, name), getattr(c, name),
                                      err_msg=f"{name} differs between build paths")


def test_cpp_builder_produces_a_byte_identical_index():
    pytest.importorskip("submission._fastbuild")
    _assert_identical(*_build_both(_synthetic_corpus(400, seed=3)))


def test_cpp_builder_produces_a_byte_identical_index_when_stemmed():
    """The C++ Porter port must produce the identical index to the Python
    stemming path, not just an equivalent stemmer function in isolation --
    this exercises the actual build_from_jsonl()/Builder wiring end to end."""
    pytest.importorskip("submission._fastbuild")
    from submission._analysis import AnalysisConfig
    _assert_identical(*_build_both(_synthetic_corpus(400, seed=3),
                                   config=AnalysisConfig(stemmer="porter")))


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
    """Stopword removal and alphanum-splitting are not implemented in the
    kernel; it must say so rather than silently building a different index.

    Porter stemming IS supported (a C++ port of nltk's NLTK_EXTENSIONS mode,
    exhaustively verified against nltk across all 207,034 distinct tokens in
    the real corpus vocabulary -- see test_porter_stemmer_matches_nltk below).
    """
    fb = pytest.importorskip("submission._fastbuild")
    from submission._analysis import AnalysisConfig
    assert fb.Builder.supports(AnalysisConfig())
    assert fb.Builder.supports(AnalysisConfig(stemmer="porter"))
    assert not fb.Builder.supports(AnalysisConfig(stemmer="snowball"))
    assert not fb.Builder.supports(AnalysisConfig(remove_stopwords=True))
    assert not fb.Builder.supports(AnalysisConfig(split_alphanum=True))


# ---------------------------------------------------------------------------
# Build-definition placement (course staff clarification, 26 Aug)
# ---------------------------------------------------------------------------
# Staff run `python setup.py build_ext --inplace` FROM INSIDE submission/, and
# only if submission/setup.py exists. Getting this wrong fails silently: the
# build is skipped or lands the .so in the wrong place, every import falls back
# to pure Python, and the submission still passes its tests -- just slowly.
# These tests pin the placement down so that cannot happen unnoticed.

def test_setup_py_lives_in_submission_where_grading_looks_for_it():
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    assert (repo / "submission" / "setup.py").is_file(), (
        "submission/setup.py is missing. Staff only build a compiled extension "
        "if this exact path exists; without it the C kernels are never compiled.")
    assert not (repo / "setup.py").exists(), (
        "A stray root setup.py is no longer the build definition and will "
        "mislead anyone running the build by hand.")


def test_extension_module_names_are_bare_so_inplace_output_lands_correctly():
    """Dotted names would nest the .so at submission/submission/_fast.so."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "submission" / "setup.py").read_text()
    assert '"submission._fast"' not in src and '"submission._fastbuild"' not in src, (
        "Module names must be bare (_fast, not submission._fast): setup.py is "
        "run with submission/ as the working directory, so a dotted name puts "
        "the built .so one directory too deep and the import fails.")
    assert 'sources=["_fast.pyx"]' in src and 'sources=["_fastbuild.pyx"]' in src, (
        "Source paths must be relative to submission/, not the repo root.")


def test_float_safety_flag_survives_the_move():
    """-ffp-contract=off is what keeps the C kernel bit-identical to NumPy."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "submission" / "setup.py").read_text()
    assert '"-ffp-contract=off"' in src, (
        "Without this flag the compiler fuses a*b+c into an FMA, which rounds "
        "once instead of twice and diverges from NumPy on the full corpus.")
    # Quoted form only: the file explains in a COMMENT why -ffast-math is
    # excluded, and that prose must not trip the check.
    assert '"-ffast-math"' not in src, "-ffast-math would break bit-identity."


def test_no_precompiled_binaries_are_tracked_by_git():
    """Staff are explicit: do not commit a precompiled .so."""
    import subprocess, pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(["git", "ls-files"], cwd=repo,
                             capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        # The grading/Docker image is python:*-slim and ships no git binary.
        # This check is about what the repository tracks, so it is meaningful
        # only where there is a repository to ask.
        pytest.skip("git not available (slim image); nothing to check here")
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = out.stdout.split()
    bad = [f for f in tracked if f.endswith((".so", ".pyd", ".dylib"))]
    assert not bad, f"precompiled binaries are tracked: {bad}"


def test_build_index_does_not_compile_anything():
    """A one-time compile must not be billed to the index-build-time metric."""
    import pathlib
    subdir = pathlib.Path(__file__).resolve().parent.parent / "submission"
    for py in subdir.glob("*.py"):
        if py.name == "setup.py":
            continue
        src = py.read_text()
        for forbidden in ("subprocess", "pyximport", "build_ext", "os.system"):
            assert forbidden not in src, (
                f"{py.name} references {forbidden!r}; compilation must happen at "
                "image-build time, never inside build_index().")


# ---------------------------------------------------------------------------
# Porter stemmer C++ port (submission/_fastbuild.pyx's porter_stem)
# ---------------------------------------------------------------------------
# A direct structural port of nltk.stem.porter.PorterStemmer's NLTK_EXTENSIONS
# mode (the only mode submission/_analysis.py uses), so the C++ build path
# can stem tokens instead of falling back to pure Python. Correctness bar:
# bit-identical to nltk for every token the corpus's tokenizer can produce.
#
# _PORTER_EXAMPLES covers every worked example from the algorithm's own
# docstrings (all 8 steps, the NLTK-only ied/ies/alli/fulli/logi extensions,
# every irregular-pool entry, alphanumeric corpus-specific tokens, and the
# length<=2 passthrough) -- runs unconditionally, no corpus needed. The
# exhaustive test below additionally checks all ~207K distinct tokens in the
# real corpus when it's present locally, skipping gracefully otherwise.

_PORTER_EXAMPLES = [
    "caresses", "ponies", "ties", "caress", "cats", "feed", "agreed",
    "plastered", "bled", "motoring", "sing", "conflated", "troubled", "sized",
    "hopping", "tanned", "falling", "hissing", "fizzed", "failing", "filing",
    "happy", "sky", "relational", "conditional", "rational", "valenci",
    "hesitanci", "digitizer", "conformabli", "radicalli", "differentli",
    "vileli", "analogousli", "vietnamization", "predication", "operator",
    "feudalism", "decisiveness", "hopefulness", "callousness", "formaliti",
    "sensitiviti", "sensibiliti", "triplicate", "formative", "formalize",
    "electriciti", "electrical", "hopeful", "goodness", "revival",
    "allowance", "inference", "airliner", "gyroscopic", "adjustable",
    "defensible", "irritant", "replacement", "adjustment", "dependent",
    "adoption", "homologou", "communism", "activate", "angulariti",
    "homologous", "effective", "bowdlerize", "probate", "rate", "cease",
    "controll", "roll", "spied", "died", "flies", "dies", "spy", "fly",
    "try", "enjoy", "enjoyment", "geology", "theology", "archaeology",
    "philology", "skies", "dying", "lying", "tying", "news", "innings",
    "inning", "outings", "canning", "howe", "proceed", "exceed", "succeed",
    "covid", "19", "sars2", "coronavirus", "pandemic", "19th", "a", "ab",
]


def test_porter_stemmer_matches_nltk_on_worked_examples():
    fb = pytest.importorskip("submission._fastbuild")
    nltk_stem = pytest.importorskip("nltk.stem.porter").PorterStemmer().stem
    for word in _PORTER_EXAMPLES:
        cpp = fb.porter_stem(word.encode()).decode()
        ref = nltk_stem(word)
        assert cpp == ref, f"{word!r}: C++={cpp!r} nltk={ref!r}"


def test_porter_stemmer_matches_nltk_exhaustively_on_the_real_corpus():
    """Every one of the ~207,034 distinct pre-stem tokens the real corpus can
    produce, not a sample. Skips if the full corpus isn't present locally --
    it's gitignored and fetched separately, not part of the required tree."""
    import json
    import os
    import re

    fb = pytest.importorskip("submission._fastbuild")
    nltk_stem = pytest.importorskip("nltk.stem.porter").PorterStemmer().stem

    corpus_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "full", "corpus.jsonl")
    if not os.path.exists(corpus_path):
        pytest.skip("data/full/corpus.jsonl not present locally")

    token_re = re.compile(r"[a-z0-9]+")
    vocab = set()
    with open(corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            vocab.update(token_re.findall(obj["text"].lower()))

    mismatches = [
        (w, fb.porter_stem(w.encode()).decode(), nltk_stem(w))
        for w in vocab
        if fb.porter_stem(w.encode()).decode() != nltk_stem(w)
    ]
    assert not mismatches, f"{len(mismatches)} mismatches, e.g. {mismatches[:10]}"


def test_porter_stemmer_handles_edge_cases_without_crashing():
    """Empty input, length-1/2 words, all-consonants, all-vowels, runs of y."""
    fb = pytest.importorskip("submission._fastbuild")
    for word in (b"", b"a", b"ab", b"y", b"yy", b"yyyy", b"bcdfg", b"aeiou",
                b"0", b"00000000", b"x" * 32):
        fb.porter_stem(word)  # must not raise
