"""
tests/test_fast_equivalence.py -- C extension must be bit identical to the
numpy path (not just close), and everything must still work with a pure
python fallback if the extension didn't compile
"""
import numpy as np
import os
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
    from submission import _traverse
    fast_results = bm25._score_fast(index, query, 10, k1, b)
    numpy_results = _traverse.score_single(index, query, "bm25", 10, k1=k1, b=b)

    assert [d for d, _ in fast_results] == [d for d, _ in numpy_results]
    for (_da, sa), (_db, sb) in zip(fast_results, numpy_results):
        assert sa == sb, f"score differs: {sa!r} vs {sb!r} (query={query!r})"


@pytest.mark.parametrize("seed", range(5))
def test_decode_postings_on_random_wide_gaps(seed):
    """multi-byte vbyte values, a toy corpus never reaches these"""
    rng = np.random.default_rng(seed)
    docs = np.unique(rng.integers(0, 5_000_000, size=3000)).astype(np.int64)
    tfs = rng.integers(1, 40_000, size=docs.size).astype(np.int64)
    c_docs, c_tfs = fast.decode_postings(
        encode_sorted_ids(docs), vbyte_encode(tfs), docs.size)
    np.testing.assert_array_equal(docs, c_docs)
    np.testing.assert_array_equal(tfs, c_tfs)


def test_fallback_produces_identical_results_when_extension_is_absent(index, monkeypatch):
    with_fast = bm25.score("beta gamma", 10, k1=4.5, b=0.60)
    monkeypatch.setattr(bm25, "HAVE_FAST", False)
    without_fast = bm25.score("beta gamma", 10, k1=4.5, b=0.60)
    assert with_fast == without_fast, "fallback path diverges from the C path"
    assert without_fast, "fallback returned nothing"


def test_extension_does_not_mutate_the_index(index):
    before = (index._docid_buf.copy(), index._tf_packed.copy(),
              index._tf_exc_val.copy(), index.doc_len.copy())
    bm25.score("alpha beta gamma", 10, k1=4.5, b=0.60)
    np.testing.assert_array_equal(before[0], index._docid_buf)
    np.testing.assert_array_equal(before[1], index._tf_packed)
    np.testing.assert_array_equal(before[2], index._tf_exc_val)
    np.testing.assert_array_equal(before[3], index.doc_len)


def _synthetic_corpus(n_docs=1500, seed=0):
    """bigger + wider length spread than the fixture -- caught a real FMA
    rounding bug once that only showed up at this scale, not on the small
    fixture"""
    rng = np.random.default_rng(seed)
    vocab = [f"t{i}" for i in range(400)]
    corpus = []
    for i in range(n_docs):
        length = int(rng.integers(3, 600))
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
                "check -ffp-contract=off in setup.py"
            )


# build side kernel (_fastbuild.pyx)

def _build_both(corpus, config=None):
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
    pytest.importorskip("submission._fastbuild")
    from submission._analysis import AnalysisConfig
    _assert_identical(*_build_both(_synthetic_corpus(400, seed=3),
                                   config=AnalysisConfig(stemmer="porter")))


@pytest.mark.parametrize("corpus", [
    [("d1", "")],
    [("d1", "   !!!   ")],
    [("d1", "a"), ("d2", "a a"), ("d3", "b")],
    [("d1", "COVID-19 SARS-CoV-2 100% α β 你好")],
    [("d1", "x" * 40 + " ok")],
    [("d1", "Ω".join(["term"] * 50))],
])
def test_cpp_builder_matches_python_on_awkward_input(corpus):
    pytest.importorskip("submission._fastbuild")
    _assert_identical(*_build_both(corpus))


def test_cpp_builder_declines_configs_it_cannot_reproduce():
    fb = pytest.importorskip("submission._fastbuild")
    from submission._analysis import AnalysisConfig
    assert fb.Builder.supports(AnalysisConfig())
    assert fb.Builder.supports(AnalysisConfig(stemmer="porter"))
    assert not fb.Builder.supports(AnalysisConfig(stemmer="snowball"))
    assert not fb.Builder.supports(AnalysisConfig(remove_stopwords=True))
    assert not fb.Builder.supports(AnalysisConfig(split_alphanum=True))


# build definition placement, staff run this from inside submission/

def test_setup_py_lives_in_submission_where_grading_looks_for_it():
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    assert (repo / "submission" / "setup.py").is_file(), (
        "submission/setup.py missing, staff only build the extension if this exact path exists")
    assert not (repo / "setup.py").exists(), "stray root setup.py, not the real build def anymore"


def test_extension_module_names_are_bare_so_inplace_output_lands_correctly():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "submission" / "setup.py").read_text()
    assert '"submission._fast"' not in src and '"submission._fastbuild"' not in src, (
        "module names must be bare, dotted names put the .so one dir too deep")
    assert 'sources=["_fast.pyx"]' in src and 'sources=["_fastbuild.pyx"]' in src


def test_float_safety_flag_survives_the_move():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "submission" / "setup.py").read_text()
    assert '"-ffp-contract=off"' in src, "without this the C kernel drifts from numpy"
    assert '"-ffast-math"' not in src, "-ffast-math would break bit-identity"


def test_no_precompiled_binaries_are_tracked_by_git():
    import subprocess, pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(["git", "ls-files"], cwd=repo,
                             capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        pytest.skip("git not available (slim image); nothing to check here")
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = out.stdout.split()
    bad = [f for f in tracked if f.endswith((".so", ".pyd", ".dylib"))]
    assert not bad, f"precompiled binaries are tracked: {bad}"


def test_build_index_does_not_compile_anything():
    """compile has to happen at image-build time, not inside build_index()"""
    import pathlib
    subdir = pathlib.Path(__file__).resolve().parent.parent / "submission"
    for py in subdir.glob("*.py"):
        if py.name == "setup.py":
            continue
        src = py.read_text()
        for forbidden in ("subprocess", "pyximport", "build_ext", "os.system"):
            assert forbidden not in src, f"{py.name} references {forbidden!r}"


# porter stemmer C++ port, bit identical to nltk required

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
    """all ~207k distinct tokens the real corpus can produce, not a sample.
    skips if the corpus isn't downloaded locally"""
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
    fb = pytest.importorskip("submission._fastbuild")
    for word in (b"", b"a", b"ab", b"y", b"yy", b"yyyy", b"bcdfg", b"aeiou",
                b"0", b"00000000", b"x" * 32):
        fb.porter_stem(word)  # must not raise


# parallel build (indexer.py build_from_jsonl_parallel)

def _write_jsonl_corpus(path, corpus):
    import json
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in corpus:
            f.write(json.dumps({"doc_id": doc_id, "text": text}) + "\n")


@pytest.mark.parametrize("n_workers", [2, 4])
def test_parallel_build_is_byte_identical_to_serial(tmp_path, n_workers):
    pytest.importorskip("submission._fastbuild")
    from submission._analysis import AnalysisConfig

    corpus_path = str(tmp_path / "corpus.jsonl")
    _write_jsonl_corpus(corpus_path, _synthetic_corpus(600, seed=5))

    for cfg in (AnalysisConfig(), AnalysisConfig(stemmer="porter")):
        serial = InvertedIndex(cfg)
        serial.build_from_jsonl(corpus_path)

        parallel = InvertedIndex(cfg)
        used = parallel.build_from_jsonl_parallel(
            corpus_path, n_workers=n_workers, min_docs=0)
        assert used, "parallel path declined despite min_docs=0"

        assert serial.N == parallel.N
        assert serial.terms == parallel.terms
        assert serial.doc_ids == parallel.doc_ids
        for name in ("doc_len", "df", "cf", "_docid_buf", "_docid_off",
                     "_tf_packed", "_tf_exc_idx", "_tf_exc_val", "_term_start"):
            np.testing.assert_array_equal(
                getattr(serial, name), getattr(parallel, name),
                err_msg=f"{name} differs (n_workers={n_workers}, "
                       f"stemmer={cfg.stemmer})")


def test_parallel_build_declines_below_the_doc_count_threshold():
    pytest.importorskip("submission._fastbuild")
    toy_corpus = os.path.join(os.path.dirname(__file__), "..", "data", "toy", "corpus.jsonl")
    ix = InvertedIndex()
    used = ix.build_from_jsonl_parallel(toy_corpus)
    assert not used, "toy corpus is far below any sane parallel-build threshold"


def test_parallel_build_declines_at_n_workers_1():
    pytest.importorskip("submission._fastbuild")
    ix = InvertedIndex()
    assert not ix.build_from_jsonl_parallel(
        os.path.join(os.path.dirname(__file__), "..", "data", "toy", "corpus.jsonl"),
        n_workers=1, min_docs=0)


def test_parallel_build_declines_for_unsupported_analysis_chains():
    pytest.importorskip("submission._fastbuild")
    from submission._analysis import AnalysisConfig
    ix = InvertedIndex(AnalysisConfig(remove_stopwords=True))
    assert not ix.build_from_jsonl_parallel(
        os.path.join(os.path.dirname(__file__), "..", "data", "toy", "corpus.jsonl"),
        min_docs=0)


def test_parallel_build_handles_empty_and_tiny_corpora(tmp_path):
    pytest.importorskip("submission._fastbuild")
    for corpus in ([], [("d1", "a b c")], [("d1", "a"), ("d2", "b")]):
        corpus_path = str(tmp_path / f"c{len(corpus)}.jsonl")
        _write_jsonl_corpus(corpus_path, corpus)
        ix = InvertedIndex()
        used = ix.build_from_jsonl_parallel(corpus_path, n_workers=4, min_docs=0)
        assert used
        assert ix.N == len(corpus)
