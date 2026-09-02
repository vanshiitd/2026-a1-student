"""
tests/test_ranking_components.py — correctness of the required retrievers
(assignment Section 7: "Boolean/VSM and BM25 retrievers are both correctly
implemented and independently verifiable, graded by unit tests against known
small examples, not leaderboard score alone").

Every expected value here is derived by hand in the comments from the formulas
in the assignment docstrings, and computed in the test from those derivations —
never by calling the implementation and pasting what it returned. A test that
asserts the code agrees with itself would pass just as happily on a wrong
implementation.

The fixture corpus is three documents chosen so that the cases worth checking
actually arise:

    d1: "a b"        tokens [a, b]        len 2
    d2: "a a c"      tokens [a, a, c]     len 3
    d3: "a c c c"    tokens [a, c, c, c]  len 4

    N = 3,  total tokens = 9,  avgdl = 3.0
    df: a = 3 (every document), b = 1 (rare), c = 2
    cf: a = 4,                  b = 1,        c = 4

`a` appearing in every document gives IDF exactly 0 under the VSM weighting,
which exercises the "term carries no discriminative signal" path; `b` is a
singleton; and d2/d3 are constructed to tie exactly under cosine similarity so
the deterministic tie-break is testable.
"""
import math
import tempfile

import pytest

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex

CORPUS = [("d1", "a b"), ("d2", "a a c"), ("d3", "a c c c")]
N = 3
AVGDL = 3.0


@pytest.fixture
def index():
    ix = InvertedIndex()
    ix.build(CORPUS)
    bm25.build(ix)
    boolean_vsm.build(ix)
    return ix


# ---------------------------------------------------------------------------
# Index statistics
# ---------------------------------------------------------------------------

def test_index_statistics_match_hand_count(index):
    assert index.N == 3
    assert index.total_tokens == 9
    assert index.avg_doc_len == pytest.approx(3.0)
    assert index.document_frequency("a") == 3
    assert index.document_frequency("b") == 1
    assert index.document_frequency("c") == 2
    assert index.document_frequency("missing") == 0
    assert index.collection_frequency("a") == 4
    assert index.collection_frequency("c") == 4


def test_postings_are_sorted_and_correct(index):
    doc_ids, tfs = index.postings("c")
    assert list(doc_ids) == [1, 2]          # d2, d3 in corpus order
    assert list(tfs) == [1, 3]
    assert list(doc_ids) == sorted(doc_ids)


# ---------------------------------------------------------------------------
# BM25 — hand-computed
# ---------------------------------------------------------------------------

def _bm25_idf(df, n=N):
    """IDF = ln( (N - df + 0.5) / (df + 0.5) + 1 )  — assignment bm25.py."""
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_term(tf, dl, df, k1, b, n=N, avgdl=AVGDL):
    """IDF * tf*(k1+1) / ( tf + k1*(1 - b + b*dl/avgdl) )."""
    return _bm25_idf(df, n) * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))


def test_bm25_single_term_query_matches_hand_computation(index):
    # Query "c" with textbook k1=1.2, b=0.75.
    #   IDF(c) = ln((3 - 2 + 0.5)/(2 + 0.5) + 1) = ln(1.6)      = 0.470004
    #   d2: tf=1, dl=3 -> norm = 1.2*(0.25 + 0.75*3/3) = 1.2
    #       contrib = 0.470004 * (1*2.2)/(1 + 1.2)   = 0.470004 * 1.000000
    #   d3: tf=3, dl=4 -> norm = 1.2*(0.25 + 0.75*4/3) = 1.5
    #       contrib = 0.470004 * (3*2.2)/(3 + 1.5)   = 0.470004 * 1.466667
    # d3 scores higher: three occurrences beats one, even in a longer document.
    results = bm25.score("c", k=10, k1=1.2, b=0.75)
    assert [doc for doc, _ in results] == ["d3", "d2"]

    expected_d3 = _bm25_term(tf=3, dl=4, df=2, k1=1.2, b=0.75)
    expected_d2 = _bm25_term(tf=1, dl=3, df=2, k1=1.2, b=0.75)
    scores = dict(results)
    assert scores["d3"] == pytest.approx(expected_d3, rel=1e-12)
    assert scores["d2"] == pytest.approx(expected_d2, rel=1e-12)


def test_bm25_multi_term_query_sums_term_contributions(index):
    # Query "b c": d1 matches only b, d2 and d3 only c. Contributions add.
    results = dict(bm25.score("b c", k=10, k1=1.2, b=0.75))
    expected = {
        "d1": _bm25_term(tf=1, dl=2, df=1, k1=1.2, b=0.75),                    # b only
        "d2": _bm25_term(tf=1, dl=3, df=2, k1=1.2, b=0.75),                    # c only
        "d3": _bm25_term(tf=3, dl=4, df=2, k1=1.2, b=0.75),                    # c only
    }
    for doc, want in expected.items():
        assert results[doc] == pytest.approx(want, rel=1e-12)
    # b is a singleton term, so its IDF dominates: d1 outranks both c-matches.
    assert max(results, key=results.get) == "d1"


def test_bm25_zero_idf_term_still_scores_nonzero(index):
    # `a` occurs in every document, so df = N = 3 and
    #   IDF = ln((3-3+0.5)/(3+0.5) + 1) = ln(1.142857) = 0.133531 > 0.
    # The +1 smoothing in the assignment's IDF form keeps this positive rather
    # than collapsing to zero as the unsmoothed ln(N/df) would.
    results = bm25.score("a", k=10, k1=1.2, b=0.75)
    assert len(results) == 3
    assert all(score > 0 for _doc, score in results)
    assert _bm25_idf(3) == pytest.approx(math.log(1 + 0.5 / 3.5), rel=1e-12)


def test_bm25_k1_and_b_are_real_parameters(index):
    """k1/b must be genuinely tunable, not constants captured in the body —
    the assignment requires sweeping them and the oral defense perturbs them."""
    default = bm25.score("c", k=10, k1=1.2, b=0.75)
    other_k1 = bm25.score("c", k=10, k1=2.5, b=0.75)
    other_b = bm25.score("c", k=10, k1=1.2, b=0.10)
    assert dict(default) != dict(other_k1), "changing k1 did not change scores"
    assert dict(default) != dict(other_b), "changing b did not change scores"


def test_bm25_b_zero_disables_length_normalisation(index):
    # With b = 0 the denominator loses its |D|/avgdl dependence entirely, so the
    # score depends only on tf and df. d2 (tf=1, dl=3) and a hypothetical
    # equal-tf document of any length must score identically.
    scores = dict(bm25.score("c", k=10, k1=1.2, b=0.0))
    expected_d2 = _bm25_idf(2) * (1 * 2.2) / (1 + 1.2)
    assert scores["d2"] == pytest.approx(expected_d2, rel=1e-12)


def test_bm25_higher_k1_reduces_tf_saturation(index):
    """Raising k1 lets repeated occurrences count for more, so d3 (tf=3) should
    gain on d2 (tf=1). This is the exact prediction the oral defense asks for."""
    def ratio(k1):
        s = dict(bm25.score("c", k=10, k1=k1, b=0.75))
        return s["d3"] / s["d2"]
    assert ratio(2.5) > ratio(0.5)


# ---------------------------------------------------------------------------
# Vector-space model — hand-computed
# ---------------------------------------------------------------------------
#   w(t,d) = tf(t,d) * ln(N/df(t))
#   idf(a) = ln(3/3) = 0.0        <- in every document, carries no signal
#   idf(b) = ln(3/1) = 1.0986123
#   idf(c) = ln(3/2) = 0.4054651
#
#   ||d1|| = |w(b)| = 1 * 1.0986123 = 1.0986123   (a contributes 0)
#   ||d2|| = |w(c)| = 1 * 0.4054651 = 0.4054651
#   ||d3|| = |w(c)| = 3 * 0.4054651 = 1.2163953
IDF_B = math.log(3 / 1)
IDF_C = math.log(3 / 2)


def test_vsm_cosine_matches_hand_computation(index):
    # Query "b c":  q = {b: 1*IDF_B, c: 1*IDF_C},  ||q|| = sqrt(B^2 + C^2)
    #   d1: dot = B*B         -> cos = B^2 / (||q|| * B)        = 0.938152
    #   d2: dot = C*(1*C)     -> cos = C^2 / (||q|| * C)        = 0.346245
    #   d3: dot = C*(3*C)     -> cos = 3C^2 / (||q|| * 3C)      = 0.346245
    # d2 and d3 tie exactly: cosine is scale-invariant, and both documents point
    # in the same direction (only c carries weight, since idf(a) = 0).
    q_norm = math.sqrt(IDF_B**2 + IDF_C**2)
    expected_d1 = (IDF_B * IDF_B) / (q_norm * IDF_B)
    expected_d2 = (IDF_C * IDF_C) / (q_norm * IDF_C)
    expected_d3 = (IDF_C * 3 * IDF_C) / (q_norm * 3 * IDF_C)

    results = boolean_vsm.vsm_score("b c", k=10)
    scores = dict(results)
    assert scores["d1"] == pytest.approx(expected_d1, rel=1e-12)
    assert scores["d2"] == pytest.approx(expected_d2, rel=1e-12)
    assert scores["d3"] == pytest.approx(expected_d3, rel=1e-12)
    assert expected_d2 == pytest.approx(expected_d3, rel=1e-12), "fixture should tie d2/d3"
    # Tie broken deterministically by corpus order.
    assert [doc for doc, _ in results] == ["d1", "d2", "d3"]


def test_vsm_cosine_is_bounded_by_one(index):
    for doc, score in boolean_vsm.vsm_score("a b c", k=10):
        assert -1e-9 <= score <= 1.0 + 1e-9, f"{doc} cosine out of range: {score}"


def test_vsm_query_of_only_zero_idf_terms_scores_zero(index):
    # `a` is in every document => idf 0 => query vector is the zero vector, so
    # cosine is undefined. Must degrade to an empty result, not divide by zero.
    assert boolean_vsm.vsm_score("a", k=10) == []


def test_vsm_identical_query_and_document_scores_one(index):
    # Query "b" against d1, whose only non-zero-weight term is b: the vectors
    # are collinear, so cosine must be exactly 1.
    scores = dict(boolean_vsm.vsm_score("b", k=10))
    assert scores["d1"] == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Boolean retrieval
# ---------------------------------------------------------------------------

def test_boolean_and_returns_intersection(index):
    assert boolean_vsm.boolean_search("a c", mode="and") == ["d2", "d3"]


def test_boolean_or_returns_union(index):
    assert boolean_vsm.boolean_search("b c", mode="or") == ["d1", "d2", "d3"]


def test_boolean_and_with_disjoint_terms_is_empty(index):
    # No document contains both b and c.
    assert boolean_vsm.boolean_search("b c", mode="and") == []


def test_boolean_unknown_term_empties_conjunction_but_not_disjunction(index):
    assert boolean_vsm.boolean_search("b zzz", mode="and") == []
    assert boolean_vsm.boolean_search("b zzz", mode="or") == ["d1"]


def test_boolean_repeated_term_is_idempotent(index):
    assert boolean_vsm.boolean_search("c c c", mode="and") == ["d2", "d3"]


def test_boolean_rejects_unknown_mode(index):
    with pytest.raises(ValueError):
        boolean_vsm.boolean_search("a", mode="xor")


# ---------------------------------------------------------------------------
# Contract / edge cases shared by every scorer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    lambda: bm25.score("a b c", k=2, k1=1.2, b=0.75),
    lambda: boolean_vsm.vsm_score("a b c", k=2),
])
def test_scorers_respect_k(index, fn):
    assert len(fn()) <= 2


@pytest.mark.parametrize("fn", [
    lambda k: bm25.score("a b c", k=k, k1=1.2, b=0.75),
    lambda k: boolean_vsm.vsm_score("a b c", k=k),
])
def test_scorers_handle_k_zero_and_k_larger_than_corpus(index, fn):
    assert fn(0) == []
    assert len(fn(100)) <= 3


@pytest.mark.parametrize("query", ["", "   ", "!!! ???", "zzz qqq"])
def test_scorers_handle_empty_and_unmatchable_queries(index, query):
    assert bm25.score(query, k=10, k1=1.2, b=0.75) == []
    assert boolean_vsm.vsm_score(query, k=10) == []
    assert boolean_vsm.boolean_search(query, mode="and") == []


@pytest.mark.parametrize("fn", [
    lambda: bm25.score("a b c", k=10, k1=1.2, b=0.75),
    lambda: boolean_vsm.vsm_score("a b c", k=10),
])
def test_scorers_never_return_duplicate_doc_ids(index, fn):
    """The harness rejects a repeated doc_id as a RUNTIME_ERROR."""
    docs = [doc for doc, _ in fn()]
    assert len(docs) == len(set(docs))


@pytest.mark.parametrize("fn", [
    lambda: bm25.score("a b c", k=10, k1=1.2, b=0.75),
    lambda: boolean_vsm.vsm_score("a b c", k=10),
])
def test_scorers_return_descending_scores(index, fn):
    scores = [s for _doc, s in fn()]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("fn", [
    lambda: bm25.score("a b c", k=10, k1=1.2, b=0.75),
    lambda: boolean_vsm.vsm_score("b c", k=10),
    lambda: boolean_vsm.boolean_search("a c", mode="and"),
])
def test_results_are_deterministic_across_repeated_calls(index, fn):
    assert fn() == fn()


def test_save_load_round_trip_preserves_rankings(index):
    """The harness builds and queries in separate processes; a persistence gap
    shows up as a changed ranking, not as an exception."""
    before_bm25 = bm25.score("b c", k=10, k1=1.2, b=0.75)
    before_vsm = boolean_vsm.vsm_score("b c", k=10)
    before_bool = boolean_vsm.boolean_search("a c", mode="and")

    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        reloaded = InvertedIndex.load(index_dir)

    bm25.build(reloaded)
    boolean_vsm.build(reloaded)
    assert bm25.score("b c", k=10, k1=1.2, b=0.75) == before_bm25
    assert boolean_vsm.vsm_score("b c", k=10) == before_vsm
    assert boolean_vsm.boolean_search("a c", mode="and") == before_bool


def test_scorers_raise_a_clear_error_before_build(index):
    """Calling a scorer with no index bound must fail loudly rather than
    returning a silently empty ranking."""
    bm25._INDEX = None
    boolean_vsm._INDEX = None
    try:
        with pytest.raises(RuntimeError, match="build"):
            bm25.score("a", k=10)
        with pytest.raises(RuntimeError, match="build"):
            boolean_vsm.vsm_score("a", k=10)
    finally:
        bm25.build(index)
        boolean_vsm.build(index)


# ---------------------------------------------------------------------------
# Robustness at the graded boundary
# ---------------------------------------------------------------------------
# The harness aborts the entire run on any exception out of retrieve(), so a
# single malformed held-out query would zero all topics. These tests pin down
# that adversarial input degrades one query rather than the whole submission,
# and -- just as important -- that the degradation is never silently wrong.

ADVERSARIAL = [
    "",                                   # empty
    "   \t\n  ",                          # whitespace only
    "!!! ??? ---",                        # punctuation only
    "你好 مرحبا αβγ",  # non-Latin scripts
    "\x00\x01 null bytes",                # control characters
    "a" * 50_000,                         # pathologically long single token
    " ".join(["covid"] * 5_000),          # pathologically long query
    "zzzqqq wwwvvv",                      # entirely out of vocabulary
    "COVID-19 SARS-CoV-2 100%",           # punctuation mixed into real terms
]


@pytest.mark.parametrize("query", ADVERSARIAL)
def test_retrieve_survives_adversarial_queries(index, query):
    from submission import retrieve as entry
    entry._INDEX = index
    bm25.build(index)
    results = entry.retrieve(query, 10)
    assert isinstance(results, list)
    assert len(results) <= 10
    docs = [d for d, _ in results]
    assert len(docs) == len(set(docs)), "duplicate doc_id would be a RUNTIME_ERROR"
    scores = [s for _d, s in results]
    assert scores == sorted(scores, reverse=True)
    for doc_id, score in results:
        assert doc_id in {"d1", "d2", "d3"}
        assert isinstance(score, (int, float))


def test_retrieve_still_raises_before_load_index():
    """The guard must not swallow genuine misuse: retrieve() before
    load_index() is a harness-contract violation and should fail loudly."""
    from submission import retrieve as entry
    saved = entry._INDEX
    entry._INDEX = None
    try:
        with pytest.raises(RuntimeError, match="load_index"):
            entry.retrieve("a", 10)
    finally:
        entry._INDEX = saved


def _build_active_strategy_state(entry):
    """Set up module state mirroring what retrieve.load_index() does, using
    the `index` fixture's plain body directly."""
    ix = InvertedIndex()
    ix.build(CORPUS)
    entry._INDEX = ix
    bm25.build(ix)
    return entry._INDEX


def test_retrieve_guard_degrades_one_query_not_the_run(index, monkeypatch, capsys):
    """A scorer blowing up must cost one query, not the whole submission."""
    from submission import retrieve as entry
    entry._INDEX = index

    def boom(*_a, **_kw):
        raise ValueError("simulated scorer failure")

    monkeypatch.setattr(entry.bm25, "score", boom)
    assert entry.retrieve("a b c", 10) == []
    assert "simulated scorer failure" in capsys.readouterr().err

    monkeypatch.undo()
    _build_active_strategy_state(entry)
    assert entry.retrieve("a b c", 10), "must recover once the fault clears"
