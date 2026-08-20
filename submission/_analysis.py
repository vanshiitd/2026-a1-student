"""
submission/_analysis.py — the single text-analysis chain.

Every token that enters the index and every token that leaves a query MUST pass
through `analyze()`. That is the whole point of this module existing: an index
built with one tokenizer and queried with another silently loses recall in a way
that looks like a bad scorer rather than a bug.

The chain is deliberately config-driven rather than hardcoded. plan.md L1 sweeps
these choices (stemming, stopwords, number handling) and the winning
configuration is frozen on 24 Aug; until then the defaults are the plain
lowercase-alphanumeric behaviour the starter shipped with, so day-1 numbers are
comparable to the starter baseline.

The active config is persisted into the index (`meta.json`) and restored at load
time, so a query can never be analysed differently from the corpus it is being
run against -- even if the defaults here change between build and load.
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import re

# Lowercase alphanumeric runs. Matches submission/indexer.py's shipped tokenizer
# so day-1 results are directly comparable to the starter baseline.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small INQUERY-style stoplist. Not enabled by default -- plan.md L1 decides
# this empirically, and note that stopwords must stay in the *positional* index
# even if excluded from scoring, or proximity distances become wrong.
DEFAULT_STOPWORDS = frozenset("""
a an and are as at be by for from has he in is it its of on that the to was were
will with this these those there their they them then than or but not no if
""".split())


@dataclass(frozen=True)
class AnalysisConfig:
    """Serialisable description of the analysis chain.

    Persisted into the index so build-time and query-time analysis provably
    match. Add fields here rather than adding parameters to `analyze()`.
    """
    lowercase: bool = True
    remove_stopwords: bool = False
    stemmer: Optional[str] = None       # None | "porter" (plan.md L1)
    min_token_len: int = 1
    max_token_len: int = 32             # guards against junk/base64 blobs
    split_alphanum: bool = False        # "covid19" -> ["covid", "19"] (L1)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "AnalysisConfig":
        # Ignore unknown keys so an index written by a newer build still loads
        # rather than exploding at grading time.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


DEFAULT_CONFIG = AnalysisConfig()

_ALPHA_NUM_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])")


class _Analyzer:
    """Applies an AnalysisConfig. Holds the stem cache, which is why this is a
    class and not a bare function: stemming is the hot loop during indexing and
    natural text has a very high repeat rate, so memoising it is worth far more
    than it costs."""

    def __init__(self, config: AnalysisConfig = DEFAULT_CONFIG):
        self.config = config
        self._stem_cache: Dict[str, str] = {}
        self._stemmer = self._make_stemmer(config.stemmer)
        self._stopwords = DEFAULT_STOPWORDS if config.remove_stopwords else frozenset()

    @staticmethod
    def _make_stemmer(name: Optional[str]):
        if name is None:
            return None
        if name == "porter":
            # Deferred until Piazza Q4 confirms NLTK is pre-approved (see
            # notes/piazza_q1.md). plan.md L1 owns this decision on 24 Aug.
            from nltk.stem import PorterStemmer  # noqa: F401  (import-time check)
            return PorterStemmer().stem
        raise ValueError(f"unknown stemmer {name!r}")

    def _stem(self, token: str) -> str:
        if self._stemmer is None:
            return token
        cached = self._stem_cache.get(token)
        if cached is None:
            cached = self._stemmer(token)
            self._stem_cache[token] = cached
        return cached

    def __call__(self, text: str) -> List[str]:
        cfg = self.config
        raw = _TOKEN_RE.findall(text.lower() if cfg.lowercase else text)

        if cfg.split_alphanum:
            split: List[str] = []
            for tok in raw:
                split.extend(_ALPHA_NUM_SPLIT_RE.split(tok))
            raw = split

        out: List[str] = []
        lo, hi = cfg.min_token_len, cfg.max_token_len
        stop = self._stopwords
        for tok in raw:
            if not (lo <= len(tok) <= hi):
                continue
            if tok in stop:
                continue
            out.append(self._stem(tok))
        return out


_default_analyzer = _Analyzer(DEFAULT_CONFIG)


def analyze(text: str, config: Optional[AnalysisConfig] = None) -> List[str]:
    """Turn raw text into the token sequence that is actually indexed/queried.

    Passing `config=None` uses the module default. Callers that hold an index
    should pass that index's persisted config instead, so analysis provably
    matches what the postings were built from.
    """
    if config is None or config == DEFAULT_CONFIG:
        return _default_analyzer(text)
    return _Analyzer(config)(text)


def make_analyzer(config: AnalysisConfig = DEFAULT_CONFIG) -> _Analyzer:
    """Build a reusable analyzer. Prefer this over repeated `analyze()` calls in
    a hot loop -- it keeps the stem cache alive across documents."""
    return _Analyzer(config)
