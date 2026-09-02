"""
submission/_analysis.py -- tokenizer/stemmer used by both indexing and
querying. config gets saved in the index meta so build and query never
go out of sync
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# small stoplist, not used by default (see report)
DEFAULT_STOPWORDS = frozenset("""
a an and are as at be by for from has he in is it its of on that the to was were
will with this these those there their they them then than or but not no if
""".split())


@dataclass(frozen=True)
class AnalysisConfig:
    lowercase: bool = True
    remove_stopwords: bool = False
    stemmer: Optional[str] = None       # None | "porter"
    min_token_len: int = 1
    max_token_len: int = 32
    split_alphanum: bool = False        # covid19 -> covid, 19

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "AnalysisConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


DEFAULT_CONFIG = AnalysisConfig()

_ALPHA_NUM_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])")


class _Analyzer:
    """holds the stem cache so class not just a function"""

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
            from nltk.stem import PorterStemmer  # noqa: F401
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
    if config is None or config == DEFAULT_CONFIG:
        return _default_analyzer(text)
    return _Analyzer(config)(text)


def make_analyzer(config: AnalysisConfig = DEFAULT_CONFIG) -> _Analyzer:
    """use this instead of analyze() in a loop, keeps stem cache warm"""
    return _Analyzer(config)
