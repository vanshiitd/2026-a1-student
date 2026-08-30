# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
submission/_fastbuild.pyx — tokenisation and posting emission in C++.

Phase-timing put 69% of the build in tokenisation, Counter + ~16.3M list
appends, and converting those lists to NumPy arrays -- because every one of
~29M tokens became a Python str object just to be looked up in the vocabulary
dict. Here tokens stay as raw bytes, interned through a C++
unordered_map<string, int>, so no Python object is created per token.

Per-document term frequencies are counted in an O(1) scratch vector indexed
by term id, with a touched-list to reset only what was used.

`text.lower()` stays in Python (already C-speed, handles Unicode case
mapping) before this scans the UTF-8 bytes for [a-z0-9]+ runs -- UTF-8
continuation bytes are all >= 0x80 so can never be mistaken for ASCII
alphanumerics, meaning this produces exactly the tokens Python's `re` would.

Only the default analysis chain is supported; `Builder.supports()` reports
that, and the caller falls back to Python for anything else.
"""

import numpy as np
cimport numpy as cnp

from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t

cnp.import_array()


cdef inline bint _is_token_byte(unsigned char c) noexcept nogil:
    return (c >= b'a' and c <= b'z') or (c >= b'0' and c <= b'9')


# ---------------------------------------------------------------------------
# Porter stemmer (nltk's NLTK_EXTENSIONS mode), ported to C++.
#
# A direct structural port of nltk.stem.porter.PorterStemmer -- same function
# names, same step order, same control flow -- so it can be reviewed against
# the reference line-for-line. Only NLTK_EXTENSIONS is implemented: it is
# nltk's own default and the only mode submission/_analysis.py uses
# (PorterStemmer() with no mode argument).
#
# Correctness bar: every token this corpus's tokenizer can produce ([a-z0-9]+,
# already lowercase, so no .lower() step is needed here) must stem identically
# to NLTK's own stem() -- exhaustively verified against the full corpus
# vocabulary, not spot-checked. See tests/test_porter_equivalence.py.
# ---------------------------------------------------------------------------
from libcpp.unordered_map cimport unordered_map as _umap


cdef inline bint _is_vowel(char c) noexcept nogil:
    return c == b'a' or c == b'e' or c == b'i' or c == b'o' or c == b'u'


cdef void _consonant_flags(const string& word, vector[bint]& flags) noexcept nogil:
    """flags[i] = True iff word[i] is a consonant, single left-to-right pass.

    A 'y' is a consonant iff the preceding letter is not one (or it starts
    the word) -- exactly the previous flag. This is nltk's own justification
    for computing all flags in one O(n) pass rather than a per-index
    backward walk over runs of 'y': the two are provably equivalent.
    """
    cdef Py_ssize_t n = <Py_ssize_t>word.size()
    cdef Py_ssize_t i
    cdef char c
    flags.clear()
    for i in range(n):
        c = word[i]
        if _is_vowel(c):
            flags.push_back(False)
        elif c == b'y':
            if i == 0:
                flags.push_back(True)
            else:
                flags.push_back(not flags[i - 1])
        else:
            flags.push_back(True)


cdef int _measure(const string& word) noexcept nogil:
    """Porter's m: count of vowel-then-consonant transitions."""
    cdef vector[bint] flags
    _consonant_flags(word, flags)
    cdef Py_ssize_t i, n = <Py_ssize_t>flags.size()
    cdef int count = 0
    for i in range(1, n):
        if (not flags[i - 1]) and flags[i]:
            count += 1
    return count


cdef inline bint _has_positive_measure(const string& word) noexcept nogil:
    return _measure(word) > 0


cdef bint _contains_vowel(const string& word) noexcept nogil:
    cdef vector[bint] flags
    _consonant_flags(word, flags)
    cdef Py_ssize_t i
    for i in range(<Py_ssize_t>flags.size()):
        if not flags[i]:
            return True
    return False


cdef bint _is_consonant_at(const string& word, Py_ssize_t i) noexcept nogil:
    cdef vector[bint] flags
    _consonant_flags(word, flags)
    return flags[i]


cdef bint _ends_double_consonant(const string& word) noexcept nogil:
    cdef Py_ssize_t n = <Py_ssize_t>word.size()
    if n < 2 or word[n - 1] != word[n - 2]:
        return False
    return _is_consonant_at(word, n - 1)


cdef bint _ends_cvc(const string& word) noexcept nogil:
    """NLTK_EXTENSIONS mode: the len==2 special case is always active."""
    cdef Py_ssize_t n = <Py_ssize_t>word.size()
    if n >= 3:
        if (_is_consonant_at(word, n - 3) and not _is_consonant_at(word, n - 2)
                and _is_consonant_at(word, n - 1)
                and word[n - 1] != b'w' and word[n - 1] != b'x' and word[n - 1] != b'y'):
            return True
    if n == 2 and (not _is_consonant_at(word, 0)) and _is_consonant_at(word, 1):
        return True
    return False


cdef inline string _replace_suffix(const string& word, const string& suffix,
                                   const string& replacement) noexcept nogil:
    if suffix.size() == 0:
        return word + replacement
    return word.substr(0, word.size() - suffix.size()) + replacement


cdef inline bint _ends_with(const string& word, const string& suffix) noexcept nogil:
    cdef size_t slen = suffix.size(), n = word.size()
    if slen > n:
        return False
    return word.substr(n - slen, slen) == suffix


cdef string _step1a(const string& word) noexcept nogil:
    if _ends_with(word, string(b"ies")) and word.size() == 4:
        return _replace_suffix(word, string(b"ies"), string(b"ie"))
    if _ends_with(word, string(b"sses")):
        return _replace_suffix(word, string(b"sses"), string(b"ss"))
    if _ends_with(word, string(b"ies")):
        return _replace_suffix(word, string(b"ies"), string(b"i"))
    if _ends_with(word, string(b"ss")):
        return word
    if _ends_with(word, string(b"s")):
        return _replace_suffix(word, string(b"s"), string(b""))
    return word


cdef string _step1b(const string& word) noexcept nogil:
    cdef string intermediate_stem, stem
    cdef bint rule_2_or_3_succeeded = False
    cdef char last_char

    if _ends_with(word, string(b"ied")):
        if word.size() == 4:
            return _replace_suffix(word, string(b"ied"), string(b"ie"))
        return _replace_suffix(word, string(b"ied"), string(b"i"))

    if _ends_with(word, string(b"eed")):
        stem = _replace_suffix(word, string(b"eed"), string(b""))
        if _measure(stem) > 0:
            return stem + string(b"ee")
        return word

    if _ends_with(word, string(b"ed")):
        intermediate_stem = _replace_suffix(word, string(b"ed"), string(b""))
        if _contains_vowel(intermediate_stem):
            rule_2_or_3_succeeded = True
    if (not rule_2_or_3_succeeded) and _ends_with(word, string(b"ing")):
        intermediate_stem = _replace_suffix(word, string(b"ing"), string(b""))
        if _contains_vowel(intermediate_stem):
            rule_2_or_3_succeeded = True

    if not rule_2_or_3_succeeded:
        return word

    if _ends_with(intermediate_stem, string(b"at")):
        return _replace_suffix(intermediate_stem, string(b"at"), string(b"ate"))
    if _ends_with(intermediate_stem, string(b"bl")):
        return _replace_suffix(intermediate_stem, string(b"bl"), string(b"ble"))
    if _ends_with(intermediate_stem, string(b"iz")):
        return _replace_suffix(intermediate_stem, string(b"iz"), string(b"ize"))
    if _ends_double_consonant(intermediate_stem):
        last_char = intermediate_stem[intermediate_stem.size() - 1]
        if last_char != b'l' and last_char != b's' and last_char != b'z':
            return intermediate_stem.substr(0, intermediate_stem.size() - 2) + string(1, last_char)
        return intermediate_stem
    if _measure(intermediate_stem) == 1 and _ends_cvc(intermediate_stem):
        return intermediate_stem + string(b"e")
    return intermediate_stem


cdef string _step1c(const string& word) noexcept nogil:
    cdef string stem
    if _ends_with(word, string(b"y")):
        stem = _replace_suffix(word, string(b"y"), string(b""))
        if stem.size() > 1 and _is_consonant_at(stem, <Py_ssize_t>stem.size() - 1):
            return stem + string(b"i")
        return word
    return word


cdef string _step2(const string& word) noexcept nogil:
    cdef string check_stem
    # NLTK-only pre-step: 'alli' -> 'al', then re-run step2 on the result,
    # rather than a single flat replacement -- this can cascade (e.g. so a
    # second '-alli' revealed underneath also gets reduced).
    if _ends_with(word, string(b"alli")):
        check_stem = _replace_suffix(word, string(b"alli"), string(b""))
        if _has_positive_measure(check_stem):
            return _step2(_replace_suffix(word, string(b"alli"), string(b"al")))

    if _ends_with(word, string(b"ational")):
        check_stem = _replace_suffix(word, string(b"ational"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ate")
        return word
    if _ends_with(word, string(b"tional")):
        check_stem = _replace_suffix(word, string(b"tional"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"tion")
        return word
    if _ends_with(word, string(b"enci")):
        check_stem = _replace_suffix(word, string(b"enci"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ence")
        return word
    if _ends_with(word, string(b"anci")):
        check_stem = _replace_suffix(word, string(b"anci"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ance")
        return word
    if _ends_with(word, string(b"izer")):
        check_stem = _replace_suffix(word, string(b"izer"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ize")
        return word
    if _ends_with(word, string(b"bli")):  # NLTK_EXTENSIONS: 'bli' not 'abli'
        check_stem = _replace_suffix(word, string(b"bli"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ble")
        return word
    if _ends_with(word, string(b"alli")):
        check_stem = _replace_suffix(word, string(b"alli"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"al")
        return word
    if _ends_with(word, string(b"entli")):
        check_stem = _replace_suffix(word, string(b"entli"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ent")
        return word
    if _ends_with(word, string(b"eli")):
        check_stem = _replace_suffix(word, string(b"eli"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"e")
        return word
    if _ends_with(word, string(b"ousli")):
        check_stem = _replace_suffix(word, string(b"ousli"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ous")
        return word
    if _ends_with(word, string(b"ization")):
        check_stem = _replace_suffix(word, string(b"ization"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ize")
        return word
    if _ends_with(word, string(b"ation")):
        check_stem = _replace_suffix(word, string(b"ation"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ate")
        return word
    if _ends_with(word, string(b"ator")):
        check_stem = _replace_suffix(word, string(b"ator"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ate")
        return word
    if _ends_with(word, string(b"alism")):
        check_stem = _replace_suffix(word, string(b"alism"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"al")
        return word
    if _ends_with(word, string(b"iveness")):
        check_stem = _replace_suffix(word, string(b"iveness"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ive")
        return word
    if _ends_with(word, string(b"fulness")):
        check_stem = _replace_suffix(word, string(b"fulness"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ful")
        return word
    if _ends_with(word, string(b"ousness")):
        check_stem = _replace_suffix(word, string(b"ousness"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ous")
        return word
    if _ends_with(word, string(b"aliti")):
        check_stem = _replace_suffix(word, string(b"aliti"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"al")
        return word
    if _ends_with(word, string(b"iviti")):
        check_stem = _replace_suffix(word, string(b"iviti"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ive")
        return word
    if _ends_with(word, string(b"biliti")):
        check_stem = _replace_suffix(word, string(b"biliti"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ble")
        return word
    if _ends_with(word, string(b"fulli")):  # NLTK-only
        check_stem = _replace_suffix(word, string(b"fulli"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ful")
        return word
    if _ends_with(word, string(b"logi")):  # NLTK-only
        # Condition checks word[:-3] (keeps the 'l' with the stem), NOT the
        # "logi"-stripped form -- deliberate, per nltk's own comment: this is
        # what lets short stems like 'geo'/'theo' behave like 'archaeo'/'philo'.
        check_stem = word.substr(0, word.size() - 3)
        if _has_positive_measure(check_stem):
            return _replace_suffix(word, string(b"logi"), string(b"log"))
        return word
    return word


cdef string _step3(const string& word) noexcept nogil:
    cdef string check_stem
    if _ends_with(word, string(b"icate")):
        check_stem = _replace_suffix(word, string(b"icate"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ic")
        return word
    if _ends_with(word, string(b"ative")):
        check_stem = _replace_suffix(word, string(b"ative"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem
        return word
    if _ends_with(word, string(b"alize")):
        check_stem = _replace_suffix(word, string(b"alize"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"al")
        return word
    if _ends_with(word, string(b"iciti")):
        check_stem = _replace_suffix(word, string(b"iciti"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ic")
        return word
    if _ends_with(word, string(b"ical")):
        check_stem = _replace_suffix(word, string(b"ical"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem + string(b"ic")
        return word
    if _ends_with(word, string(b"ful")):
        check_stem = _replace_suffix(word, string(b"ful"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem
        return word
    if _ends_with(word, string(b"ness")):
        check_stem = _replace_suffix(word, string(b"ness"), string(b""))
        if _has_positive_measure(check_stem):
            return check_stem
        return word
    return word


cdef string _step4(const string& word) noexcept nogil:
    cdef string check_stem
    if _ends_with(word, string(b"al")):
        check_stem = _replace_suffix(word, string(b"al"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ance")):
        check_stem = _replace_suffix(word, string(b"ance"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ence")):
        check_stem = _replace_suffix(word, string(b"ence"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"er")):
        check_stem = _replace_suffix(word, string(b"er"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ic")):
        check_stem = _replace_suffix(word, string(b"ic"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"able")):
        check_stem = _replace_suffix(word, string(b"able"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ible")):
        check_stem = _replace_suffix(word, string(b"ible"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ant")):
        check_stem = _replace_suffix(word, string(b"ant"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ement")):
        check_stem = _replace_suffix(word, string(b"ement"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ment")):
        check_stem = _replace_suffix(word, string(b"ment"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ent")):
        check_stem = _replace_suffix(word, string(b"ent"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ion")):
        check_stem = _replace_suffix(word, string(b"ion"), string(b""))
        if _measure(check_stem) > 1 and check_stem.size() > 0:
            if check_stem[check_stem.size() - 1] == b's' or check_stem[check_stem.size() - 1] == b't':
                return check_stem
        return word
    if _ends_with(word, string(b"ou")):
        check_stem = _replace_suffix(word, string(b"ou"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ism")):
        check_stem = _replace_suffix(word, string(b"ism"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ate")):
        check_stem = _replace_suffix(word, string(b"ate"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"iti")):
        check_stem = _replace_suffix(word, string(b"iti"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ous")):
        check_stem = _replace_suffix(word, string(b"ous"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ive")):
        check_stem = _replace_suffix(word, string(b"ive"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    if _ends_with(word, string(b"ize")):
        check_stem = _replace_suffix(word, string(b"ize"), string(b""))
        return check_stem if _measure(check_stem) > 1 else word
    return word


cdef string _step5a(const string& word) noexcept nogil:
    cdef string stem
    if _ends_with(word, string(b"e")):
        stem = _replace_suffix(word, string(b"e"), string(b""))
        if _measure(stem) > 1:
            return stem
        if _measure(stem) == 1 and not _ends_cvc(stem):
            return stem
    return word


cdef string _step5b(const string& word) noexcept nogil:
    if _ends_with(word, string(b"ll")):
        if _measure(word.substr(0, word.size() - 1)) > 1:
            return word.substr(0, word.size() - 1)
    return word


cdef _umap[string, string] _IRREGULAR_POOL
_IRREGULAR_POOL[string(b"sky")] = string(b"sky")
_IRREGULAR_POOL[string(b"skies")] = string(b"sky")
_IRREGULAR_POOL[string(b"dying")] = string(b"die")
_IRREGULAR_POOL[string(b"lying")] = string(b"lie")
_IRREGULAR_POOL[string(b"tying")] = string(b"tie")
_IRREGULAR_POOL[string(b"news")] = string(b"news")
_IRREGULAR_POOL[string(b"innings")] = string(b"inning")
_IRREGULAR_POOL[string(b"inning")] = string(b"inning")
_IRREGULAR_POOL[string(b"outings")] = string(b"outing")
_IRREGULAR_POOL[string(b"outing")] = string(b"outing")
_IRREGULAR_POOL[string(b"cannings")] = string(b"canning")
_IRREGULAR_POOL[string(b"canning")] = string(b"canning")
_IRREGULAR_POOL[string(b"howe")] = string(b"howe")
_IRREGULAR_POOL[string(b"proceed")] = string(b"proceed")
_IRREGULAR_POOL[string(b"exceed")] = string(b"exceed")
_IRREGULAR_POOL[string(b"succeed")] = string(b"succeed")
# Populated once here, at module import time (which always holds the GIL) --
# not lazily from inside the nogil stemming path, which would need to
# reacquire the GIL on every single token just to check whether it's built.


cdef string _porter_stem(const string& word) noexcept nogil:
    """Full pipeline, matching nltk.stem.porter.PorterStemmer().stem()
    exactly for NLTK_EXTENSIONS mode. Caller guarantees `word` is already
    lowercase [a-z0-9]+, so nltk's own `.lower()` step is a no-op, skipped."""
    cdef _umap[string, string].iterator it = _IRREGULAR_POOL.find(word)
    if it != _IRREGULAR_POOL.end():
        return deref(it).second
    if word.size() <= 2:
        return word
    cdef string stem = word
    stem = _step1a(stem)
    stem = _step1b(stem)
    stem = _step1c(stem)
    stem = _step2(stem)
    stem = _step3(stem)
    stem = _step4(stem)
    stem = _step5a(stem)
    stem = _step5b(stem)
    return stem


def porter_stem(bytes word not None) -> bytes:
    """Python-callable wrapper, for exhaustive validation against nltk and
    for use from Python (submission/_analysis.py could call this too, but
    currently only the C++ tokenizer path below does)."""
    cdef string s = string(<char*>word, len(word))
    return _porter_stem(s)


cdef class Builder:
    """Accumulates (term_id, doc_id, tf) triples across the whole corpus.

    Holding the output in C++ vectors rather than Python lists is a large part
    of the win: the previous path performed three Python-level `append` calls per
    posting, ~48M in total.
    """
    cdef unordered_map[string, int] vocab
    cdef vector[string] term_bytes          # term id -> its bytes, for the dictionary
    cdef vector[int] scratch_tf             # term id -> tf within the current document
    cdef vector[int] touched                # term ids used by the current document
    # Postings held per term, not one document-ordered stream: documents are
    # processed in ascending id order, so each term's doc list is already
    # ascending, removing the 16.3M-element np.lexsort the old layout needed
    # (3.06s of a 5.78s build).
    cdef vector[vector[int32_t]] post_docs
    cdef vector[vector[int32_t]] post_tfs
    cdef Py_ssize_t max_token_len
    cdef Py_ssize_t min_token_len
    cdef bint stem_tokens
    # Mirrors _analysis.py's own _stem_cache: natural text repeats tokens
    # heavily, so caching the stemmed FORM (not just vocab interning, which
    # only dedups AFTER stemming) avoids re-running the full algorithm on
    # every occurrence of a common word.
    cdef unordered_map[string, string] stem_cache

    def __cinit__(self, Py_ssize_t min_token_len=1, Py_ssize_t max_token_len=32,
                 bint stem_tokens=False):
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.stem_tokens = stem_tokens

    @staticmethod
    def supports(config) -> bool:
        """True for the analysis chains this kernel reproduces exactly: the
        default chain, and that same chain with Porter stemming (the only
        stemmer this project uses -- see submission/_analysis.py)."""
        d = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        return (d.get("lowercase", True)
                and not d.get("remove_stopwords", False)
                and d.get("stemmer") in (None, "porter")
                and not d.get("split_alphanum", False))

    def add_document(self, bytes lowered_utf8, int doc_id, Py_ssize_t prefix_tokens=-1):
        """Tokenise one document and emit its postings. Returns the token count.

        `prefix_tokens >= 0` stops after that many surviving tokens, which is how
        the pseudo-title field is built without a second pass over the text.
        """
        cdef const unsigned char* buf = <const unsigned char*>lowered_utf8
        cdef Py_ssize_t n = len(lowered_utf8)
        cdef Py_ssize_t i = 0, start
        cdef Py_ssize_t length
        cdef Py_ssize_t n_tokens = 0
        cdef int tid
        cdef string key, stemmed
        cdef unordered_map[string, int].iterator it
        cdef unordered_map[string, string].iterator cache_it
        cdef Py_ssize_t j
        cdef int t

        self.touched.clear()

        while i < n:
            if not _is_token_byte(buf[i]):
                i += 1
                continue
            start = i
            while i < n and _is_token_byte(buf[i]):
                i += 1
            length = i - start
            if length < self.min_token_len or length > self.max_token_len:
                continue
            # Document length counts tokens that SURVIVE the filter -- the
            # Python analyzer appends to its output list only after filtering,
            # and doc_len feeds BM25's length normalisation, so an off-by-any
            # here would silently change every score.
            n_tokens += 1

            key = string(<const char*>(buf + start), length)
            if self.stem_tokens:
                # Length filter above already ran on the RAW token, matching
                # _analysis.py's order exactly: filter, then stem.
                cache_it = self.stem_cache.find(key)
                if cache_it == self.stem_cache.end():
                    stemmed = _porter_stem(key)
                    self.stem_cache[key] = stemmed
                    key = stemmed
                else:
                    key = deref(cache_it).second
            it = self.vocab.find(key)
            if it == self.vocab.end():
                tid = <int>self.term_bytes.size()
                self.vocab[key] = tid
                self.term_bytes.push_back(key)
                self.scratch_tf.push_back(0)
                self.post_docs.push_back(vector[int32_t]())
                self.post_tfs.push_back(vector[int32_t]())
            else:
                tid = deref(it).second

            if self.scratch_tf[tid] == 0:
                self.touched.push_back(tid)
            self.scratch_tf[tid] += 1

            if prefix_tokens >= 0 and n_tokens >= prefix_tokens:
                break

        # Flush this document's postings, resetting only what was touched.
        for j in range(<Py_ssize_t>self.touched.size()):
            t = self.touched[j]
            self.post_docs[t].push_back(<int32_t>doc_id)
            self.post_tfs[t].push_back(<int32_t>self.scratch_tf[t])
            self.scratch_tf[t] = 0
        return n_tokens

    def terms(self):
        """Vocabulary in first-seen order, as Python strings."""
        cdef Py_ssize_t v = <Py_ssize_t>self.term_bytes.size()
        cdef Py_ssize_t i
        out = [None] * v
        for i in range(v):
            out[i] = self.term_bytes[i].decode("utf-8")
        return out

    def finish_sorted(self, const int32_t[::1] order):
        """Concatenate postings in sorted-term order. Returns (docs, tfs, df).

        `order[i]` is the original term id of the i-th alphabetically-sorted
        term. Walking terms in that order and copying each one's vector produces
        exactly the layout the encoder wants -- grouped by term, ascending by
        doc id within a term -- in a single pass, with no sort anywhere.
        """
        cdef Py_ssize_t n_terms = order.shape[0]
        cdef Py_ssize_t i, j, t, total = 0, pos = 0

        for i in range(n_terms):
            total += <Py_ssize_t>self.post_docs[order[i]].size()

        docs_arr = np.empty(total, dtype=np.int32)
        tfs_arr = np.empty(total, dtype=np.int32)
        df_arr = np.empty(n_terms, dtype=np.int64)
        cdef int32_t[::1] dv = docs_arr
        cdef int32_t[::1] fv = tfs_arr
        cdef long long[::1] df = df_arr
        cdef Py_ssize_t m

        with nogil:
            for i in range(n_terms):
                t = order[i]
                m = <Py_ssize_t>self.post_docs[t].size()
                df[i] = m
                for j in range(m):
                    dv[pos] = self.post_docs[t][j]
                    fv[pos] = self.post_tfs[t][j]
                    pos += 1
        return docs_arr, tfs_arr, df_arr
