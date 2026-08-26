# Submission Interface Contract

This is the canonical, binding version of the interface referenced in the
assignment spec (Section 5, "Submission Interface & Conformance
Checking"): *"exact signature given in the starter repo."* This file is
that reference.

## What the harness calls

Exactly three functions, in `submission/retrieve.py`, in this order —
and, critically, **not in the same process**:

```python
def build_index(corpus_path: str, index_dir: str) -> None:
    """Called once, in its own process. Build the index and WRITE IT to
    index_dir; nothing held only in memory here survives into
    load_index()."""

def load_index(index_dir: str) -> None:
    """Called once, in a fresh process, before any retrieve() calls.
    Reconstruct everything retrieve() needs by reading index_dir, and
    only index_dir."""

def retrieve(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Called once per query, after load_index() has run in the same
    process."""
```

Nothing else about your submission is inspected by the harness — how you
organise `boolean_vsm.py`, `bm25.py`,
`custom_scorer.py`, or `indexer.py` internally, what you name your helper
functions, or how you structure your index, is entirely up to you, as
long as `retrieve.py` exposes these three entrypoints with these exact
names and this parameter order.

## Why two separate processes, not two function calls

`harness/run_harness.py` spawns `build_index()` and
`load_index()`/`retrieve()` as two genuinely separate subprocess
invocations of itself. This is deliberate, and it is the actual grading
mechanism, not a test-only detail: it is the only way to prove
`load_index()` really reconstructs everything from `index_dir` rather
than quietly relying on module-level state left over from `build_index()`
having run earlier in the same process. If your `load_index()` is
incomplete, it will fail or score badly in the query-phase subprocess —
there is nothing left in memory from the build phase to paper over the
gap. See the module docstring in `harness/run_harness.py` for the exact
mechanics.

## Contract details

**`build_index(corpus_path, index_dir)`**
- `corpus_path` is a path to a `corpus.jsonl` file (see `data/README.md`
  for the format).
- `index_dir` is a directory you should write your index into. It exists
  when `build_index()` is called; write whatever files you need under it.
- Called exactly once per harness run, before `load_index()`.
- Its wall-clock time is measured and reported as your index-build-time
  efficiency metric — do expensive one-time work here, not in `retrieve()`.
- The **on-disk byte size of `index_dir` after this returns** is measured
  and is its own graded leaderboard component (assignment Section 7,
  "index size," 0-10% of your score, relative to the class median). Write
  only what `retrieve()` actually needs, and consider compressing it —
  see `submission/indexer.py`'s `InvertedIndex.save()` docstring for
  concrete starting points.
- Must not require network access or any file other than `corpus_path`.

**`load_index(index_dir)`**
- Called exactly once per harness run, in a **fresh process** (no
  leftover state from `build_index()`), before any `retrieve()` call.
- Must reconstruct everything `retrieve()` needs by reading only
  `index_dir` — the same directory `build_index()` wrote to.
- Its wall-clock time is measured and reported as your index-load-time
  metric.
- Must not require network access, must not read `corpus_path` (it is not
  passed to `load_index()`), and must not depend on anything other than
  `index_dir`'s contents.

**`retrieve(query, k=10)`**
- `query` is a raw query string (not pre-tokenised).
- `k` is the number of results requested (the harness always passes an
  explicit value; the default is only there so you can call `retrieve()`
  manually while testing).
- Must return a `list` of `(doc_id, score)` tuples:
  - `doc_id` must be a `doc_id` value that appeared in the corpus passed
    to `build_index()`.
  - `score` must be numeric (`int` or `float`); higher = more relevant.
  - The list must have length `<= k`.
  - **No `doc_id` may appear more than once in the list.** This is
    checked and rejected (`RUNTIME_ERROR`), not silently deduplicated —
    a run that repeats one doc_id lets that document's relevance get
    counted multiple times, which would push nDCG@10/MAP@10 above their
    theoretical maximum of 1.0 (see `tests/test_metrics.py` for the exact
    numbers). If you ever see this error, it means your ranking logic
    produced the same document twice, not that duplicates are a minor
    style issue.
  - The harness re-sorts by score descending defensively, but you should
    return it already sorted — an unsorted return is a strong signal
    something is wrong.
- Must be deterministic: the same query, called twice against the same
  index, should return the same ranking. (Ties in score may break
  arbitrarily but consistently.)
- Must not read `corpus_path` (it is not passed to `retrieve()`), hit the
  network, or depend on global state from a previous `retrieve()` call
  beyond what `load_index()` set up.

## Compiled extensions (C/C++ via Cython/pybind11)

You do not have to write pure Python. `build_index`, `load_index`, and
`retrieve` just need to be plain Python-callable functions in
`submission/retrieve.py` — what runs underneath is your choice, including
a C/C++ extension you compile yourself and call into. `Cython` is already
in `requirements.txt` for this; `pybind11` works too if you add it
yourself.

- **Provide `submission/setup.py`.** If you use a compiled extension,
  define it in this file. The provided Dockerfile, CI workflow, and course
  staff's grading image detect the file and run
  `python setup.py build_ext --inplace` from inside `submission/`.
- **Build it at image-build time, not inside `build_index()`.** The
  student and grading images include a C/C++ toolchain (`build-essential`)
  and compile the extension while the image is built. The grading
  container runs with `--network none` at scoring time, and
  anything that happens inside `build_index()` is charged against your
  index-build-time efficiency metric — a one-time compile isn't indexing
  work and shouldn't be billed as such.
- **This is not a loophole around "no existing search library"**
  (assignment Section 4.1/10). Your indexing and scoring logic still has
  to be your own work; the rule is about not importing Lucene/
  Elasticsearch/Pyserini/Whoosh, not about which language you wrote your
  own implementation in.
- **Expect a real efficiency/index-size advantage from this**, relative
  to an equivalent pure-Python submission — that's expected, not a
  scoring bug (assignment Section 7).

## What "conformance" means in practice

`tests/test_interface_conformance.py` is the literal check the CI job
(`.github/workflows/conformance.yml`) runs on every push, and the same
checks the harness performs before scoring your submission for real:

1. `submission.retrieve` exposes `build_index`, `load_index`, and
   `retrieve` with the right signatures (checked via `inspect.signature`,
   not just `hasattr`).
2. `build_index()` actually writes something to `index_dir`.
3. `load_index()` then `retrieve()`, run in a genuinely separate
   subprocess from `build_index()`, work on the toy corpus without
   raising — this is the real proof that persistence works, not an
   in-process simulation of it.
4. Every `retrieve()` result is well-formed: a list of `(doc_id, score)`
   pairs, length `<= k`, no repeated `doc_id`, sorted descending, all
   `doc_id`s valid, all `score`s numeric.
5. Nothing pathologically slow happens on a 20-document toy corpus
   (catches accidental quadratic blowups early, before they bite you on
   the real corpus).

None of this checks ranking *quality* — a submission that always returns
an empty list passes conformance trivially and scores 0 on every metric.
Conformance is a floor, not a target.

## Conformance freeze

48 hours before the assignment deadline, your repository's CI conformance
check must be green. Interface problems reported for the first time after
the freeze are graded as a submission error (assignment Section 9,
Grading Rubric — "Interface conformance"), not treated as a harness bug.
Run `scripts/smoke_test.sh` locally any time before then.
