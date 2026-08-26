# Submitting as a Docker Container

You do not submit a Docker image directly — you submit your repository,
exactly as described in the main README. This document explains what the
`Dockerfile` in the root of this repo is for, how to use it to test
yourself before pushing, and — plainly, so there's no ambiguity — how
course staff actually use it (and don't use it) when grading.

## What the Dockerfile is for

It pins the exact Python version and dependency versions your submission
runs under, so "works on my machine" cannot cause a harness failure. It
also includes a C/C++ toolchain (`build-essential`), so if you compile
part of your submission as a Cython or pybind11 extension (see
`docs/SUBMISSION_INTERFACE.md`, "Compiled extensions"), it builds
correctly here too. Put its build definition in `submission/setup.py`;
the student image, CI, and course staff's grading image all run
`python setup.py build_ext --inplace` from that directory. Build and run
it locally before you push, especially close to the conformance freeze:

```bash
docker build -t my-a1-submission .
docker run --rm my-a1-submission
```

The default command runs the full harness against the toy set (see the
`CMD` at the bottom of the `Dockerfile`) and prints the same report you'd
get running `bash scripts/smoke_test.sh` directly. If it works in the
container, it will work in CI and at grading time — that's the entire
point.

To run it against the real assignment corpus instead of the toy set,
mount your `data/full/` directory and override the command:

```bash
docker run --rm -v "$(pwd)/data/full:/repo/data/full" my-a1-submission \
  python -m harness.run_harness \
  --corpus data/full/corpus.jsonl \
  --queries data/full/queries_dev.tsv \
  --qrels data/full/qrels_dev.txt \
  --run-out runs/dev_run.trec --report-out runs/dev_report.json
```

## What course staff actually run at grading time

Plainly, so you know exactly what's being trusted and what isn't: **your
copy of `harness/` is never executed for your real grade**, whether or
not you use Docker. Course infrastructure builds a *separate* image from
a *separate* Dockerfile that copies in course staff's own trusted copy of
`harness/` and only your `submission/` directory — nothing else from your
repo (including your own `Dockerfile`, `harness/`, or `tests/`) is part
of the image that produces your score. If `submission/setup.py` exists,
the grading image builds it in place before scoring. The private held-out
corpus/queries/qrels are mounted read-only at grading time and are never
baked into any image or committed anywhere you have access to; the
grading container also runs with networking disabled.

This isn't a secret mechanism — it's described here on purpose. It means
there is no version of "quietly adjust the scoring code" that does
anything except cost you the oral defense (Section 7.1 of the assignment
spec), where you'll be asked to explain your own submission live. Put
your effort into `submission/`, which is the only thing that's actually
graded.

## Conformance freeze reminder

Same rule as the rest of the assignment: `docker build` and `docker run`
against your own repo should succeed cleanly by 48 hours before the
deadline (docs/SUBMISSION_INTERFACE.md). If your image doesn't build,
CI's conformance job won't pass either — they run the same checks.
