# Pinned, minimal image so local testing matches the grading machine's
# toolchain (assignment Section 5, "Containerisation": Ubuntu 24.04.3 LTS,
# Python 3.12.3, GCC/G++ 13.3.0). Course staff never build or run this
# Dockerfile itself for scoring -- their own separate image copies in only
# submission/ (see docs/DOCKER_SUBMISSION.md) -- this is purely so `docker
# build . && docker run` gives you a representative local check.
FROM python:3.12-slim

WORKDIR /repo

# C/C++ toolchain — present so a submission that compiles part of itself
# as a Cython or pybind11 extension (see docs/SUBMISSION_INTERFACE.md,
# "Compiled extensions") builds correctly here and in course staff's
# grading image (instructor-tools/Dockerfile.grading, kept in lockstep
# with this one). Installed at image-build time only; the grading
# container still runs with --network none at scoring time.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Compile the extension at image-build time, while network access is still
# available and where the cost is not charged to any graded metric. The
# assignment is explicit that build_index() must not do this: anything it does
# counts against the index-build-time efficiency score.
#
# Run from inside submission/ against submission/setup.py -- the exact location
# and working directory course staff's grading image uses. If this step were
# ever removed the submission still runs: submission/bm25.py imports the
# extension behind try/except and falls back to a pure-NumPy path.
RUN if [ -f submission/setup.py ]; then \
        cd submission && python setup.py build_ext --inplace; \
    fi

# Default command: run the interface conformance + smoke-test suite
# against the toy set. Course staff override CMD to point at the real
# corpus/topics/qrels for scoring.
CMD ["python", "-m", "harness.run_harness", \
     "--corpus", "data/toy/corpus.jsonl", \
     "--queries", "data/toy/queries_dev.tsv", \
     "--qrels", "data/toy/qrels_dev.txt", \
     "--baseline-run", "data/toy/reference_bm25_run_dev.trec", \
     "--run-out", "runs/dev_run.trec", \
     "--report-out", "runs/dev_report.json"]
