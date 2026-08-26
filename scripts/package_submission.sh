#!/usr/bin/env bash
# Build the submission zip and prove the extracted copy runs unmodified.
#
# Assignment Instruction 3 is strict: the zip must be named <REGNO>.zip and must
# deflate to a single directory ./<REGNO>/ (uppercase) containing the required
# tree. A submission that fails this "might be rejected and not be evaluated",
# so the packaging is verified here rather than assumed on deadline day.
#
# `git archive` is used deliberately: it exports exactly the tracked tree at
# HEAD, so anything gitignored -- the 198MB corpus, virtualenvs, index caches,
# the raw sweep log -- cannot leak into the zip by accident. The assignment also
# says not to submit data files.
#
# Usage:
#     bash scripts/package_submission.sh 2024XXX0000
set -euo pipefail

REGNO="${1:-}"
if [[ -z "$REGNO" ]]; then
  echo "usage: bash scripts/package_submission.sh <REGNO>   (e.g. 2024CSZ8888)" >&2
  exit 2
fi
if [[ "$REGNO" != "$(printf '%s' "$REGNO" | tr '[:lower:]' '[:upper:]')" ]]; then
  echo "ERROR: registration number must be uppercase (got '$REGNO')" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
REPO="$PWD"
OUT="$REPO/dist"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "WARNING: working tree is dirty; the zip reflects HEAD, not your edits." >&2
  git status --short >&2
  echo >&2
fi

mkdir -p "$OUT"
ZIP="$OUT/$REGNO.zip"
git archive --format=zip --prefix="$REGNO/" HEAD -o "$ZIP"
echo "built $ZIP"

echo
echo "== verifying the extracted copy =="
unzip -q "$ZIP" -d "$WORK"

top_level="$(find "$WORK" -mindepth 1 -maxdepth 1 -printf '%f\n' 2>/dev/null || ls -1 "$WORK")"
if [[ "$top_level" != "$REGNO" ]]; then
  echo "ERROR: zip must deflate to exactly one directory named $REGNO; got: $top_level" >&2
  exit 1
fi

missing=0
for f in assignment1.tex conftest.py data/README.md data/toy Dockerfile \
         docs/DOCKER_SUBMISSION.md docs/SUBMISSION_INTERFACE.md harness \
         README.md requirements.txt runs scripts submission tests \
         submission/setup.py; do
  if [[ -e "$WORK/$REGNO/$f" ]]; then
    echo "  OK   $f"
  else
    echo "  MISS $f" >&2
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || { echo "ERROR: required entries missing" >&2; exit 1; }

# Nothing large should have slipped in. The corpus alone is ~190MB.
if find "$WORK/$REGNO" -type f -size +5M | grep -q .; then
  echo "ERROR: files over 5MB found in the archive:" >&2
  find "$WORK/$REGNO" -type f -size +5M -exec ls -lh {} \; >&2
  exit 1
fi

# Course staff are explicit: do not commit a precompiled .so, and do not
# compile inside build_index(). The archive must ship sources only.
if find "$WORK/$REGNO" -name '*.so' -o -name '*.pyd' | grep -q .; then
  echo "ERROR: precompiled binaries found in the archive:" >&2
  find "$WORK/$REGNO" \( -name '*.so' -o -name '*.pyd' \) >&2
  exit 1
fi
echo "  OK   no precompiled binaries in the archive"

# Build the extension the way the grading image does -- from inside submission/,
# against submission/setup.py. Doing it here, on the EXTRACTED copy, is the only
# way to prove the archive is self-sufficient: the working tree's .so files are
# gitignored and therefore absent from the zip, so if this step fails, grading
# would silently fall back to the slow pure-Python path.
if [[ -f "$WORK/$REGNO/submission/setup.py" ]]; then
  echo
  echo "== building the compiled extension from the extracted copy =="
  ( cd "$WORK/$REGNO/submission" && python setup.py build_ext --inplace >/dev/null 2>&1 ) \
    && echo "  OK   extension built from the archive" \
    || { echo "ERROR: submission/setup.py failed to build from the archive" >&2; exit 1; }
  ( cd "$WORK/$REGNO" && python -c "
from submission import bm25
assert bm25.HAVE_FAST, 'extension built but not picked up by bm25'
import submission.indexer as ix
assert ix._FASTBUILD is not None, 'build kernel not picked up by indexer'
print('  OK   both kernels import and are active')
" ) || exit 1
fi

echo
echo "== running the shipped smoke test from the extracted copy =="
( cd "$WORK/$REGNO" && bash scripts/smoke_test.sh )

echo
echo "PACKAGED AND VERIFIED: $ZIP  ($(du -h "$ZIP" | cut -f1))"
echo "Upload this file to Moodle."
