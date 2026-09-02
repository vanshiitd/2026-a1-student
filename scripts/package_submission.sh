#!/usr/bin/env bash
# builds the submission zip + checks the extracted copy actually runs.
# zip has to deflate to ./REGNO/ with the exact required tree or it might
# not even get evaluated
#
# usage:
#     bash scripts/package_submission.sh 2024XXX0000
set -euo pipefail

REQUIRED_PATHS=(assignment1.tex conftest.py data Dockerfile docs harness
                README.md requirements.txt runs scripts submission tests)

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
# git archive so gitignored stuff (corpus, venvs, index caches) can't leak in
git archive --format=zip --prefix="$REGNO/" -o "$ZIP" HEAD -- "${REQUIRED_PATHS[@]}"
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
for f in "${REQUIRED_PATHS[@]}" data/README.md data/toy \
         docs/DOCKER_SUBMISSION.md docs/SUBMISSION_INTERFACE.md \
         submission/setup.py; do
  if [[ -e "$WORK/$REGNO/$f" ]]; then
    echo "  OK   $f"
  else
    echo "  MISS $f" >&2
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || { echo "ERROR: required entries missing" >&2; exit 1; }

# nothing extra either, catches forgetting to update this list
extra=0
for f in "$WORK/$REGNO"/*; do
  name="$(basename "$f")"
  found=0
  for req in "${REQUIRED_PATHS[@]}"; do
    [[ "$name" == "$req" ]] && found=1 && break
  done
  if [[ "$found" -eq 0 ]]; then
    echo "  EXTRA $name (not in REQUIRED_PATHS)" >&2
    extra=1
  fi
done
[[ "$extra" -eq 0 ]] || { echo "ERROR: unexpected top-level entries in the archive" >&2; exit 1; }
echo "  OK   no top-level entries beyond the required tree"

if find "$WORK/$REGNO" -type f -size +5M | grep -q .; then
  echo "ERROR: files over 5MB found in the archive:" >&2
  find "$WORK/$REGNO" -type f -size +5M -exec ls -lh {} \; >&2
  exit 1
fi

if find "$WORK/$REGNO" -name '*.so' -o -name '*.pyd' | grep -q .; then
  echo "ERROR: precompiled binaries found in the archive:" >&2
  find "$WORK/$REGNO" \( -name '*.so' -o -name '*.pyd' \) >&2
  exit 1
fi
echo "  OK   no precompiled binaries in the archive"

# building from the extracted copy proves the archive is self sufficient
# (working tree's .so is gitignored so it's never in the zip anyway)
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
