#!/usr/bin/env bash
set -euo pipefail
# Usage: compute_base.sh [PR_TITLE]
PR_TITLE="${1:-}"

git fetch --tags --no-recurse-submodules
last_tag=$(git tag --list 'v*.*.*' --sort=-version:refname | grep -v rc | head -n1 || true)
if [ -z "$last_tag" ]; then
  last_tag="v0.0.0"
fi
part="${last_tag#v}"
major=$(echo "$part" | cut -d. -f1)
minor=$(echo "$part" | cut -d. -f2)
patch=$(echo "$part" | cut -d. -f3)

lower_title=$(echo "$PR_TITLE" | tr '[:upper:]' '[:lower:]')
if echo "$lower_title" | grep -q 'major'; then
  major=$((major + 1))
  minor=0
  patch=0
elif echo "$lower_title" | grep -q 'minor'; then
  minor=$((minor + 1))
  patch=0
else
  patch=$((patch + 1))
fi

base="v${major}.${minor}.${patch}"
echo "$base"
