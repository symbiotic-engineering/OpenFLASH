#!/usr/bin/env bash
set -euo pipefail
# Usage: create_rc_tag.sh BASE PR_NUM

base="$1"
pr_num="$2"

git fetch --tags --no-recurse-submodules
# Use underscores in RC tags to avoid issues with conda metadata parsing
# pattern example: v1.2.3_rc1_pr45
pattern="${base}_rc*_pr${pr_num}"
highest=$(git tag --list "$pattern" --sort=-v:refname | head -n1 || true)
if [ -z "$highest" ]; then
  rc=1
else
  rc=$(echo "$highest" | sed -E 's/.*_rc([0-9]+)_pr[0-9]+$/\1/')
  rc=$((rc + 1))
fi
newtag="${base}_rc${rc}_pr${pr_num}"

echo "${newtag}"
