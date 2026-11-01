#!/usr/bin/env bash
# Regenerate the R reference fixtures that are written by redirecting a
# generator's stdout (`Rscript generator.R > tests/fixtures/x.rs`), into
# OUT_DIR (default: in place). Used by the scheduled `validation` workflow,
# which diffs the result against the committed fixtures.
#
#   validation/regenerate_r_fixtures.sh            # in place
#   OUT_DIR=/tmp/fx validation/regenerate_r_fixtures.sh
set -euo pipefail
cd "$(dirname "$0")/.."
out="${OUT_DIR:-tests/fixtures}"
mkdir -p "$out"
status=0
while read -r script fixture _tolerance; do
  [[ -z "$script" || "$script" == \#* ]] && continue
  target="$out/$(basename "$fixture")"
  if ! Rscript "$script" > "$target.tmp" 2> "$target.log"; then
    echo "FAILED: $script (see $target.log)" >&2
    status=1
    continue
  fi
  mv "$target.tmp" "$target"
  rm -f "$target.log"
done < validation/r_fixtures.txt
exit "$status"
