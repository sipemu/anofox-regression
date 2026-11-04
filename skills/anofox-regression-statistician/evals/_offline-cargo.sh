#!/usr/bin/env bash
# Scaffold helper: the eval sandbox has no network and hides ~/.cargo, so copy
# the vendored crates.io sources (made by prepare-offline.sh) into the case's
# working directory and point cargo at them with .cargo/config.toml, which
# cargo finds from any project below it.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
key="$(sha256sum "$here/../assets/analysis-template/Cargo.toml" | cut -c1-16)"
cache="$here/../../../target/skill-eval-vendor/$key"
if [ ! -d "$cache/vendor" ]; then
  echo "no vendored crates: run skills/anofox-regression-statistician/evals/prepare-offline.sh first" >&2
  exit 1
fi
cp -R "$cache/vendor" ./vendor
mkdir -p .cargo
cat > .cargo/config.toml <<CFG
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "$PWD/vendor"

[net]
offline = true
CFG
