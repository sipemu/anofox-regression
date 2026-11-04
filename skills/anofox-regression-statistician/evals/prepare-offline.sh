#!/usr/bin/env bash
# Run once before `claude plugin eval` (with network): vendor the analysis
# template's crates.io dependencies into the repository's target/ (git-ignored,
# outside the skill), where every case's scaffold copies them from
# (_offline-cargo.sh). The eval sandbox has no network and no ~/.cargo.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
template="$here/../assets/analysis-template"
key="$(sha256sum "$template/Cargo.toml" | cut -c1-16)"
cache="$here/../../../target/skill-eval-vendor/$key"
[ -d "$cache/vendor" ] && { echo "vendor cache ready: $cache"; exit 0; }
tmp="$(mktemp -d)"
cp -R "$template/." "$tmp/"
rm -rf "$tmp/target" "$tmp/Cargo.lock"
mkdir -p "$cache"
(cd "$tmp" && cargo generate-lockfile -q && cargo vendor -q --versioned-dirs "$cache/vendor.partial" >/dev/null)
mv "$cache/vendor.partial" "$cache/vendor"
rm -rf "$tmp"
echo "vendor cache ready: $cache"
