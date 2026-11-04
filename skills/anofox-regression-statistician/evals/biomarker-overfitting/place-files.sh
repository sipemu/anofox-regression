#!/usr/bin/env bash
set -eu
cp -R "$(dirname "$0")/files/." .
"$(dirname "$0")/../_offline-cargo.sh"
