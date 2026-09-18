#!/usr/bin/env bash
# Local mirror of .github/workflows/ci.yml gates that can run on this machine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> fmt"
cargo fmt --all --check

echo "==> clippy (cpu)"
cargo clippy --all-targets -- -D warnings

echo "==> test release (cpu, all targets + doc)"
cargo test --release --all-targets
cargo test --release --doc

echo "==> clippy + suite (metal)"
cargo clippy --features metal --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --features metal
MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 \
  cargo test --features metal --release --all-targets -- --test-threads=1

echo "==> examples (metal)"
cargo run --release --features metal --example readme

echo "==> release-ready gate (requires clean tree + version triad)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "note: skipping check_release_ready.sh while tree is dirty"
else
  ./scripts/check_release_ready.sh
fi

echo "ci:local OK"
