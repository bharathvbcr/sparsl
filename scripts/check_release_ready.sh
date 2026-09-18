#!/usr/bin/env bash
# Fail closed before tag / cargo publish / GitHub Release.
# Catches version triad drift, dirty trees, and oversized/stale package contents.
# Portable: uses python3 + grep/sed only (no ripgrep) so GitHub runners work.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

crate="$(sed -n 's/^name = "\(.*\)"/\1/p' Cargo.toml | head -1)"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
if [[ -z "$crate" || -z "$version" ]]; then
  echo "error: could not parse name/version from Cargo.toml" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is dirty; commit or stash before release" >&2
  git status --porcelain >&2
  exit 1
fi

readme_ver="$(
  python3 - <<'PY'
import pathlib, re, sys
text = pathlib.Path("README.md").read_text()
m = re.search(r"\*\*Status\*\* \| \[`([0-9]+\.[0-9]+\.[0-9]+)`\]", text)
sys.stdout.write(m.group(1) if m else "")
PY
)"
if [[ "$readme_ver" != "$version" ]]; then
  echo "error: README Status is '${readme_ver:-<missing>}' but Cargo.toml is '$version'" >&2
  exit 1
fi

changelog_ver="$(
  python3 - <<'PY'
import pathlib, re, sys
text = pathlib.Path("CHANGELOG.md").read_text()
m = re.search(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\]", text, re.M)
sys.stdout.write(m.group(1) if m else "")
PY
)"
if [[ "$changelog_ver" != "$version" ]]; then
  echo "error: newest CHANGELOG version is '${changelog_ver:-<missing>}' but Cargo.toml is '$version'" >&2
  exit 1
fi

if ! grep -q "^\[${version}\]:" CHANGELOG.md; then
  echo "error: CHANGELOG.md missing footer link [$version]:" >&2
  exit 1
fi

if [[ -f Cargo.lock ]]; then
  lock_ver="$(
    CRATE="$crate" python3 - <<'PY'
import os, pathlib, re, sys
text = pathlib.Path("Cargo.lock").read_text()
crate = os.environ["CRATE"]
m = re.search(rf'^name = "{re.escape(crate)}"\nversion = "([^"]+)"', text, re.M)
sys.stdout.write(m.group(1) if m else "")
PY
  )"
  if [[ -n "$lock_ver" && "$lock_ver" != "$version" ]]; then
    echo "error: Cargo.lock has $crate $lock_ver but Cargo.toml is $version" >&2
    exit 1
  fi
fi

echo "==> cargo package --locked --list ($crate $version)"
cargo package --locked --list >"/tmp/${crate}-package.list"
if grep -E '\.devmap/|@2048\.png|build_logo\.py|assets/concepts/' "/tmp/${crate}-package.list"; then
  echo "error: package list contains excluded/stale paths (listed above)" >&2
  exit 1
fi

echo "==> cargo package --locked (verify build)"
cargo package --locked >/tmp/"${crate}-package.log" 2>&1

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "==> DOCS_RS=1 cargo check (docs.rs AOT skip)"
  DOCS_RS=1 cargo check --locked >/tmp/"${crate}-docsrs.log" 2>&1 || {
    echo "error: DOCS_RS=1 cargo check failed; see /tmp/${crate}-docsrs.log" >&2
    exit 1
  }
fi

echo "release-ready: $crate $version"
