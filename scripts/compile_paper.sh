#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LATEX_DIR="$ROOT_DIR/latex"

TECTONIC_BIN="${TECTONIC_BIN:-tectonic}"
if ! command -v "$TECTONIC_BIN" >/dev/null 2>&1; then
  if [ -x /opt/homebrew/bin/tectonic ]; then
    TECTONIC_BIN="/opt/homebrew/bin/tectonic"
  else
    echo "tectonic not found. Install it or set TECTONIC_BIN." >&2
    exit 1
  fi
fi

cd "$LATEX_DIR"
"$TECTONIC_BIN" --keep-logs --keep-intermediates sn-article.tex
