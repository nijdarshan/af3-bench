#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# activate venv
if [[ -d .venv ]]; then
  source .venv/bin/activate
else
  echo "No .venv found. Run: python3 -m venv .venv && source .venv/bin/activate"
  exit 1
fi

# point PyO3 at this venv's Python
export PYO3_PYTHON="$PWD/.venv/bin/python"

# make Python see our shims + both repos
export PYTHONPATH="$PWD/py:$PWD/third_party/alphafold3-deepmind:$PWD/third_party/alphafold3-ligo"

echo "env ready."
echo "  PYTHON     = $(which python)"
echo "  CARGO      = $(which cargo)"
echo "  PYO3_PYTHON= $PYO3_PYTHON"
echo "  PYTHONPATH = $PYTHONPATH"

