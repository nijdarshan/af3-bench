#!/usr/bin/env bash
set -euo pipefail
# Resolve script directory robustly when sourced or executed
_src_path="${BASH_SOURCE[0]:-$0}"
cd "$(cd "$(dirname "$_src_path")" && pwd)/.."

# helper: exit safely when sourced
_exit_or_return() {
  if [[ "${BASH_SOURCE[0]-}$0" != "$0" ]]; then
    return "$1"
  else
    exit "$1"
  fi
}

# activate venv
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate || _exit_or_return 1
else
  echo "No .venv found. Run: python3 -m venv .venv && source .venv/bin/activate"
  _exit_or_return 1
fi

# point PyO3 at this venv's Python
export PYO3_PYTHON="$PWD/.venv/bin/python"

# make Python see our shims + both repos
_tp="$PWD/third_party"
_py="$PWD/py"
_paths=(
  "$_py"
  "$PWD/third_party/alphafold3-deepmind"
  "$PWD/third_party/alphafold3-ligo"
  "$PWD/third_party/AlphaFold3"
  "$PWD/third_party/alphafold3"
)
_joined=$(IFS=:; echo "${_paths[*]}")
export PYTHONPATH="$_joined"

# XLA niceties (helps avoid large preallocation and caps fraction)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85

echo "env ready."
# Avoid set -e aborts if commands are missing
_py_bin=$(command -v python || true)
_cargo_bin=$(command -v cargo || true)
echo "  PYTHON     = ${_py_bin:-'(not found)'}"
echo "  CARGO      = ${_cargo_bin:-'(not found)'}"
echo "  PYO3_PYTHON= $PYO3_PYTHON"
echo "  PYTHONPATH = $PYTHONPATH"
echo "  XLA_PREALLOCATE = $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "  XLA_MEM_FRACTION= $XLA_PYTHON_CLIENT_MEM_FRACTION"

