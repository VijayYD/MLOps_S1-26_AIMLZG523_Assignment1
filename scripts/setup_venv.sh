#!/usr/bin/env bash
# Create .venv with the Python version pinned for this OS, and install
# requirements-dev.txt (superset of requirements.txt) into it.
#
# Interpreter resolution order:
#   1. $HEART_MLOPS_PYTHON, if set (explicit override).
#   2. 'python3' on PATH, if its major.minor matches the pin for this OS.
#   3. A pyenv-managed install matching the pin for this OS, if pyenv is
#      available (handles machines where PATH python3 is the OS-provided
#      interpreter rather than the pinned one).
# This exists because PATH python3 is not guaranteed to be the pinned
# version on every machine (see docs/DECISIONS.md) -- picking the wrong
# interpreter here would silently break the Python-3.12/3.13 reproducibility
# story the whole project depends on.
set -euo pipefail

cd "$(dirname "$0")/.."

OS="$(uname -s)"
REQUIRED_PY="3.13"
[ "$OS" = "Linux" ] && REQUIRED_PY="3.12"

resolve_python() {
  if [ -n "${HEART_MLOPS_PYTHON:-}" ]; then
    echo "$HEART_MLOPS_PYTHON"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    if python3 --version 2>&1 | grep -q " ${REQUIRED_PY}\."; then
      command -v python3
      return 0
    fi
  fi

  if command -v pyenv >/dev/null 2>&1; then
    local match
    match="$(pyenv versions --bare 2>/dev/null | grep -E "^${REQUIRED_PY}\.[0-9]+$" | sort -V | tail -n1)"
    if [ -n "$match" ]; then
      echo "$(pyenv root)/versions/${match}/bin/python3"
      return 0
    fi
  fi

  return 1
}

PY_BIN="$(resolve_python)" || {
  echo "ERROR: could not find a Python ${REQUIRED_PY}.x interpreter." >&2
  echo "  - PATH python3 is: $(command -v python3 2>/dev/null || echo 'not found') ($(python3 --version 2>&1 || true))" >&2
  echo "  - Install it, e.g. 'pyenv install ${REQUIRED_PY}.5' (macOS) or use your OS package manager (Linux)," >&2
  echo "    or set HEART_MLOPS_PYTHON=/path/to/python3 to point at one explicitly." >&2
  exit 1
}

echo "Using interpreter: $PY_BIN ($("$PY_BIN" --version 2>&1))"

"$PY_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-dev.txt

echo
echo "=== pip list ==="
pip list
