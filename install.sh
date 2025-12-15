#!/usr/bin/env bash
set -euo pipefail

# This script bootstraps the project using uv, installing the managed Python,
# dependencies described in pyproject.toml, and GPU-aware PyTorch wheels.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="3.10.12"
DEFAULT_INSTALLER_URL="https://astral.sh/uv/install.sh"

# Ensure `~/.local/bin` is considered when checking for uv.
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  INSTALLER_URL="${UV_INSTALLER_URL:-$DEFAULT_INSTALLER_URL}"
  echo "uv not found on PATH; installing from ${INSTALLER_URL}..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "${INSTALLER_URL}" | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "${INSTALLER_URL}" | sh
  else
    echo "Neither curl nor wget is available; please install one and re-run." >&2
    exit 1
  fi
  hash -r
fi

cd "${PROJECT_ROOT}"

# Make sure the pinned interpreter is available so `uv sync` does not prompt.
if ! uv python list --only-installed | grep -q "${PYTHON_VERSION}"; then
  echo "Installing Python ${PYTHON_VERSION} via uv..."
  uv python install "${PYTHON_VERSION}"
fi

# Create the project venv (uses the uv-managed Python above).
uv venv
uv pip install -e . --torch-backend=auto -p .venv/bin/python

# Loosen transformers' hub pin so hub 1.x works.
.venv/bin/python -c "import transformers.dependency_versions_table as t;from pathlib import Path;p=Path(t.__file__);p.write_text(p.read_text().replace('huggingface-hub>=0.34.0,<1.0','huggingface-hub>=0.34.0'))"

echo "Environment ready. Activate it with 'source .venv/bin/activate' when needed."
echo "By default wandb logging is enabled, remember to run 'wandb init' before training."
