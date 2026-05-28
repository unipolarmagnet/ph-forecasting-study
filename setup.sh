#!/usr/bin/env bash
# One-time setup on a Linux machine:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
echo "Using $($PY --version)"
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Setup done.  activate: source .venv/bin/activate   then: bash run.sh"
