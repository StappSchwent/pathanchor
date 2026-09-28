#!/usr/bin/env bash
# Fit the path scorer on the training split.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m pathanchor.cli.train \
  --config "${ROOT}/configs/experiment/main.yaml" \
  --report "${ROOT}/reports/train.txt" \
  "$@"
