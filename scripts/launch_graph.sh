#!/usr/bin/env bash
# Build the released graph snapshot and its census.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m pathanchor.cli.build_graph \
  --config "${ROOT}/configs/experiment/main.yaml" \
  --out "${ROOT}/data/graph/snapshot.npz" \
  --report "${ROOT}/reports/graph_census.txt"
