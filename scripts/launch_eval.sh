#!/usr/bin/env bash
# Score the held-out test split and run the reported panels.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m pathanchor.cli.evaluate \
  --config "${ROOT}/configs/experiment/main.yaml" \
  --report "${ROOT}/reports/test_split.txt"
"${PYTHON_BIN}" -m pathanchor.cli.ablate \
  --config "${ROOT}/configs/experiment/ablation_component_panel.yaml" \
  --panel panel_a --report "${ROOT}/reports/panel_a.txt"
"${PYTHON_BIN}" -m pathanchor.cli.ablate \
  --config "${ROOT}/configs/experiment/ablation_leave_one_out.yaml" \
  --panel panel_b --report "${ROOT}/reports/panel_b.txt"
