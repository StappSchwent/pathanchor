#!/usr/bin/env bash
# The construction chain reads its sources from their own distribution points
# (see dataset_urls.txt) and the caller-supplied guideline and external profile
# tables. This script only creates the directories the reader writes into; the
# cohort and the guideline text are never bundled.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/data/graph" "${ROOT}/data/external" "${ROOT}/reports"
echo "Place a licensed guideline statement table at ${ROOT}/data/guideline/statements.yaml"
echo "Place a harmonised external profile table at ${ROOT}/data/external/profiles.tsv"
