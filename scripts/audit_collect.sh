#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-results/mt_lowpx_core}"
CFG_PATH="${2:-configs/settings.yaml}"
OUT_ROOT="results/audits"
mkdir -p "${OUT_ROOT}"
python scripts/audit_collect.py --run-dir "${RUN_DIR}" --config "${CFG_PATH}" --out-dir "${OUT_ROOT}"

