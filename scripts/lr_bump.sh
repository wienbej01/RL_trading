#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-./results}"
touch "${RUN_DIR}/.lr_bump"
echo "Queued a live LR bump in ${RUN_DIR}"

