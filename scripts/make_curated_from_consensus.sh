#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/make_curated_from_consensus.sh <run_name> [TOP_N]
# Example:
#   ./scripts/make_curated_from_consensus.sh feature_screen_mt_core 40

RUN_NAME="${1:-feature_screen_mt_core}"
TOPN="${2:-40}"

BASE="results/features/${RUN_NAME}"
CONS="${BASE}/consensus_importance.parquet"
OUT="${BASE}/curated_top${TOPN}.txt"

if [[ ! -f "${CONS}" ]]; then
  echo "Consensus file not found: ${CONS}" >&2
  exit 1
fi

python - "$CONS" "$OUT" "$TOPN" <<'PY'
import sys
import pandas as pd

cons, out, topn = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_parquet(cons)

# Prefer 'consensus'; fall back to first available z_* or imp_*
col = 'consensus' if 'consensus' in df.columns else None
if col is None:
    for c in df.columns:
        if c.startswith('z_') or c.startswith('imp_'):
            col = c
            break
if col is None:
    raise SystemExit(f"No consensus/importance column in {cons}; cols={list(df.columns)}")

df[['feature', col]].dropna(subset=['feature']).sort_values(col, ascending=False).head(topn)['feature'].to_csv(out, index=False, header=False)
print(out)
PY

echo "Wrote ${OUT}"

