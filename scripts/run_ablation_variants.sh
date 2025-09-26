#!/usr/bin/env bash
set -euo pipefail

# Quick RL ablations (300k‑step probes)
# Usage:
#   ./scripts/run_ablation_variants.sh <screen_run_name> <ablate_run_name> [TIMESTEPS] [SEED]
# Example:
#   ./scripts/run_ablation_variants.sh feature_screen_mt_core ablate_mt_core 300000 123

SCREEN_RUN="${1:-feature_screen_mt_core}"
ABLAT_RUN="${2:-ablate_mt_core}"
TIMESTEPS="${3:-300000}"
SEED="${4:-123}"

CFG="configs/settings.yaml"

# Generate curated list (top‑N) if not present
CONS="results/features/${SCREEN_RUN}/consensus_importance.parquet"
if [[ ! -f "$CONS" ]]; then
  echo "Consensus importance not found at $CONS. Run feature_screen.py first." >&2
  exit 2
fi
./scripts/make_curated_from_consensus.sh "$SCREEN_RUN" 40

# Variants: curated and curated minus each pack
VARIANTS=(
  "curated"
  "curated_minus:microstructure"
  "curated_minus:context"
  "curated_minus:price_vol"
  "curated_minus:crosssec"
)

for V in "${VARIANTS[@]}"; do
  echo "[abl] $V"
  PYTHONPATH=. python scripts/rl_ablate.py \
    --config "$CFG" \
    --run-name "$SCREEN_RUN" \
    --feature-pack "$V" \
    --timesteps "$TIMESTEPS" \
    --seed "$SEED"
done

echo "Done. See results/ablations/${SCREEN_RUN}/ for summaries."

