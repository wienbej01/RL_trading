#!/usr/bin/env bash
set -euo pipefail

# Static low-priced universe ($10–$20 approx; edit as needed)
UNIVERSE_FILE=${UNIVERSE_FILE:-scripts/universe_lowpx_10_20.txt}
if [[ ! -f "$UNIVERSE_FILE" ]]; then
  echo "Universe file not found: $UNIVERSE_FILE" >&2
  exit 1
fi

# Dates (edit as needed)
TRAIN_START=${TRAIN_START:-2024-04-01}
TRAIN_END=${TRAIN_END:-2024-08-15}
TEST_START=${TEST_START:-2024-08-16}
TEST_END=${TEST_END:-2024-09-30}

# Output and steps
OUT_DIR=${OUT_DIR:-results/mt_lowpx_core}
MAX_STEPS=${MAX_STEPS:-3000000}
SEED=${SEED:-123}

# Require Polygon key for collection
if [[ -z "${POLYGON_API_KEY:-}" ]]; then
  echo "POLYGON_API_KEY is not set. Export it and rerun." >&2
  exit 2
fi

TICKERS=$(tr '\n' ',' < "$UNIVERSE_FILE" | sed 's/,$//')
echo "[info] Universe: $TICKERS"

PY=python

echo "[1/3] Collecting minute aggregates via Polygon..."
$PY scripts/collect_polygon_us_stocks.py \
  --tickers "$TICKERS" \
  --start-date "$TRAIN_START" \
  --end-date "$TEST_END" \
  --types aggregates \
  --concurrency 20

echo "[2/3] Aggregating to single Parquet per ticker..."
IFS="," read -ra ARR <<< "$TICKERS"
for tk in "${ARR[@]}"; do
  $PY scripts/aggregate_us_stock_data.py \
    --ticker "$tk" \
    --start-date "$TRAIN_START" \
    --end-date "$TEST_END"
done

echo "[3/3] Training + backtesting (portfolio env)..."
PYTHONPATH=. $PY scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --tickers ${TICKERS//,/ } \
  --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
  --test-start  "$TEST_START"  --test-end  "$TEST_END" \
  --output-dir "$OUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED" \
  --portfolio-env \
  --strict-test-window

echo "[done] Artifacts under $OUT_DIR/"
