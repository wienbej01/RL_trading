#!/usr/bin/env bash
set -euo pipefail

# ====== 0) Optional: force CPU ======
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# ====== 1) Define training & OOS lists (comma-separated) ======
# Replace with your screened sets
TRAIN_TICKERS="AA,ACI,AMD,APA,APLS,AR,ATEC,BBBYQ,BEDU,BGCP,BLNK,BNED,BWX,CEG,CHPT,CLF,CLNE,COIN,CRSR,DNMR,ENVX,FCEL,FTCH,GENI,GRWG,HLIT,IONQ,LAZR,LCID,MTTR,PATH,PLUG,RIOT,RUN,SOFI,UPST"
OOS_TICKERS="AUPH,BE,BERY,CANO,CRON,DKNG,FATE,FUTU,GME,HUT,JOBY,OPEN"

# ====== 2) Convert to space-separated arrays for argparse ======
IFS=',' read -r -a TRAIN_ARR <<< "$TRAIN_TICKERS"
IFS=',' read -r -a OOS_ARR   <<< "$OOS_TICKERS"

# ====== 3) Optional: set Polygon key if you intend to auto-download US stocks ======
# export POLYGON_API_KEY="YOUR_POLYGON_API_KEY"

# ====== 4) Run multi-ticker pipeline (download → features → train → backtest) ======
OUT="results/multiticker_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
PYTHONPATH=. ${PYTHON:-python} scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --tickers "${TRAIN_ARR[@]}" \
  --output-dir "$OUT"

# ====== 5) Locate produced artifacts (as written by your pipeline) ======
DATA_FILE="$(ls "$OUT"/data/multiticker_data_*.parquet 2>/dev/null | head -n1 || true)"
FEAT_FILE="$(ls "$OUT"/features/multiticker_features_*.parquet 2>/dev/null | head -n1 || true)"
MODEL_PATH="$OUT/models/model"   # SB3 usually emits model.zip; loader accepts the base path too

if [[ -z "${DATA_FILE}" || -z "${FEAT_FILE}" ]]; then
  echo "Could not auto-locate data/features under $OUT. Contents:"
  find "$OUT" -maxdepth 3 -type f | sed 's/^/  /'
  echo "Please point --data and --features to the actual files your run produced."
  exit 1
fi

echo "Using data:      $DATA_FILE"
echo "Using features:  $FEAT_FILE"
echo "Using model base $MODEL_PATH"

# ====== 6) OOS evaluation on held-out names (scripts/oos_eval.py requires these flags) ======
OOS_OUT="$OUT/oos_eval"
mkdir -p "$OOS_OUT"
PYTHONPATH=. ${PYTHON:-python} scripts/oos_eval.py \
  --config configs/settings.yaml \
  --data "$DATA_FILE" \
  --features "$FEAT_FILE" \
  --model "$MODEL_PATH" \
  --test-tickers "${OOS_ARR[@]}" \
  --windows 3 \
  --output "$OOS_OUT"

echo "✅ OOS results written to: $OOS_OUT/oos_results.json"

# ====== 7) Optional: quick repo checks ======
# PYTHONPATH=. ${PYTHON:-python} -m pytest -q || true
