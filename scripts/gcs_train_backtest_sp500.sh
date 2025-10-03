#!/usr/bin/env bash
set -euo pipefail

# End-to-end helper: (1) download-from-GCS + train, (2) post-train backtest.
#
# Requirements:
# - gsutil installed and authenticated (gcloud auth login or service account)
# - Python env with project deps (venv recommended)
# - Repo layout as in rl-intraday
#
# Usage (defaults are sensible):
#   bash scripts/gcs_train_backtest_sp500.sh \
#     --bucket gs://jwss_data_store \
#     --train-start 2020-09-01 --train-end 2024-12-31 \
#     --test-start  2025-01-01 --test-end  2025-06-30 \
#     --output results/sp500_run_$(date +%Y%m%d_%H%M%S)
#
# Options:
#   --bucket GS_URI                GCS bucket (default: gs://jwss_data_store)
#   --config PATH                 Config YAML (default: configs/settings.yaml)
#   --train-start YYYY-MM-DD      Train start date
#   --train-end   YYYY-MM-DD      Train end date
#   --test-start  YYYY-MM-DD      Test start date
#   --test-end    YYYY-MM-DD      Test end date
#   --train-tickers "..."         Space-separated list (defaults to curated S&P500 set)
#   --oos-tickers   "..."         Space-separated list (defaults to curated OOS set)
#   --output DIR                   Output base directory

BUCKET="gs://jwss_data_store"
CONFIG="configs/settings.yaml"
TRAIN_START="2020-09-01"
TRAIN_END="2024-12-31"
TEST_START="2025-01-01"
TEST_END="2025-06-30"
TRAIN_TICKERS=(AAPL MSFT AMZN GOOGL NVDA AVGO AMD JPM BAC V MA XOM CVX UNH JNJ PG KO PEP HD COST WMT TGT ORCL CRM)
OOS_TICKERS=(META NFLX DIS INTC GE UPS UNP LLY ABBV MRK PFE TXN)
OUTPUT="results/sp500_run_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket) BUCKET="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --train-start) TRAIN_START="$2"; shift 2;;
    --train-end) TRAIN_END="$2"; shift 2;;
    --test-start) TEST_START="$2"; shift 2;;
    --test-end) TEST_END="$2"; shift 2;;
    --train-tickers) IFS=' ' read -r -a TRAIN_TICKERS <<<"$2"; shift 2;;
    --oos-tickers) IFS=' ' read -r -a OOS_TICKERS <<<"$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    -h|--help)
      sed -n '1,80p' "$0" | sed -n '1,80p'; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Resolve python binary (prefer venv)
PY_BIN="${PYTHON:-python3}"
if [[ -x "venv/bin/python" ]]; then PY_BIN="venv/bin/python"; fi

# Resolve gsutil
GSUTIL_BIN="$(command -v gsutil || true)"
if [[ -z "$GSUTIL_BIN" && -x "/home/$USER/Downloads/google-cloud-sdk/bin/gsutil" ]]; then
  GSUTIL_BIN="/home/$USER/Downloads/google-cloud-sdk/bin/gsutil"
fi
if [[ -z "$GSUTIL_BIN" ]]; then
  echo "gsutil not found in PATH. Install Google Cloud SDK or export gsutil path." >&2
  exit 1
fi

echo ">> Settings"
echo "  BUCKET       : $BUCKET"
echo "  CONFIG       : $CONFIG"
echo "  TRAIN window : $TRAIN_START .. $TRAIN_END"
echo "  TEST  window : $TEST_START  .. $TEST_END"
echo "  TRAIN tickers: ${TRAIN_TICKERS[*]}"
echo "  OOS   tickers: ${OOS_TICKERS[*]}"
echo "  OUTPUT base  : $OUTPUT"

mkdir -p "$OUTPUT"

year_seq() {
  local s="$1" e="$2"
  local y1 y2
  y1=$(date -d "$s" +%Y)
  y2=$(date -d "$e" +%Y)
  local y
  for ((y=y1; y<=y2; y++)); do printf "%s\n" "$y"; done
}

copy_gcs_monthlies() {
  # $1: ticker list (array name); $2: dest dir; $3..: years
  local -n _TKS=$1
  local dest="$2"; shift 2
  local years=("$@")
  mkdir -p "$dest"
  for T in "${_TKS[@]}"; do
    for Y in "${years[@]}"; do
      # Copy monthly parquet files if they exist
      $GSUTIL_BIN -m cp -n "$BUCKET/stocks/$T/$Y/${T}_${Y}-*.parquet" "$dest/" 2>/dev/null || true
    done
  done
}

build_multiticker_parquet() {
  # $1: src glob dir (monthlies), $2: out path, requires pandas/pyarrow
  local src_dir="$1" out_path="$2"
  "$PY_BIN" - <<'PY'
import os, glob, pandas as pd, sys
src_dir = os.environ['SRC_DIR']
out_path = os.environ['OUT_PATH']
files = sorted(glob.glob(os.path.join(src_dir, '*.parquet')))
if not files:
    print('No parquet files found in', src_dir, file=sys.stderr)
    sys.exit(1)
parts = []
def _std_cols(df):
    # Lowercase columns and map common aliases to standard OHLCV names
    colmap = {c: c.lower() for c in df.columns}
    df = df.rename(columns=colmap)
    alias = {
        'o': 'open', 'op': 'open', 'open_price': 'open',
        'h': 'high', 'hi': 'high',
        'l': 'low',
        'c': 'close', 'close_price': 'close', 'adj_close': 'close', 'adjusted_close': 'close',
        'v': 'volume', 'vol': 'volume',
        'vw': 'vwap', 'avgpx': 'vwap'
    }
    for k,v in alias.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df
def _detect_ts(df):
    for c in ('timestamp','datetime','time','date','dt','ts','t'):
        if c in df.columns:
            return c
    return None
for f in files:
    t = os.path.basename(f).split('_')[0]
    try:
        df = pd.read_parquet(f)
    except Exception as e:
        print('read fail', f, e, file=sys.stderr)
        continue
    df = _std_cols(df)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.loc[ts.notna()].copy(); df['timestamp']=ts; df = df.set_index('timestamp')
    else:
        ts_col = _detect_ts(df)
        if ts_col is not None:
            ts = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
            df = df.loc[ts.notna()].copy(); df[ts_col]=ts; df = df.set_index(ts_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            idx = pd.to_datetime(df.index, utc=True, errors='coerce')
            df = df.loc[idx.notna()]; df.index = idx[idx.notna()]
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    idx = idx.tz_convert('America/New_York')
    df.index = idx
    df = df.sort_index()
    keep = [c for c in ['open','high','low','close','volume','vwap'] if c in df.columns]
    if not keep:
        # If nothing standard, print sample columns for debugging and skip
        print('no OHLCV in', os.path.basename(f), 'cols=', list(df.columns)[:10], file=sys.stderr)
        continue
    df = df[keep].copy(); df['ticker']=t
    parts.append(df)
if not parts:
    print('No valid OHLCV columns across files in', src_dir, file=sys.stderr)
    sys.exit(1)
all_ = pd.concat(parts, axis=0).sort_index()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
all_.to_parquet(out_path)
print('Wrote', out_path, all_.shape)
PY
}

# ---------------------------
# Step 1: Download TRAIN + Train model
# ---------------------------
TRAIN_OUT="$OUTPUT/train"
mkdir -p "$TRAIN_OUT/data/gcs"
readarray -t YEARS < <(year_seq "$TRAIN_START" "$TEST_END")
echo ">> Copying TRAIN monthlies from GCS …"
copy_gcs_monthlies TRAIN_TICKERS "$TRAIN_OUT/data/gcs" "${YEARS[@]}"

export SRC_DIR="$TRAIN_OUT/data/gcs"
export OUT_PATH="$TRAIN_OUT/data/multiticker_data_${TRAIN_START}_to_${TEST_END}.parquet"
echo ">> Building TRAIN multi-ticker parquet …"
build_multiticker_parquet "$SRC_DIR" "$OUT_PATH"

echo ">> Generating features and training model …"
PYTHONPATH=. "$PY_BIN" scripts/run_multiticker_pipeline.py \
  --config "$CONFIG" \
  --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
  --test-start  "$TEST_START"  --test-end  "$TEST_END" \
  --tickers "${TRAIN_TICKERS[*]}" \
  --output-dir "$TRAIN_OUT" \
  --skip-download \
  --skip-backtest

MODEL_PATH="$TRAIN_OUT/models/model"
if [[ ! -e "$MODEL_PATH.zip" && ! -e "$MODEL_PATH" ]]; then
  echo "Model not found at $MODEL_PATH(.zip). Training may have failed." >&2
  exit 1
fi

# ---------------------------
# Step 2: Download OOS + Backtest
# ---------------------------
OOS_OUT="$OUTPUT/oos"
mkdir -p "$OOS_OUT/data/gcs"
echo ">> Copying OOS monthlies from GCS …"
copy_gcs_monthlies OOS_TICKERS "$OOS_OUT/data/gcs" "${YEARS[@]}"

export SRC_DIR="$OOS_OUT/data/gcs"
export OUT_PATH="$OOS_OUT/data/multiticker_data_${TRAIN_START}_to_${TEST_END}.parquet"
echo ">> Building OOS multi-ticker parquet …"
build_multiticker_parquet "$SRC_DIR" "$OUT_PATH"

echo ">> Generating OOS features …"
PYTHONPATH=. "$PY_BIN" scripts/run_multiticker_pipeline.py \
  --config "$CONFIG" \
  --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
  --test-start  "$TEST_START"  --test-end  "$TEST_END" \
  --tickers "${OOS_TICKERS[*]}" \
  --output-dir "$OOS_OUT" \
  --skip-download \
  --skip-training \
  --skip-backtest

echo ">> Running OOS backtest …"
OOS_FEAT="$OOS_OUT/features/multiticker_features_${TRAIN_START}_to_${TEST_END}.parquet"
if [[ ! -e "$OOS_FEAT" ]]; then
  echo "OOS features not found at $OOS_FEAT" >&2
  exit 1
fi

BT_OUT="$TRAIN_OUT/oos_eval"
mkdir -p "$BT_OUT"
PYTHONPATH=. "$PY_BIN" scripts/oos_eval.py \
  --config "$CONFIG" \
  --data "$OUT_PATH" \
  --features "$OOS_FEAT" \
  --model "$MODEL_PATH" \
  --test-tickers "${OOS_TICKERS[*]}" \
  --windows 3 \
  --output "$BT_OUT"

echo "✅ Done. Outputs:"
echo "  Train dir : $TRAIN_OUT"
echo "  OOS dir   : $OOS_OUT"
echo "  OOS eval  : $BT_OUT"
