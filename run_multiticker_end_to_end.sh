#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# CONFIG / REQUIREMENTS
# -----------------------------
: "${POLYGON_API_KEY:?Set POLYGON_API_KEY first (export POLYGON_API_KEY=...)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"   # optional: force CPU

REQ_CMDS=(curl jq)
for c in "${REQ_CMDS[@]}"; do
  command -v "$c" >/dev/null 2>&1 || { echo "Missing dependency: $c"; exit 1; }
done

# Training/OOS target sizes
TRAIN_N=36
OOS_N=12
# Screening knobs (override via env if exported)
PRICE_MIN=${PRICE_MIN:-10}
PRICE_MAX=${PRICE_MAX:-20}
MIN_DVOL=${MIN_DVOL:-500000}   # min prev-day volume to keep candidate
# Rate limiting (Polygon free tiers ~5 req/s). Be conservative.
RATE_SLEEP=${RATE_SLEEP:-0.25}
# Reference pagination cap
MAX_PAGES=${MAX_PAGES:-8}
# Max candidates to process in the prev-close screen (safety)
MAX_CANDIDATES=${MAX_CANDIDATES:-5000}
# HTTP retry / timeout knobs
MAX_RETRIES=${MAX_RETRIES:-5}
CURL_TIMEOUT=${CURL_TIMEOUT:-10}

# Heartbeat helper
heartbeat() { printf "\r[HB] %s" "$1"; }

# HTTP GET with basic retry/backoff. Prints body to stdout; returns 0 on 200.
http_get() {
  local url="$1" attempt=1 body code
  while :; do
    # Capture body + status code on last line
    local out
    out="$(curl -sS -m "$CURL_TIMEOUT" -w '\n%{http_code}' "$url" || true)"
    code="$(printf '%s' "$out" | tail -n1)"
    body="$(printf '%s' "$out" | sed '$d')"
    if [[ "$code" == "200" ]]; then
      printf '%s' "$body"
      return 0
    fi
    # Retry on 429/5xx with linear backoff tied to RATE_SLEEP
    if [[ "$code" == "429" || "$code" =~ ^5 ]]; then
      sleep "$(awk "BEGIN { print $RATE_SLEEP * $attempt }")"
      attempt=$((attempt+1))
      if (( attempt > MAX_RETRIES )); then
        return 1
      fi
    else
      return 1
    fi
  done
}
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/multiticker_${STAMP}"
mkdir -p "$OUT/tmp"

# -----------------------------
# 1) FETCH ACTIVE US COMMON STOCKS (PAGINATED)
# -----------------------------
echo ">>> Fetching active US common stocks from Polygon v3/reference/tickers …"
NEXT_URL="https://api.polygon.io/v3/reference/tickers?market=stocks&type=CS&active=true&limit=1000&apiKey=${POLYGON_API_KEY}"
ALL_TICKERS_JSON="$OUT/tmp/all_tickers.jsonl"
: > "$ALL_TICKERS_JSON"

PAGE=1
while [[ -n "$NEXT_URL" ]]; do
  echo "  Page $PAGE"
  RESP="$(http_get "$NEXT_URL" || true)"
  if [[ -z "$RESP" ]]; then echo "Failed fetching $NEXT_URL"; exit 1; fi
  echo "$RESP" | jq -c '.results[]' >> "$ALL_TICKERS_JSON"
  NEXT_URL="$(echo "$RESP" | jq -r '.next_url // ""')"
  if [[ -n "$NEXT_URL" && "$NEXT_URL" != "null" ]]; then
    [[ "$NEXT_URL" == *"apiKey="* ]] || NEXT_URL="${NEXT_URL}&apiKey=${POLYGON_API_KEY}"
  else
    NEXT_URL=""
  fi
  PAGE=$((PAGE+1))
  # Safety cap: MAX_PAGES pages is usually plenty for a big candidate pool
  if (( PAGE > MAX_PAGES )); then break; fi
done

# -----------------------------
# 2) FILTER TO MAJOR EXCHANGES (XNAS, XNYS, ARCX, BATS)
# -----------------------------
echo ">>> Filtering to major exchanges: XNAS XNYS ARCX BATS"
CANDIDATES="$OUT/tmp/candidates_symbols.txt"
jq -r '
  select(
    .primary_exchange=="XNAS" or
    .primary_exchange=="XNYS" or
    .primary_exchange=="ARCX" or
    .primary_exchange=="BATS"
  ) | .ticker
' "$ALL_TICKERS_JSON" | sort -u > "$CANDIDATES"

if [[ ! -s "$CANDIDATES" ]]; then
  echo "No candidate tickers after exchange filter."
  exit 1
fi

# -----------------------------
# 3) GET PREVIOUS CLOSE, FILTER $10–$20, RANK BY VOLUME
# -----------------------------
echo ">>> Querying previous close for price band ${PRICE_MIN}–${PRICE_MAX} and ranking by volume …"
PREV_JSON="$OUT/tmp/prev_filtered.jsonl"
: > "$PREV_JSON"

APPENDED=0
PROCESSED=0
while IFS= read -r T; do
  URL="https://api.polygon.io/v2/aggs/ticker/${T}/prev?adjusted=true&apiKey=${POLYGON_API_KEY}"
  BODY="$(http_get "$URL" || true)"
  HAS_RES="$(echo "$BODY" | jq '(.results|length) // 0' 2>/dev/null || echo 0)"
  if [[ "$HAS_RES" -gt 0 ]]; then
    CLOSE="$(echo "$BODY" | jq -r '.results[0].c // empty')"
    VOL="$(echo "$BODY" | jq -r '.results[0].v // 0')"
    if [[ -n "$CLOSE" && "$CLOSE" != "null" ]]; then
      # shell -> awk for numeric filter
      awk -v t="$T" -v c="$CLOSE" -v v="$VOL" -v minv="$MIN_DVOL" -v lo="$PRICE_MIN" -v hi="$PRICE_MAX" '
        BEGIN {
          pr = c + 0.0;
          vv = v + 0.0;
          if (pr >= lo && pr <= hi && vv >= minv) {
            printf("{\"ticker\":\"%s\",\"close\":%.4f,\"volume\":%.0f}\n", t, pr, vv);
          }
        }' >> "$PREV_JSON"
      APPENDED=$(wc -l < "$PREV_JSON" | tr -d ' ')
    fi
  fi
  PROCESSED=$((PROCESSED+1))
  if (( PROCESSED % 25 == 0 )); then heartbeat "prev-close processed=$PROCESSED appended=$APPENDED"; fi
  sleep "$RATE_SLEEP"
  # Stop once we have a healthy appended pool
  if (( APPENDED >= 600 )); then break; fi
  if (( PROCESSED >= MAX_CANDIDATES )); then echo; echo "Reached MAX_CANDIDATES=$MAX_CANDIDATES without enough symbols; continuing with what we have."; break; fi
done < "$CANDIDATES"
echo # newline after heartbeat

if [[ ! -s "$PREV_JSON" ]]; then
  echo "No tickers in $10–$20 with volume >= $MIN_DVOL were found. Adjust MIN_DVOL or widen pages."
  exit 1
fi

SORTED="$OUT/tmp/sorted_by_vol.json"
jq -s 'sort_by(-.volume)' "$PREV_JSON" > "$SORTED"

# -----------------------------
# 4) VERIFY MINUTE DATA EXISTS SINCE 2020-10-01
# -----------------------------
echo ">>> Verifying minute data availability since 2020-10-01 …"
VERIFIED="$OUT/tmp/verified_symbols.txt"
: > "$VERIFIED"

check_minute() {
  local sym="$1"
  local url="https://api.polygon.io/v2/aggs/ticker/${sym}/range/1/minute/2020-10-01/2020-10-08?limit=1&apiKey=${POLYGON_API_KEY}"
  local cnt
  cnt="$(curl -sf "$url" | jq -r '.resultsCount // 0' 2>/dev/null || echo 0)"
  [[ "$cnt" -gt 0 ]]
}

TOTAL=0
while IFS= read -r sym; do
  if check_minute "$sym"; then
    echo "$sym" >> "$VERIFIED"
    TOTAL=$((TOTAL+1))
  fi
  if (( TOTAL >= TRAIN_N + OOS_N + 30 )); then break; fi
  if (( TOTAL % 10 == 0 )); then heartbeat "verified=$TOTAL target=$((TRAIN_N+OOS_N))"; fi
  sleep "$RATE_SLEEP"
done < <(jq -r '.[].ticker' "$SORTED")
echo # newline after heartbeat

FOUND=$(wc -l < "$VERIFIED" | tr -d ' ')
if (( FOUND < TRAIN_N + OOS_N )); then
  echo "Not enough verified symbols. Found $FOUND."
  exit 1
fi

# -----------------------------
# 5) SPLIT INTO TRAIN (36) + OOS (12)
# -----------------------------
mapfile -t TRAIN_LIST < <(head -n "$TRAIN_N" "$VERIFIED")
mapfile -t OOS_LIST   < <(tail -n +"$((TRAIN_N+1))" "$VERIFIED" | head -n "$OOS_N")

echo ">>> TRAIN (${#TRAIN_LIST[@]}): ${TRAIN_LIST[*]}"
echo ">>> OOS   (${#OOS_LIST[@]}): ${OOS_LIST[*]}"

# -----------------------------
# 6) RUN YOUR PIPELINE
# -----------------------------
echo ">>> Running pipeline: scripts/run_multiticker_pipeline.py"
# Prefer project venv if available
PY_BIN="${PYTHON:-python}"
if [[ -x "venv/bin/python" ]]; then PY_BIN="venv/bin/python"; fi
PYTHONPATH=. "$PY_BIN" scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --tickers "${TRAIN_LIST[@]}" \
  --output-dir "$OUT"

# -----------------------------
# 7) LOCATE ARTIFACTS PRODUCED BY YOUR PIPELINE
# -----------------------------
DATA_FILE="$(ls "$OUT"/data/multiticker_data_*.parquet 2>/dev/null | head -n1 || true)"
FEAT_FILE="$(ls "$OUT"/features/multiticker_features_*.parquet 2>/dev/null | head -n1 || true)"
MODEL_PATH="$OUT/models/model"   # SB3 often saves model.zip; loader accepts base path

if [[ -z "${DATA_FILE}" || -z "${FEAT_FILE}" ]]; then
  echo "Could not auto-locate data/features under $OUT. Contents:"
  find "$OUT" -maxdepth 3 -type f | sed 's/^/  /'
  exit 1
fi

echo "Using data:     $DATA_FILE"
echo "Using features: $FEAT_FILE"
echo "Using model:    $MODEL_PATH"

# -----------------------------
# 8) OOS EVALUATION
# -----------------------------
OOS_OUT="$OUT/oos_eval"
mkdir -p "$OOS_OUT"

echo ">>> Running OOS on ${#OOS_LIST[@]} held-out symbols"
PYTHONPATH=. "$PY_BIN" scripts/oos_eval.py \
  --config configs/settings.yaml \
  --data "$DATA_FILE" \
  --features "$FEAT_FILE" \
  --model "$MODEL_PATH" \
  --test-tickers "${OOS_LIST[@]}" \
  --windows 3 \
  --output "$OOS_OUT"

echo "✅ Done."
echo "Train/OOS lists saved in: $OUT/tmp/"
echo "OOS results:              $OOS_OUT/oos_results.json"
