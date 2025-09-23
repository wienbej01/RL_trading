#!/usr/bin/env python3
"""
Download VIX minute aggregates from Polygon with 5 calls/min throttling and
save a daily-aligned series to data/external/vix.parquet.

Usage:
  export POLYGON_API_KEY=...
  PYTHONPATH=. python scripts/download_vix_polygon.py \
    --start 2024-01-01 --end 2024-09-30 --symbol I:VIX

Notes:
- Respects 5 req/min by sleeping across chunks.
- Writes parquet with columns: vix (close), and keeps DatetimeIndex in America/New_York.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests


def _fetch_agg_minute(symbol: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    base = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{end}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
    }
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    results = []
    url = base
    session = requests.Session()
    while True:
        r = session.get(url, params=params if url == base else None, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        if data.get('results'):
            results.extend(data['results'])
        next_url = data.get('next_url')
        if not next_url:
            break
        url = next_url
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    mapping = {'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'vw': 'vwap', 'n': 'transactions'}
    for src, dst in mapping.items():
        if src in df.columns:
            df[dst] = df[src]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp').set_index('timestamp')
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Download VIX via Polygon (5 cpm throttle)")
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--symbol', default='I:VIX', help='Polygon index symbol (default I:VIX)')
    ap.add_argument('--out', default='data/external/vix.parquet')
    ap.add_argument('--chunk-days', type=int, default=14)
    args = ap.parse_args()

    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise SystemExit('POLYGON_API_KEY not set')

    s = pd.to_datetime(args.start)
    e = pd.to_datetime(args.end)
    chunk = timedelta(days=int(args.chunk_days))
    cursor = s
    parts = []
    calls = 0
    while cursor < e:
        stop = min(cursor + chunk, e)
        df = _fetch_agg_minute(args.symbol, cursor.strftime('%Y-%m-%d'), stop.strftime('%Y-%m-%d'), api_key)
        parts.append(df)
        calls += 1
        # Throttle to ~5 calls/min
        if calls % 5 == 0:
            time.sleep(60)
        else:
            time.sleep(12)
        cursor = stop

    if not parts:
        raise SystemExit('No data fetched')
    allm = pd.concat(parts).sort_index()
    # Convert to NY timezone and resample to daily close
    idx = allm.index.tz_convert('America/New_York')
    allm = allm.copy()
    allm.index = idx
    daily = allm['close'].resample('1D').last().to_frame('vix')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out)
    print(f"Saved VIX series ({len(daily)} rows) to {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

