#!/usr/bin/env python3
"""
Download VIX term-structure series from Polygon with parallel day fetches.

Attempts using upgraded US indices tickers first (I:VIX, I:VIX9D, I:VIX3M),
falling back to the original per-day downloader logic if necessary.

Outputs a consolidated parquet at data/external/vix.parquet with columns
  close (1m ffilled to minute cadence optional), vix9d, vix3m if available.

Usage:
  export POLYGON_API_KEY=YOUR_KEY
  python scripts/download_vix_term_structure.py --start-date 2020-01-01 --end-date 2025-09-04
"""
import argparse
import sys
import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import aiohttp
import pandas as pd

BASE_URL = "https://api.polygon.io"


async def fetch_minute_aggs(session: aiohttp.ClientSession, api_key: str, ticker: str, day: str) -> pd.DataFrame:
    ep = f"/v2/aggs/ticker/{ticker}/range/1/minute/{day}/{day}"
    async with session.get(f"{BASE_URL}{ep}", params={"apiKey": api_key, "limit": 50000}) as resp:
        resp.raise_for_status()
        js = await resp.json()
        res = js.get("results") or []
        if not res:
            return pd.DataFrame()
        df = pd.DataFrame(res)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert('America/New_York')
        df = df.set_index('timestamp').sort_index()
        df = df.rename(columns={"c": "close"})
        return df[['close']]


async def collect_series(symbol: str, start: datetime, end: datetime, api_key: str) -> pd.Series:
    connector = aiohttp.TCPConnector(limit=16)
    timeout = aiohttp.ClientTimeout(total=60)
    days = pd.date_range(start, end, freq='D', tz='America/New_York')
    ticker = f"I:{symbol}"
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        chunks = []
        for i in range(0, len(days), 16):
            chunk_days = days[i:i+16]
            tasks = [fetch_minute_aggs(session, api_key, ticker, d.strftime('%Y-%m-%d')) for d in chunk_days]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for df in results:
                if isinstance(df, Exception) or df is None or df.empty:
                    continue
                chunks.append(df)
    if not chunks:
        return pd.Series(dtype=float)
    df_all = pd.concat(chunks, axis=0).sort_index()
    s = df_all['close']
    return s


def fallback_collect_daily(symbol: str, start: datetime, end: datetime, api_key: str) -> pd.Series:
    # Fallback: use 1-day aggregates with 1-day range (coarser)
    import requests
    ep = f"{BASE_URL}/v2/aggs/ticker/I:{symbol}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    r = requests.get(ep, params={"apiKey": api_key, "limit": 50000}, timeout=30)
    r.raise_for_status()
    js = r.json()
    res = js.get('results') or []
    if not res:
        return pd.Series(dtype=float)
    df = pd.DataFrame(res)
    idx = pd.to_datetime(df['t'], unit='ms', utc=True).tz_convert('America/New_York')
    return pd.Series(df['c'].values, index=idx).sort_index()


def main():
    ap = argparse.ArgumentParser(description="Download VIX term structure time series from Polygon")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--output", default="data/external/vix.parquet")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    api_key = args.api_key or os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY required")

    start = pd.Timestamp(args.start_date).tz_localize('America/New_York').to_pydatetime()
    end = pd.Timestamp(args.end_date).tz_localize('America/New_York').to_pydatetime()

    # First try minute series for I:VIX, I:VIX9D, I:VIX3M
    symbols = ['VIX', 'VIX9D', 'VIX3M']
    out: Dict[str, pd.Series] = {}
    # Create a fresh event loop to avoid deprecation warnings / missing loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sym in symbols:
        try:
            s = loop.run_until_complete(collect_series(sym, start, end, api_key))
            if s.empty:
                # fallback to daily if minute not available or forbidden
                try:
                    s = fallback_collect_daily(sym, start, end, api_key)
                except Exception:
                    s = pd.Series(dtype=float)
        except Exception:
            try:
                s = fallback_collect_daily(sym, start, end, api_key)
            except Exception:
                s = pd.Series(dtype=float)
        if not s.empty:
            out[sym.lower()] = s

    if not out:
        print("No VIX series available from Polygon APIs (subscription limits).", file=sys.stderr)
        sys.exit(1)

    # Build a DF with available columns and save
    df = pd.DataFrame(index=sorted(set().union(*[s.index for s in out.values()])))
    for k, s in out.items():
        df[k] = s.reindex(df.index)
    df = df.sort_index()

    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(op)
    print(f"Wrote {op} with columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
