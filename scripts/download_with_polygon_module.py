#!/usr/bin/env python3
"""
Download intraday minute bars via the external standardized Polygon module
(`trade_system_modules`) and save into this repo's RAW layout expected by
UnifiedDataLoader:

  data/polygon/historical/symbol=SYMBOL/year=YYYY/month=MM/day=DD/data.parquet

Usage:
  ./venv/bin/python scripts/download_with_polygon_module.py \
    --module-path /path/to/trade_system_modules/src \
    --tickers AAPL MSFT NVDA \
    --start 2024-01-02 --end 2024-01-12

Notes:
- Does NOT modify or replace existing loaders. It only writes files.
- Requires POLYGON_API_KEY in the environment as used by the external module.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import requests


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch minute bars via standardized Polygon module")
    ap.add_argument("--module-path", required=True, help="Path to trade_system_modules/src (added to sys.path)")
    ap.add_argument("--tickers", nargs="+", required=True, help="Symbols to download")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--chunk-days", type=int, default=30, help="Chunk size in days for ranged fetch (default: 30)")
    ap.add_argument("--asset-class", type=str, default="stocks", choices=["stocks","others"], help="Asset class for rate-plan behavior")
    ap.add_argument("--verbose", action='store_true', help="Enable verbose logging per chunk and write")
    ap.add_argument("--out-root", default="data/polygon/historical", help="Root for RAW partitioned output")
    return ap.parse_args()


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize expected columns from adapter
    # Expected: timestamp/ts, open, high, low, close, volume, vwap?, trades/transactions
    out = df.copy()
    if 'timestamp' not in out.columns:
        if 'ts' in out.columns:
            out = out.rename(columns={'ts': 'timestamp'})
    if 'transactions' not in out.columns and 'trades' in out.columns:
        out = out.rename(columns={'trades': 'transactions'})
    # Ensure datetime index for partitioning
    ts = pd.to_datetime(out['timestamp'], utc=True, errors='coerce') if 'timestamp' in out.columns else None
    if ts is not None:
        out = out.loc[ts.notna()].copy()
        out['timestamp'] = ts
        out = out.sort_values('timestamp').set_index('timestamp')
    if out.index.tz is None:
        out.index = out.index.tz_localize('UTC')
    # Convert to NY for consistency with loader dirs if desired (kept UTC for storage agnostic)
    return out


def _write_partitioned(df: pd.DataFrame, symbol: str, root: Path, *, verbose: bool = False) -> int:
    """Write df partitioned by year/month/day under symbol=SYMBOL folders.

    Returns number of day files written.
    """
    if df.empty:
        return 0
    # Group by calendar day in America/New_York to align with loader
    local = df.copy()
    local.index = local.index.tz_convert('America/New_York')
    by_day = local.groupby(local.index.normalize())
    written = 0
    for day, day_df in by_day:
        y = day.year
        m = f"{day.month:02d}"
        d = f"{day.day:02d}"
        out_dir = root / f"symbol={symbol}" / f"year={y}" / f"month={m}" / f"day={d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write parquet
        day_df_utc = day_df.copy()
        day_df_utc.index = day_df_utc.index.tz_convert('UTC')
        day_df_utc.to_parquet(out_dir / "data.parquet")
        if verbose:
            print(f"    wrote {symbol} {day} rows={len(day_df_utc)} -> {out_dir / 'data.parquet'}")
        written += 1
    return written


def _fallback_get_agg_minute(symbol: str, start: str, end: str, api_key: str, verbose: bool=False) -> pd.DataFrame:
    """Fallback fetch using Polygon v2 aggregates with correct auth.

    Loops over pagination via next_url if present.
    """
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
        if verbose:
            print(f"    [fallback] GET {r.url} -> {r.status_code}")
        try:
            r.raise_for_status()
        except requests.HTTPError as he:
            # Print body for diagnostics
            try:
                print(f"      body: {r.text[:200]}...")
            except Exception:
                pass
            raise
        data = r.json()
        if data.get('results'):
            results.extend(data['results'])
        next_url = data.get('next_url')
        if not next_url:
            break
        # next_url requires auth again via header (no query apiKey)
        url = next_url

    if not results:
        return pd.DataFrame()
    # Map Polygon fields: t=timestamp(ms), o,h,l,c,v,vw,n
    df = pd.DataFrame(results)
    # Normalize column names to expected schema
    mapping = {
        't': 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'vw': 'vwap',
        'n': 'transactions'
    }
    for src, dst in mapping.items():
        if src in df.columns:
            df[dst] = df[src]
    return df


def main() -> None:
    args = _parse_args()
    module_path = Path(args.module_path).resolve()
    if not module_path.exists():
        raise SystemExit(f"Module path not found: {module_path}")
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

    try:
        from trade_system_modules.data.polygon_adapter import get_agg_minute
    except Exception as e:
        raise SystemExit(f"Failed to import trade_system_modules from {module_path}: {e}")

    if not os.getenv("POLYGON_API_KEY"):
        print("WARNING: POLYGON_API_KEY not set; the external module may fail to authenticate.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for sym in args.tickers:
        print(f"Fetching {sym} {args.start} → {args.end} …")
        try:
            import pandas as _pd
            from datetime import timedelta
            # Always chunk by ~month to overcome per-request limits
            s = _pd.to_datetime(args.start)
            e = _pd.to_datetime(args.end)
            chunk = timedelta(days=int(args.chunk_days))
            parts = []
            cursor = s
            while cursor < e:
                stop = min(cursor + chunk, e)
                try:
                    if args.verbose:
                        print(f"  [fetch] {sym} {cursor.date()} → {stop.date()} ...", end="")
                    # Try module first
                    try:
                        dfi = get_agg_minute(sym, cursor.strftime('%Y-%m-%d'), stop.strftime('%Y-%m-%d'))
                    except Exception as ee:  # module-level failure → fallback
                        if args.verbose:
                            print(f" module_error: {ee}; trying fallback")
                        dfi = _fallback_get_agg_minute(sym, cursor.strftime('%Y-%m-%d'), stop.strftime('%Y-%m-%d'), os.getenv('POLYGON_API_KEY',''), verbose=args.verbose)

                    # If module returned but 403 was raised internally, fallback above handled it. If still empty, attempt fallback once more
                    if dfi is None or len(dfi) == 0:
                        # final fallback
                        dfi = _fallback_get_agg_minute(sym, cursor.strftime('%Y-%m-%d'), stop.strftime('%Y-%m-%d'), os.getenv('POLYGON_API_KEY',''), verbose=args.verbose)

                    if dfi is not None and len(dfi) > 0:
                        parts.append(dfi)
                        if args.verbose:
                            print(f" ok rows={len(dfi)}")
                    else:
                        if args.verbose:
                            print(" empty")
                except Exception as ee:
                    print(f"    [error] chunk {cursor.date()}→{stop.date()} error: {ee}")
                # Throttle only for non-stocks asset classes
                if args.asset_class != 'stocks':
                    import time as _time
                    _time.sleep(12)
                cursor = stop
            if not parts:
                print(f"  No data for {sym}")
                continue
            df = _pd.concat(parts, ignore_index=True)
        except Exception as e:
            print(f"  ERROR fetching {sym}: {e}")
            continue
        df = _ensure_cols(df)
        if args.verbose:
            try:
                rng = (df.index.min(), df.index.max())
                print(f"  [normalize] {sym} index tz={df.index.tz} range=[{rng[0]} .. {rng[1]}] rows={len(df)}")
            except Exception:
                pass
        written = _write_partitioned(df, sym, out_root, verbose=args.verbose)
        total_written += written
        print(f"  Wrote {written} day partitions for {sym} under {out_root}/symbol={sym}")

    print(f"Done. Total day files written: {total_written}")


if __name__ == "__main__":
    main()
