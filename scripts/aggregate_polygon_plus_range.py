#!/usr/bin/env python3
"""
Aggregate local Polygon+ partitioned OHLCV from
data/polygon_plus/us_stocks/aggregates/symbol=<SYMBOL>/year=YYYY/month=MM/day=DD/data.parquet
into a single repo-relative parquet for a given date range.

Usage:
  PYTHONPATH=. python scripts/aggregate_polygon_plus_range.py \
    --symbol BBVA --start 2025-01-01 --end 2025-06-30 \
    --out data/raw/BBVA_20250101_20250630_1min.parquet

No API calls; uses local parquet partitions only.
"""
import argparse
from pathlib import Path
import pandas as pd


def load_day_file(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    # Normalize index to America/New_York
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.loc[ts.notna()].copy()
        df.index = ts[ts.notna()].tz_convert('America/New_York')
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            for c in ('ts','time','datetime','date'):
                if c in df.columns:
                    ts = pd.to_datetime(df[c], utc=True, errors='coerce')
                    df = df.loc[ts.notna()].copy()
                    df.index = ts[ts.notna()].tz_convert('America/New_York')
                    break
        else:
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize('UTC')
            df.index = idx.tz_convert('America/New_York')

    # Standardize column names if needed
    rename_map = {}
    if 'o' in df.columns: rename_map['o']='open'
    if 'h' in df.columns: rename_map['h']='high'
    if 'l' in df.columns: rename_map['l']='low'
    if 'c' in df.columns: rename_map['c']='close'
    if 'v' in df.columns: rename_map['v']='volume'
    if rename_map:
        df = df.rename(columns=rename_map)

    # Keep common OHLCV (+ optional vwap/transactions)
    keep = [c for c in ('open','high','low','close','volume','vwap','transactions') if c in df.columns]
    df = df[keep]
    # Filter to RTH
    df = df.between_time('09:30','16:00')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    root = Path('data/polygon_plus/us_stocks/aggregates') / f'symbol={args.symbol}'
    if not root.exists():
        raise SystemExit(f'Not found: {root} (expected local Polygon+ partitions)')

    start = pd.Timestamp(args.start, tz='America/New_York')
    end = pd.Timestamp(args.end, tz='America/New_York')

    files = []
    try:
        years = range(start.year, end.year + 1)
        for y in years:
            ydir = root / f'year={y}'
            if not ydir.exists():
                continue
            m_start = 1 if y > start.year else start.month
            m_end = 12 if y < end.year else end.month
            for m in range(m_start, m_end + 1):
                mdir = ydir / f'month={m:02d}'
                if not mdir.exists():
                    continue
                for ddir in sorted(mdir.glob('day=*')):
                    p = ddir / 'data.parquet'
                    if p.exists():
                        files.append(p)
    except Exception as e:
        raise SystemExit(f'Error scanning partitions: {e}')

    if not files:
        raise SystemExit('No day parquet files found in requested range')

    frames = []
    for p in files:
        try:
            df = load_day_file(p)
            mask = (df.index >= start) & (df.index <= end)
            if mask.any():
                frames.append(df[mask])
        except Exception:
            continue

    if not frames:
        raise SystemExit('No rows in requested range after normalization')

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep='first')]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f'Wrote {out_path} rows={len(out)} cols={list(out.columns)}')
    print(f'Range: {out.index.min()} -> {out.index.max()}')


if __name__ == '__main__':
    main()

