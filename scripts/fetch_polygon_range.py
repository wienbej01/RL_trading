#!/usr/bin/env python3
"""
Fetch Polygon OHLCV for a symbol and date range, then aggregate to a single parquet.

Usage:
  PYTHONPATH=. python scripts/fetch_polygon_range.py \
    --symbol BBVA --start 2025-01-01 --end 2025-06-30 \
    --out data/raw/BBVA_20250101_20250630_1min.parquet

Requires POLYGON_API_KEY to be set in the environment or via configs/settings.yaml.
"""
import argparse
import asyncio
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import Settings
from src.data.polygon_ingestor import PolygonDataIngestor


async def fetch(symbol: str, start: str, end: str, settings: Settings):
    ingestor = PolygonDataIngestor(settings)
    try:
        await ingestor.fetch_historical_data(
            symbols=[symbol],
            start_date=start,
            end_date=end,
            data_types=['ohlcv']
        )
    finally:
        await ingestor.aclose()


def aggregate(symbol: str, start: str, end: str, out_path: Path) -> int:
    root = Path('data/polygon/historical') / f'symbol={symbol}'
    if not root.exists():
        # fallback legacy path if needed
        root = Path('rl-intraday/data/polygon/historical') / f'symbol={symbol}'
    files = sorted(root.glob('year=*/month=*/day=*/data.parquet'))
    dfs = []
    if not files:
        raise FileNotFoundError(f"No polygon partitions found under {root}")
    for p in files:
        try:
            df = pd.read_parquet(p)
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York', ambiguous='infer')
            else:
                df.index = df.index.tz_convert('America/New_York')
            m = (df.index >= pd.Timestamp(start, tz='America/New_York')) & (df.index <= pd.Timestamp(end, tz='America/New_York'))
            if m.any():
                dfs.append(df[m])
        except Exception:
            continue
    if not dfs:
        raise FileNotFoundError(f"No rows in range {start}..{end} for {symbol}")
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.between_time('09:30','16:00')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    return len(combined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='configs/settings.yaml')
    args = ap.parse_args()

    settings = Settings.from_paths(args.config)

    # Fetch
    asyncio.run(fetch(args.symbol, args.start, args.end, settings))

    # Aggregate
    out_path = Path(args.out)
    n = aggregate(args.symbol, args.start, args.end, out_path)
    print(f"Wrote {n} rows to {out_path}")


if __name__ == '__main__':
    main()

