#!/usr/bin/env python3
"""
Bulk ingest Polygon minute bars into partitioned parquet under data/polygon/historical,
respecting a conservative rate limit (default 5 calls/min).

Usage examples:
  PYTHONPATH=. python scripts/polygon_bulk_ingest.py \
    --config configs/settings.yaml \
    --tickers SPY,QQQ \
    --start 2022-01-01 --end 2024-12-31

Notes:
- Requires POLYGON_API_KEY in environment or config.
- Saves per-day parquet partitions compatible with the loader.
- Uses monthly slices to minimize number of API calls.
"""
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List

from src.utils.config_loader import Settings
from src.data.polygon_ingestor import PolygonDataIngestor


def month_slices(start: str, end: str) -> List[tuple[str, str]]:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    out = []
    cur = datetime(s.year, s.month, 1)
    # end boundary is inclusive for last day of month not exceeding 'e'
    while cur <= e:
        month_start = cur
        month_end = (cur + relativedelta(months=1)) - timedelta(days=1)
        if month_start < s:
            month_start = s
        if month_end > e:
            month_end = e
        out.append((month_start.date().isoformat(), month_end.date().isoformat()))
        cur = (cur + relativedelta(months=1))
    return out


async def run(args):
    settings = Settings.from_paths(args.config)
    ingestor = PolygonDataIngestor(settings)
    # Optional override of client calls-per-minute
    try:
        if getattr(args, 'cpm', None):
            ingestor.polygon_client.rate_limiter.calls_per_minute = int(args.cpm)
            print(f"Rate limiter set to {int(args.cpm)} calls/min")
    except Exception:
        pass

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    slices = month_slices(args.start, args.end)

    total_calls = len(tickers) * len(slices)
    print(f"Planned calls: {total_calls} (tickers={len(tickers)} x months={len(slices)})")
    print(f"Output dir: {ingestor.data_dir}")

    # Iterate sequentially to respect 5 calls/min rate limiter on the client
    for t in tickers:
        for (ms, me) in slices:
            print(f"Fetching {t} {ms} -> {me} ...")
            try:
                res = await ingestor.fetch_historical_data(
                    symbols=[t],
                    start_date=ms,
                    end_date=me,
                    data_types=['ohlcv'],
                    incremental=False
                )
                ok = res.get('successful_fetches', 0)
                fail = res.get('failed_fetches', 0)
                print(f"  Done: success={ok} fail={fail}")
            except Exception as e:
                print(f"  ERROR: {e}")

    await ingestor.aclose()
    print("Ingestion complete.")


def main():
    ap = argparse.ArgumentParser(description="Bulk ingest Polygon OHLCV minute bars with monthly slices")
    ap.add_argument('--config', default='configs/settings.yaml')
    ap.add_argument('--tickers', required=True, help='Comma-separated list (e.g., SPY,QQQ)')
    ap.add_argument('--start', required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', required=True, help='YYYY-MM-DD')
    ap.add_argument('--cpm', type=int, default=None, help='Override calls per minute (default 5)')
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
