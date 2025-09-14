#!/usr/bin/env python3
"""
Futures minute-bar backfill using the existing Polygon ingestion stack.

- Accepts Polygon futures tickers directly (e.g., C:ES, C:NQ, C:CL) or explicit contracts.
- Stores partitioned Parquet under data/polygon/historical/symbol=.../year=YYYY/month=MM/day=DD/data.parquet
  so it is compatible with UnifiedDataLoader and the rest of the pipeline.

Usage:
  export POLYGON_API_KEY=YOUR_KEY
  ./venv/bin/python scripts/collect_futures_data.py \
      --symbols C:ES,C:NQ \
      --start-date 2020-09-01 --end-date 2025-06-30

Notes:
  - For continuous futures, use Polygon's continuous tickers (e.g., C:ES) if available on your plan.
  - For individual contracts, pass their Polygon symbols directly.
"""
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.config_loader import Settings
from src.utils.logging import get_logger
from src.data.polygon_ingestor import PolygonDataIngestor

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill futures minute bars via Polygon")
    p.add_argument("--symbols", required=True, help="Comma-separated Polygon symbols (e.g., C:ES,C:NQ or contract codes)")
    p.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--config", default="configs/settings.yaml", help="Path to settings YAML (default: configs/settings.yaml)")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    settings = Settings.from_paths(args.config)
    ingestor = PolygonDataIngestor(settings)

    symbols: List[str] = [s.strip() for s in args.symbols.split(',') if s.strip()]

    # Fetch OHLCV for all symbols. The ingestor handles per-day partitioning and validation.
    logger.info(f"Starting futures backfill for {len(symbols)} symbols from {args.start_date} to {args.end_date}")
    results = await ingestor.fetch_historical_data(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        data_types=['ohlcv'],
        incremental=False,
        dry_run=False,
        progress_callback=lambda sym, dt, n: logger.info(f"Fetched {n} {dt} rows for {sym}")
    )

    ok = results.get('failed_fetches', 0) == 0
    logger.info(f"Backfill complete. Success: {results.get('successful_fetches', 0)}, Failed: {results.get('failed_fetches', 0)}")
    return 0 if ok else 1


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()

