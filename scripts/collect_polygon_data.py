#!/usr/bin/env python3
"""
Polygon Data Collection Script for RL Trading System

This script collects historical market data from Polygon API for backtesting
the RL trading system. Optimized for free tier limits (5 calls/minute) and
designed for local/VM execution with efficient storage.

Features:
- Rate limiting compliance (5 calls/minute)
- Daily data fetching for NYSE trading days
- Progress tracking and error handling
- Efficient Parquet storage
- Data quality validation
- Command-line interface

Usage:
    python scripts/collect_polygon_data.py --symbols SPY --start-date 2024-01-01 --end-date 2024-06-30
"""

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import aiohttp
import pandas_market_calendars as mcal

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger
from src.utils.config_loader import Settings

logger = get_logger(__name__)

# Configuration
POLYGON_BASE_URL = "https://api.polygon.io"
RATE_LIMIT_DELAY = 12  # 5 calls/minute = 12 seconds between calls
REQUEST_TIMEOUT = 30

class PolygonDataCollector:
    """
    Polygon API data collector with rate limiting and error handling.
    """

    def __init__(self, api_key: str, data_dir: Path = None):
        """
        Initialize the data collector.
        Args:
            api_key: Polygon API key
            data_dir: Directory to store collected data
        """
        self.api_key = api_key
        self.data_dir = data_dir or Path("data/polygon/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info(f"Initialized PolygonDataCollector with data directory: {self.data_dir}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _enforce_rate_limit(self):
        """Enforce rate limiting (5 calls per minute)."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a rate-limited API request with retry logic.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        self._enforce_rate_limit()

        url = f"{POLYGON_BASE_URL}{endpoint}"
        params['apiKey'] = self.api_key

        for attempt in range(5):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", "12"))
                        wait_time = min(retry_after, 60)
                        logger.warning(f"Rate limit exceeded (429). Waiting {wait_time} seconds.")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status >= 500:
                        wait_time = min(2 ** attempt, 30)
                        logger.warning(f"Server error ({response.status}). Retrying in {wait_time} seconds.")
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    return await response.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Request failed: {e}. Attempt {attempt + 1} of 5.")
                if attempt == 4:
                    raise
                await asyncio.sleep(min(2 ** attempt, 30))
        
        raise RuntimeError("Should not reach here")


    async def get_daily_bars(self, symbol: str, day_str: str) -> pd.DataFrame:
        """
        Get aggregate bars (OHLCV) data for a symbol for a single day.
        """
        endpoint = f"/v2/aggs/ticker/{symbol}/range/1/minute/{day_str}/{day_str}"
        
        data = await self._make_request(endpoint, {'limit': 50000})
        
        if not data.get('results'):
            return pd.DataFrame()

        df = pd.DataFrame(data['results'])
        logger.info(f"[FETCH] {day_str} raw_rows={len(df)}")

        df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        df = df.set_index("ts").sort_index()
        df = df.between_time("09:30", "16:00")
        logger.info(f"[RTH]   {day_str} rth_rows={len(df)}")

        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume","vw":"vwap","n":"transactions"})
        keep = [c for c in ["open","high","low","close","volume","vwap","transactions"] if c in df.columns]
        df = df[keep]
        
        return df


async def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description="Collect Polygon data for RL trading")
    parser.add_argument('--symbols', type=str, default='SPY', help='Comma-separated list of symbols')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--api-key', type=str, help='Polygon API key (or set POLYGON_API_KEY env var)')
    parser.add_argument('--data-dir', type=str, help='Data directory path')

    args = parser.parse_args()

    api_key = args.api_key or os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("Polygon API key required. Set POLYGON_API_KEY environment variable or use --api-key")

    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Use a sub-directory of the user-provided data-dir, or default to a project-relative path
    base_data_dir = Path(args.data_dir) if args.data_dir else Path.cwd() / "data"
    data_dir = base_data_dir / "polygon" / "historical"


    nyse = mcal.get_calendar("XNYS")
    trading_days = nyse.schedule(start_date=args.start_date, end_date=args.end_date).index

    async with PolygonDataCollector(api_key, data_dir) as collector:
        for symbol in symbols:
            logger.info(f"Starting data collection for {symbol} from {args.start_date} to {args.end_date}")
            for day in tqdm(trading_days, desc=f"Processing {symbol}"):
                day_str = day.strftime('%Y-%m-%d')
                year, month, day_of_month = day.year, day.month, day.day

                base = data_dir / f"symbol={symbol}" / f"year={year}" / f"month={month:02d}" / f"day={day_of_month:02d}"
                out_path = base / "data.parquet"

                if out_path.exists():
                    try:
                        df_existing = pd.read_parquet(out_path)
                        if len(df_existing) >= 375 and out_path.stat().st_size > 1024:
                            logger.info(f"Skipping {day_str}, already complete.")
                            continue
                    except Exception as e:
                        logger.warning(f"Could not read existing file {out_path}, will overwrite. Error: {e}")

                try:
                    df = await collector.get_daily_bars(symbol, day_str)

                    if len(df) == 0:
                        logger.warning(f"No rows after filtering; skipping write for {day_str}")
                        continue

                    logger.info(f"Creating directory: {out_path.parent}")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(out_path, engine="pyarrow")
                    logger.info(f"File exists after write: {out_path.exists()}")
                    
                    file_size = out_path.stat().st_size
                    assert file_size > 1024, "Parquet too small; investigate filters/index."
                    
                    logger.info(f"[WRITE] {day_str} rows={len(df)} size={file_size} path={out_path}")
                    print(f"OK {day_str} rows={len(df)} size={file_size}")

                except Exception as e:
                    logger.error(f"Failed to process {day_str} for {symbol}: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
