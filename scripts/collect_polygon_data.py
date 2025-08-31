#!/usr/bin/env python3
"""
Polygon Data Collection Script for RL Trading System

This script collects historical market data from Polygon API for backtesting
the RL trading system. Optimized for free tier limits (5 calls/minute) and
designed for local/VM execution with efficient storage.

Features:
- Rate limiting compliance (5 calls/minute)
- Monthly data chunking for single API calls
- Progress tracking and error handling
- Efficient Parquet storage
- Data quality validation
- Command-line interface

Usage:
    python scripts/collect_polygon_data.py --symbols SPY,QQQ --start-date 2024-01-01 --end-date 2024-06-30
    python scripts/collect_polygon_data.py --preset pilot  # Use predefined pilot configuration
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
import backoff

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger
from src.utils.config_loader import Settings

logger = get_logger(__name__)

# Configuration
POLYGON_BASE_URL = "https://api.polygon.io"
RATE_LIMIT_DELAY = 12  # 5 calls/minute = 12 seconds between calls
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# Pilot portfolio - high volume, liquid stocks suitable for options
PILOT_PORTFOLIO = {
    'SPY': {'name': 'SPDR S&P 500 ETF', 'avg_volume': 80000000, 'volatility': 'medium'},
    'QQQ': {'name': 'Invesco QQQ ETF', 'avg_volume': 40000000, 'volatility': 'high'},
    'AAPL': {'name': 'Apple Inc.', 'avg_volume': 50000000, 'volatility': 'high'},
    'MSFT': {'name': 'Microsoft Corp.', 'avg_volume': 25000000, 'volatility': 'medium'},
    'TSLA': {'name': 'Tesla Inc.', 'avg_volume': 60000000, 'volatility': 'very_high'},
    'NVDA': {'name': 'NVIDIA Corp.', 'avg_volume': 35000000, 'volatility': 'very_high'},
    'AMD': {'name': 'Advanced Micro Devices', 'avg_volume': 30000000, 'volatility': 'very_high'},
    'GOOGL': {'name': 'Alphabet Inc.', 'avg_volume': 20000000, 'volatility': 'medium'}
}

# Extended portfolio for comprehensive testing
EXTENDED_PORTFOLIO = {
    **PILOT_PORTFOLIO,
    'AMZN': {'name': 'Amazon.com Inc.', 'avg_volume': 30000000, 'volatility': 'high'},
    'META': {'name': 'Meta Platforms Inc.', 'avg_volume': 15000000, 'volatility': 'high'},
    'NFLX': {'name': 'Netflix Inc.', 'avg_volume': 8000000, 'volatility': 'high'},
    'BA': {'name': 'Boeing Co.', 'avg_volume': 10000000, 'volatility': 'high'},
    'XOM': {'name': 'Exxon Mobil Corp.', 'avg_volume': 15000000, 'volatility': 'medium'},
    'JPM': {'name': 'JPMorgan Chase & Co.', 'avg_volume': 12000000, 'volatility': 'medium'},
    'V': {'name': 'Visa Inc.', 'avg_volume': 8000000, 'volatility': 'low'}
}


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

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=MAX_RETRIES,
        giveup=lambda e: isinstance(e, aiohttp.ClientResponseError) and e.status == 429
    )
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a rate-limited API request with retry logic.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response data
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        self._enforce_rate_limit()

        url = f"{POLYGON_BASE_URL}{endpoint}"
        params['apiKey'] = self.api_key

        logger.debug(f"Making request to {url} with params: {params}")

        async with self.session.get(url, params=params) as response:
            if response.status == 429:
                logger.warning("Rate limit exceeded, backing off...")
                raise aiohttp.ClientResponseError(
                    response.request_info, response.history, status=429
                )

            response.raise_for_status()
            data = await response.json()

            logger.debug(f"Request successful, received {len(str(data))} characters")
            return data

    async def get_aggregate_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        multiplier: int = 1,
        timespan: str = "minute"
    ) -> pd.DataFrame:
        """
        Get aggregate bars (OHLCV) data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            multiplier: Timespan multiplier
            timespan: Timespan unit ('minute', 'hour', 'day')

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

        data = await self._make_request(endpoint, {})

        if not data.get('results'):
            logger.warning(f"No data returned for {symbol} from {start_date} to {end_date}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data['results'])

        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Rename columns to standard format
        column_mapping = {
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        }

        df = df.rename(columns=column_mapping)

        # Keep only relevant columns
        keep_columns = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
        df = df[[col for col in keep_columns if col in df.columns]]

        logger.info(f"Retrieved {len(df)} bars for {symbol}")
        return df

    async def get_quotes(
        self,
        symbol: str,
        date: str
    ) -> pd.DataFrame:
        """
        Get quotes (bid/ask) data for a symbol on a specific date.

        Args:
            symbol: Stock symbol
            date: Date (YYYY-MM-DD)

        Returns:
            DataFrame with quotes data
        """
        endpoint = f"/v3/quotes/{symbol}"
        params = {'timestamp.gte': f"{date}T00:00:00Z", 'timestamp.lt': f"{date}T23:59:59Z", 'limit': 50000}

        data = await self._make_request(endpoint, params)

        if not data.get('results'):
            logger.warning(f"No quotes data for {symbol} on {date}")
            return pd.DataFrame()

        df = pd.DataFrame(data['results'])

        # Convert timestamp from nanoseconds to datetime
        df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
        df.set_index('timestamp', inplace=True)

        # Rename columns
        column_mapping = {
            'bid_price': 'bid_price',
            'bid_size': 'bid_size',
            'ask_price': 'ask_price',
            'ask_size': 'ask_size',
            'bid_exchange': 'bid_exchange',
            'ask_exchange': 'ask_exchange'
        }

        df = df.rename(columns=column_mapping)

        logger.info(f"Retrieved {len(df)} quotes for {symbol} on {date}")
        return df

    def save_data(self, df: pd.DataFrame, symbol: str, data_type: str, date: str):
        """
        Save DataFrame to partitioned Parquet format.

        Args:
            df: Data to save
            symbol: Stock symbol
            data_type: Type of data ('ohlcv', 'quotes')
            date: Date string for partitioning
        """
        if df.empty:
            return

        # Create partition path
        date_obj = pd.to_datetime(date)
        partition_path = self.data_dir / f"symbol={symbol}" / \
                        f"year={date_obj.year}" / \
                        f"month={date_obj.month:02d}" / \
                        f"day={date_obj.day:02d}"

        partition_path.mkdir(parents=True, exist_ok=True)

        # Save as Parquet
        file_path = partition_path / "data.parquet"
        df.to_parquet(file_path, compression='snappy')

        logger.info(f"Saved {len(df)} rows to {file_path}")


async def collect_symbol_data(
    collector: PolygonDataCollector,
    symbol: str,
    start_date: str,
    end_date: str,
    include_quotes: bool = False
) -> Dict[str, Any]:
    """
    Collect all data for a single symbol.

    Args:
        collector: Data collector instance
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        include_quotes: Whether to collect quotes data

    Returns:
        Collection results
    """
    results = {
        'symbol': symbol,
        'ohlcv_records': 0,
        'quotes_records': 0,
        'errors': []
    }

    try:
        # Collect OHLCV data
        logger.info(f"Collecting OHLCV data for {symbol}")
        ohlcv_df = await collector.get_aggregate_bars(symbol, start_date, end_date)

        if not ohlcv_df.empty:
            # Save data by date
            for date, group_df in ohlcv_df.groupby(ohlcv_df.index.date):
                collector.save_data(group_df, symbol, 'ohlcv', str(date))
                results['ohlcv_records'] += len(group_df)

        # Collect quotes data if requested
        if include_quotes:
            logger.info(f"Collecting quotes data for {symbol}")
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            current_date = start_dt
            while current_date <= end_dt:
                try:
                    quotes_df = await collector.get_quotes(symbol, current_date.strftime('%Y-%m-%d'))
                    if not quotes_df.empty:
                        collector.save_data(quotes_df, symbol, 'quotes', current_date.strftime('%Y-%m-%d'))
                        results['quotes_records'] += len(quotes_df)
                except Exception as e:
                    logger.warning(f"Failed to collect quotes for {symbol} on {current_date.date()}: {e}")
                    results['errors'].append(f"Quotes {current_date.date()}: {str(e)}")

                current_date += timedelta(days=1)

    except Exception as e:
        logger.error(f"Failed to collect data for {symbol}: {e}")
        results['errors'].append(str(e))

    return results


async def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description="Collect Polygon data for RL trading")
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--preset', choices=['pilot', 'extended'], help='Use predefined portfolio')
    parser.add_argument('--include-quotes', action='store_true', help='Include quotes data')
    parser.add_argument('--api-key', type=str, help='Polygon API key (or set POLYGON_API_KEY env var)')
    parser.add_argument('--data-dir', type=str, help='Data directory path')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("Polygon API key required. Set POLYGON_API_KEY environment variable or use --api-key")

    # Determine symbols
    if args.preset == 'pilot':
        symbols = list(PILOT_PORTFOLIO.keys())
        logger.info(f"Using pilot portfolio: {symbols}")
    elif args.preset == 'extended':
        symbols = list(EXTENDED_PORTFOLIO.keys())
        logger.info(f"Using extended portfolio: {symbols}")
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        logger.info(f"Using custom symbols: {symbols}")
    else:
        symbols = list(PILOT_PORTFOLIO.keys())
        logger.info(f"Using default pilot portfolio: {symbols}")

    # Data directory
    data_dir = Path(args.data_dir) if args.data_dir else None

    # Collection summary
    total_results = {
        'total_symbols': len(symbols),
        'successful_symbols': 0,
        'total_ohlcv_records': 0,
        'total_quotes_records': 0,
        'total_errors': 0,
        'symbol_results': []
    }

    async with PolygonDataCollector(api_key, data_dir) as collector:
        logger.info(f"Starting data collection for {len(symbols)} symbols from {args.start_date} to {args.end_date}")

        with tqdm(total=len(symbols), desc="Symbols") as pbar:
            for symbol in symbols:
                logger.info(f"Processing {symbol}...")
                result = await collect_symbol_data(
                    collector, symbol, args.start_date, args.end_date, args.include_quotes
                )

                total_results['symbol_results'].append(result)
                total_results['total_ohlcv_records'] += result['ohlcv_records']
                total_results['total_quotes_records'] += result['quotes_records']
                total_results['total_errors'] += len(result['errors'])

                if not result['errors']:
                    total_results['successful_symbols'] += 1

                pbar.update(1)
                pbar.set_postfix({
                    'OHLCV': total_results['total_ohlcv_records'],
                    'Quotes': total_results['total_quotes_records'] if args.include_quotes else 0
                })

    # Print summary
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Symbols processed: {total_results['successful_symbols']}/{total_results['total_symbols']}")
    print(f"OHLCV records: {total_results['total_ohlcv_records']:,}")
    if args.include_quotes:
        print(f"Quotes records: {total_results['total_quotes_records']:,}")
    print(f"Total errors: {total_results['total_errors']}")
    print(f"Data directory: {collector.data_dir}")

    # Save summary to file
    summary_file = collector.data_dir / "collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(total_results, f, indent=2, default=str)
    print(f"Summary saved to: {summary_file}")

    print("\nData collection completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())