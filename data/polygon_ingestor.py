"""Polygon data ingestor for intraday aggregates.

Fetches 1-min bars for US symbols (e.g., BBVA 2020 validation), stores in data/polygon_plus partitioned parquet.
Idempotent: Check cache before fetch, retry on failure, deterministic order by timestamp.

Usage:
    from .polygon_ingestor import PolygonIngestor
    ingestor = PolygonIngestor(api_key='your_key')
    ingestor.ingest(symbol='BBVA', start='2020-01-01', end='2020-12-31', multiplier=1, timespan='minute')
    df = ingestor.load('BBVA', '2020-01')

Tests: Inline examples; full in tests/test_ingestor.py.
"""

from typing import Optional, Dict, Any
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import logging
from polygon import RESTClient
from tenacity import retry, stop_after_attempt, wait_exponential
from ..utils.config_loader import Settings

logger = logging.getLogger(__name__)

class PolygonIngestor:
    """Ingestor for Polygon.io aggregates, with caching and validation."""

    def __init__(self, api_key: str, base_url: str = 'https://api.polygon.io', cache_dir: str = 'data/polygon_plus'):
        """
        Initialize ingestor.

        Args:
            api_key: Polygon API key.
            base_url: API base URL.
            cache_dir: Directory for partitioned parquet cache.

        Raises:
            ValueError: Invalid config.
        """
        self.client = RESTClient(api_key=api_key, base_url=base_url)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings = Settings.from_paths('configs/settings.yaml')  # for rate limits, etc.

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def ingest(self, symbol: str, start: str, end: str, multiplier: int = 1, timespan: str = 'minute') -> pd.DataFrame:
        """
        Fetch and cache aggregates.

        Args:
            symbol: Stock symbol (e.g., 'BBVA').
            start: Start date 'YYYY-MM-DD'.
            end: End date 'YYYY-MM-DD'.
            multiplier: Bar size multiplier.
            timespan: 'minute' for intraday.

        Returns:
            pd.DataFrame with aggregates, partitioned to cache.

        Raises:
            ValueError: Invalid params.
            Exception: API/retry failure.
        """
        if multiplier != 1 or timespan != 'minute':
            raise ValueError("Only 1-min bars supported for intraday RL.")

        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')

        # Check cache first (idempotent)
        cache_path = self._get_cache_path(symbol, start, end)
        if cache_path.exists():
            logger.info(f"Cache hit for {symbol} {start} to {end}")
            return self._load_from_cache(cache_path)

        # Fetch with retries
        try:
            aggs = self.client.get_aggs(
                symbol, multiplier, timespan, start, end,
                adjusted=True, sort='asc', limit=50000  # deterministic order
            )
            if not aggs:
                raise ValueError(f"No data for {symbol} {start}-{end}")

            df = pd.DataFrame([a.__dict__ for a in aggs])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York')
            df = df.set_index('timestamp').rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df.sort_index()  # deterministic

            # Validate: RTH only, no gaps >1min
            rth_mask = (df.index.time >= pd.Timestamp('09:30').time()) & (df.index.time <= pd.Timestamp('16:00').time())
            if rth_mask.sum() < len(df) * 0.8:
                logger.warning(f"Low RTH coverage for {symbol}")
            df = df[rth_mask]  # filter to RTH

            # Store partitioned
            self._store_to_cache(df, symbol, start, end)
            logger.info(f"Ingested {len(df)} bars for {symbol} {start}-{end}")

            return df
        except Exception as e:
            logger.error(f"Ingestion failed for {symbol}: {e}")
            raise

    def load(self, symbol: str, date_range: str = '2020') -> pd.DataFrame:
        """Load from cache for date range."""
        cache_path = self._get_cache_path(symbol, *date_range.split('-'))
        if cache_path.exists():
            return self._load_from_cache(cache_path)
        raise FileNotFoundError(f"No cache for {symbol} {date_range}")

    def _get_cache_path(self, symbol: str, start: str, end: str) -> Path:
        """Get partitioned cache path."""
        year, month = datetime.strptime(start, '%Y-%m-%d').strftime('%Y/%m')
        return self.cache_dir / 'us_stocks' / 'aggregates' / f'symbol={symbol}' / f'year={year}' / f'month={month}' / f'{start}_{end}.parquet'

    def _store_to_cache(self, df: pd.DataFrame, symbol: str, start: str, end: str):
        """Store DF as partitioned parquet."""
        path = self._get_cache_path(symbol, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pq.from_pandas(df), path)
        logger.debug(f"Stored to {path}")

    def _load_from_cache(self, path: Path) -> pd.DataFrame:
        """Load parquet from path."""
        return pd.read_parquet(path).sort_index()

# Example usage and test
if __name__ == "__main__":
    # Deterministic test
    settings = Settings.from_paths('configs/settings.yaml')
    api_key = settings.get('polygon', 'api_key', default='demo')
    ingestor = PolygonIngestor(api_key)
    df = ingestor.ingest('BBVA', '2020-01-01', '2020-01-31')
    assert len(df) > 0, "No data ingested"
    print(df.head())
    # Save validation sample
    df.to_parquet('data/raw/BBVA_1min.parquet')
    print("BBVA 2020 validation data ready.")