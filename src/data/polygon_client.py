"""
Polygon API client for historical market data.

This module provides interfaces for accessing historical market data
from Polygon.io with efficient data retrieval and caching.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not available. Polygon client will be disabled.")

try:
    from pydantic import BaseModel, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logging.warning("pydantic not available. Data validation will be disabled.")

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class PolygonAPIError(Exception):
    """Exception raised for Polygon API errors."""
    pass


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = 5):
        self.calls_per_minute = calls_per_minute
        self.call_times: List[datetime] = []

    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = datetime.now()

        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if (now - t).seconds < 60]

        if len(self.call_times) >= self.calls_per_minute:
            # Wait until the oldest call is more than 1 minute old
            wait_time = 60 - (now - self.call_times[0]).seconds
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)

        self.call_times.append(now)


if PYDANTIC_AVAILABLE:
    class PolygonBar(BaseModel):
        """Pydantic model for Polygon bar data."""
        timestamp: int
        open: float
        high: float
        low: float
        close: float
        volume: float
        vwap: Optional[float] = None
        transactions: Optional[int] = None

        @validator('timestamp')
        def timestamp_must_be_positive(cls, v):
            if v <= 0:
                raise ValueError('timestamp must be positive')
            return v

    class PolygonQuote(BaseModel):
        """Pydantic model for Polygon quote data."""
        timestamp: int
        bid_price: float
        bid_size: int
        ask_price: float
        ask_size: int
        bid_exchange: Optional[int] = None
        ask_exchange: Optional[int] = None

    class PolygonTrade(BaseModel):
        """Pydantic model for Polygon trade data."""
        timestamp: int
        price: float
        size: int
        exchange: int
        conditions: Optional[List[int]] = None
        trade_id: Optional[str] = None
else:
    # Fallback classes if pydantic not available
    PolygonBar = dict
    PolygonQuote = dict
    PolygonTrade = dict


class PolygonClient:
    """
    Polygon API client for historical market data retrieval.

    This class handles connection to Polygon API and provides methods
    for retrieving historical market data with efficient caching.
    """

    def __init__(self, settings: Settings):
        """
        Initialize Polygon client.

        Args:
            settings: Configuration settings
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for Polygon client")

        self.settings = settings
        self.client = httpx.AsyncClient(timeout=30.0)
        self.rate_limiter = RateLimiter(calls_per_minute=5)  # Conservative default
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_dir = Path("data/cache/polygon")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API key from environment
        self.api_key = settings.get('data', 'polygon_api_key')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")

        # Base URL
        self.base_url = "https://api.polygon.io"

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make authenticated request to Polygon API with rate limiting and retries.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response data
        """
        await self.rate_limiter.wait_if_needed()

        url = f"{self.base_url}{endpoint}"
        params['apiKey'] = self.api_key

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code >= 500:  # Server error
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Server error, retrying in {wait_time} seconds")
                        await asyncio.sleep(wait_time)
                        continue
                logger.error(f"HTTP error: {e}")
                raise PolygonAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise PolygonAPIError(f"Request failed after {max_retries} attempts: {e}")

        raise PolygonAPIError("Max retries exceeded")

    def _get_cache_path(self, symbol: str, date: str) -> Path:
        """
        Get cache file path for symbol and date.

        Args:
            symbol: Instrument symbol
            date: Date string (YYYY-MM-DD)

        Returns:
            Path to cache file
        """
        year, month, day = date.split('-')
        return self.cache_dir / f"symbol={symbol}" / f"year={year}" / f"month={month}" / f"day={day}" / "data.parquet"

    def _load_from_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available.

        Args:
            symbol: Instrument symbol
            start_date: Start date
            end_date: End date

        Returns:
            Cached DataFrame or None
        """
        # For simplicity, check if we have data for the date range
        # In a full implementation, we'd need to check all dates in range
        cache_key = f"{symbol}_{start_date}_{end_date}"
        cache_file = self._get_cache_path(symbol, start_date)

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached data for {symbol} from {cache_file}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        return None

    def _save_to_cache(self, df: pd.DataFrame, symbol: str, date: str):
        """
        Save data to partitioned Parquet cache.

        Args:
            df: DataFrame to cache
            symbol: Instrument symbol
            date: Date string
        """
        if df.empty:
            return

        cache_file = self._get_cache_path(symbol, date)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Add partition columns
            df_cache = df.copy()
            df_cache['symbol'] = symbol
            df_cache['year'] = int(date.split('-')[0])
            df_cache['month'] = int(date.split('-')[1])
            df_cache['day'] = int(date.split('-')[2])

            # Write partitioned Parquet
            table = pa.Table.from_pandas(df_cache)
            pq.write_table(table, str(cache_file))
            logger.info(f"Cached data to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    async def _get_aggregates(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        multiplier: int = 1,
        timespan: str = 'minute'
    ) -> pd.DataFrame:
        """
        Get aggregate bars from Polygon API.

        Args:
            symbol: Instrument symbol
            start_date: Start date
            end_date: End date
            multiplier: Multiplier for timespan
            timespan: Timespan (minute, hour, day, etc.)

        Returns:
            DataFrame with bar data
        """
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

        data = await self._make_request(endpoint, {})

        if not data.get('results'):
            return pd.DataFrame()

        bars = []
        for result in data['results']:
            bar = {
                'timestamp': result['t'],
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v'],
                'vwap': result.get('vw', None),
                'transactions': result.get('n', None)
            }
            bars.append(bar)

        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    async def _get_trades(self, symbol: str, date: str) -> pd.DataFrame:
        """
        Get trade data from Polygon API.

        Args:
            symbol: Instrument symbol
            date: Date string

        Returns:
            DataFrame with trade data
        """
        endpoint = f"/v3/trades/{symbol}"

        params = {
            'timestamp.gte': f"{date}T00:00:00.000Z",
            'timestamp.lt': f"{date}T23:59:59.999Z",
            'limit': 50000
        }

        data = await self._make_request(endpoint, params)

        if not data.get('results'):
            return pd.DataFrame()

        trades = []
        for result in data['results']:
            trade = {
                'timestamp': result['sip_timestamp'] if 'sip_timestamp' in result else result['participant_timestamp'],
                'price': result['price'],
                'size': result['size'],
                'exchange': result['exchange'],
                'conditions': result.get('conditions', None),
                'trade_id': result.get('trade_id', None)
            }
            trades.append(trade)

        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        df.set_index('timestamp', inplace=True)

        return df

    async def _get_quotes(self, symbol: str, date: str) -> pd.DataFrame:
        """
        Get quote data from Polygon API.

        Args:
            symbol: Instrument symbol
            date: Date string

        Returns:
            DataFrame with quote data
        """
        endpoint = f"/v3/quotes/{symbol}"

        params = {
            'timestamp.gte': f"{date}T00:00:00.000Z",
            'timestamp.lt': f"{date}T23:59:59.999Z",
            'limit': 50000
        }

        data = await self._make_request(endpoint, params)

        if not data.get('results'):
            return pd.DataFrame()

        quotes = []
        for result in data['results']:
            quote = {
                'timestamp': result['sip_timestamp'] if 'sip_timestamp' in result else result['participant_timestamp'],
                'bid_price': result['bid_price'],
                'bid_size': result['bid_size'],
                'ask_price': result['ask_price'],
                'ask_size': result['ask_size'],
                'bid_exchange': result.get('bid_exchange', None),
                'ask_exchange': result.get('ask_exchange', None)
            }
            quotes.append(quote)

        df = pd.DataFrame(quotes)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        df.set_index('timestamp', inplace=True)

        return df

    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        dataset: str = 'continuous',
        stype: str = 'raw_symbol',
        schema: str = 'ohlcv-1s',
        freq: str = '1min'
    ) -> pd.DataFrame:
        """
        Retrieve historical market data from Polygon.

        Args:
            symbol: Instrument symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            dataset: Dataset name (ignored for Polygon)
            stype: Symbol type (ignored for Polygon)
            schema: Data schema ('ohlcv-1s', 'topbook-1s')
            freq: Frequency for resampling ('1min', '1h', etc.)

        Returns:
            DataFrame with historical market data
        """
        try:
            # Convert dates to string if datetime objects
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')

            # Check cache first
            cached_data = self._load_from_cache(symbol, start_date, end_date)
            if cached_data is not None:
                return cached_data

            # Determine data type based on schema
            if schema == 'ohlcv-1s':
                # Get aggregate bars
                if freq == '1min':
                    multiplier = 1
                    timespan = 'minute'
                elif freq == '1h':
                    multiplier = 1
                    timespan = 'hour'
                elif freq == '1d':
                    multiplier = 1
                    timespan = 'day'
                else:
                    # Parse custom frequency
                    if freq.endswith('min'):
                        multiplier = int(freq[:-3])
                        timespan = 'minute'
                    elif freq.endswith('h'):
                        multiplier = int(freq[:-1])
                        timespan = 'hour'
                    else:
                        raise ValueError(f"Unsupported frequency: {freq}")

                df = asyncio.run(self._get_aggregates(symbol, start_date, end_date, multiplier, timespan))

            elif schema == 'topbook-1s':
                # Get quote data for order book
                df = asyncio.run(self._get_quotes(symbol, start_date))

            else:
                raise ValueError(f"Unsupported schema: {schema}")

            # Cache the data
            self._save_to_cache(df, symbol, start_date)

            return df

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise

    def get_minute_bars(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get minute bar data for an instrument.

        Args:
            symbol: Instrument symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with minute bars
        """
        return self.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            schema='ohlcv-1s',
            freq='1min'
        )

    def get_tick_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get tick-level data for an instrument.

        Args:
            symbol: Instrument symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with tick data
        """
        return self.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            schema='topbook-1s',
            freq='1s'
        )

    def get_order_flow_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get order flow imbalance data.

        Args:
            symbol: Instrument symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with order flow data
        """
        # Get quote data
        quotes = self.get_tick_data(symbol, start_date, end_date)

        if quotes.empty:
            return pd.DataFrame()

        # Calculate order flow metrics
        quotes['mid_price'] = (quotes['bid_price'] + quotes['ask_price']) / 2
        quotes['spread'] = quotes['ask_price'] - quotes['bid_price']
        quotes['imbalance'] = (quotes['bid_size'] - quotes['ask_size']) / (quotes['bid_size'] + quotes['ask_size'])

        return quotes[['mid_price', 'spread', 'imbalance', 'bid_size', 'ask_size']]

    def get_vwap_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get VWAP data for an instrument.

        Args:
            symbol: Instrument symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with VWAP data
        """
        # Get trade data
        trades = asyncio.run(self._get_trades(symbol, start_date))

        if trades.empty:
            return pd.DataFrame()

        # Calculate VWAP for each minute
        trades['price_volume'] = trades['price'] * trades['size']

        vwap_data = trades.resample('1min').agg({
            'price_volume': 'sum',
            'size': 'sum'
        })

        vwap_data['vwap'] = vwap_data['price_volume'] / vwap_data['size']
        vwap_data = vwap_data[['vwap']].dropna()

        return vwap_data

    def get_liquidity_metrics(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Calculate liquidity metrics from order book data.

        Args:
            symbol: Instrument symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with liquidity metrics
        """
        # Get order book data
        quotes = self.get_tick_data(symbol, start_date, end_date)

        if quotes.empty:
            return pd.DataFrame()

        # Calculate liquidity metrics
        liquidity_metrics = pd.DataFrame(index=quotes.index)

        liquidity_metrics['spread'] = quotes['ask_price'] - quotes['bid_price']
        liquidity_metrics['spread_pct'] = liquidity_metrics['spread'] / ((quotes['bid_price'] + quotes['ask_price']) / 2)
        liquidity_metrics['bid_depth'] = quotes['bid_size']
        liquidity_metrics['ask_depth'] = quotes['ask_size']
        liquidity_metrics['total_depth'] = liquidity_metrics['bid_depth'] + liquidity_metrics['ask_depth']

        # Resample to 1 minute
        liquidity_metrics = liquidity_metrics.resample('1min').agg({
            'spread': 'mean',
            'spread_pct': 'mean',
            'bid_depth': 'sum',
            'ask_depth': 'sum',
            'total_depth': 'sum'
        }).dropna()

        return liquidity_metrics

    def get_trading_session_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get data filtered to regular trading hours.

        Args:
            symbol: Instrument symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with RTH-filtered data
        """
        # Get minute data
        data = self.get_minute_bars(symbol, start_date, end_date)

        if data.empty:
            return data

        # Filter to regular trading hours (9:30 AM - 4:00 PM ET)
        rth_start = '09:30'
        rth_end = '16:00'

        # Create time mask for RTH
        time_mask = (
            (data.index.time >= pd.to_datetime(rth_start).time()) &
            (data.index.time <= pd.to_datetime(rth_end).time())
        )

        # Apply mask
        rth_data = data[time_mask]

        return rth_data

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear data cache.

        Args:
            symbol: Optional symbol to clear cache for. If None, clears all cache.
        """
        if symbol:
            # Clear cache for specific symbol
            symbol_dir = self.cache_dir / f"symbol={symbol}"
            if symbol_dir.exists():
                import shutil
                shutil.rmtree(symbol_dir)
                logger.info(f"Cleared cache for symbol: {symbol}")
        else:
            # Clear all cache
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared all cache")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {
                'cache_directory': str(self.cache_dir),
                'total_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'symbols': []
            }

        total_size = 0
        symbols = []
        total_files = 0

        for symbol_dir in self.cache_dir.glob("symbol=*"):
            if symbol_dir.is_dir():
                symbol = symbol_dir.name.split('=')[1]
                symbols.append(symbol)

                for parquet_file in symbol_dir.rglob("*.parquet"):
                    total_files += 1
                    total_size += parquet_file.stat().st_size

        return {
            'cache_directory': str(self.cache_dir),
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'symbols': symbols
        }

    async def aclose(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.aclose())