"""
Polygon Data Ingestion Pipeline

This module provides a comprehensive data ingestion pipeline for fetching and storing
Polygon market data with support for bulk operations, incremental updates, and
partitioned Parquet storage.

Key Features:
- Bulk data fetching for multiple symbols and data types
- Incremental updates with metadata tracking
- Parallel processing for performance optimization
- Comprehensive error handling and retry logic
- Partitioned Parquet storage compatible with unified data loader
- Progress tracking and logging
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm

from .polygon_client import PolygonClient, PolygonAPIError
from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionMetadata:
    """Metadata for tracking ingestion progress and state."""
    symbol: str
    data_type: str
    last_update: Optional[datetime] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    total_records: int = 0
    last_successful_fetch: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None


class PolygonDataIngestor:
    """
    Comprehensive data ingestion pipeline for Polygon market data.

    This class manages the entire data ingestion process from API fetching
    to partitioned Parquet storage, with support for incremental updates,
    parallel processing, and robust error handling.
    """

    SUPPORTED_DATA_TYPES = ['ohlcv', 'quotes', 'trades']
    DEFAULT_DATA_DIR = Path("data/polygon/historical")

    def __init__(self, settings: Settings):
        """
        Initialize the Polygon data ingestor.

        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.polygon_client = PolygonClient(settings)

        # Configuration (use nested Settings.get to avoid dict chaining issues)
        self.data_dir = Path(settings.get('data', 'polygon', 'data_dir', default=self.DEFAULT_DATA_DIR))
        self.metadata_dir = self.data_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Performance settings
        self.max_workers = settings.get('data', 'max_workers', default=4)
        self.batch_size = settings.get('data', 'batch_size', default=1000)
        self.chunk_size = settings.get('data', 'chunk_size', default=50000)

        # Rate limiting and retry settings
        self.retry_attempts = settings.get('data', 'polygon', 'retry', 'attempts', default=3)
        self.retry_backoff = settings.get('data', 'polygon', 'retry', 'backoff_factor', default=2.0)

        # Storage settings
        self.compression = settings.get('data', 'compression', default='snappy')
        self.row_group_size = settings.get('data', 'row_group_size', default=100000)

        # Validation settings
        self.enable_validation = settings.get('data', 'validation', 'enabled', default=True)

        # Metadata cache
        self.metadata_cache: Dict[str, IngestionMetadata] = {}
        self._load_metadata_cache()

        logger.info(f"Initialized PolygonDataIngestor with data directory: {self.data_dir}")

    def _load_metadata_cache(self):
        """Load metadata cache from disk."""
        if not self.metadata_dir.exists():
            return

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)

                # Convert string dates back to datetime
                if data.get('last_update'):
                    data['last_update'] = datetime.fromisoformat(data['last_update'])
                if data.get('date_range_start'):
                    data['date_range_start'] = datetime.fromisoformat(data['date_range_start'])
                if data.get('date_range_end'):
                    data['date_range_end'] = datetime.fromisoformat(data['date_range_end'])
                if data.get('last_successful_fetch'):
                    data['last_successful_fetch'] = datetime.fromisoformat(data['last_successful_fetch'])

                metadata = IngestionMetadata(**data)
                cache_key = f"{metadata.symbol}_{metadata.data_type}"
                self.metadata_cache[cache_key] = metadata

            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")

    def _save_metadata(self, metadata: IngestionMetadata):
        """Save metadata to disk."""
        cache_key = f"{metadata.symbol}_{metadata.data_type}"
        self.metadata_cache[cache_key] = metadata

        metadata_file = self.metadata_dir / f"{cache_key}.json"

        try:
            data = {
                'symbol': metadata.symbol,
                'data_type': metadata.data_type,
                'last_update': metadata.last_update.isoformat() if metadata.last_update else None,
                'date_range_start': metadata.date_range_start.isoformat() if metadata.date_range_start else None,
                'date_range_end': metadata.date_range_end.isoformat() if metadata.date_range_end else None,
                'total_records': metadata.total_records,
                'last_successful_fetch': metadata.last_successful_fetch.isoformat() if metadata.last_successful_fetch else None,
                'error_count': metadata.error_count,
                'last_error': metadata.last_error
            }

            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metadata for {cache_key}: {e}")

    def _get_storage_path(self, symbol: str, date: Union[str, datetime]) -> Path:
        """
        Get the partitioned storage path for a symbol and date.

        Args:
            symbol: Instrument symbol
            date: Date for partitioning

        Returns:
            Path to the partitioned data file
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)

        year = date.year
        month = date.month
        day = date.day

        return self.data_dir / f"symbol={symbol}" / f"year={year}" / f"month={month:02d}" / f"day={day:02d}" / "data.parquet"

    def _ensure_partition_directory(self, symbol: str, date: Union[str, datetime]) -> Path:
        """
        Ensure the partition directory exists and return the data file path.

        Args:
            symbol: Instrument symbol
            date: Date for partitioning

        Returns:
            Path to the data file
        """
        data_file = self._get_storage_path(symbol, date)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        return data_file

    async def _fetch_single_symbol_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data_type: str,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Fetch data for a single symbol and data type.

        Args:
            symbol: Instrument symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_type: Type of data ('ohlcv', 'quotes', 'trades')
            progress_callback: Optional progress callback function

        Returns:
            DataFrame with fetched data
        """
        try:
            if data_type == 'ohlcv':
                # Fetch OHLCV data (minute bars) without nested event-loop issues
                try:
                    # Prefer async path to avoid asyncio.run() inside a running loop
                    df = await self.polygon_client._get_aggregates(symbol, start_date, end_date, 1, 'minute')
                except Exception:
                    # Fallback to sync helper (may raise if loop is running)
                    df = self.polygon_client.get_minute_bars(symbol, start_date, end_date)

            elif data_type == 'quotes':
                # Fetch quote data
                df = await self.polygon_client._get_quotes(symbol, start_date)

            elif data_type == 'trades':
                # Fetch trade data
                df = await self.polygon_client._get_trades(symbol, start_date)

            else:
                raise ValueError(f"Unsupported data type: {data_type}")

            if progress_callback:
                progress_callback(symbol, data_type, len(df))

            return df

        except Exception as e:
            logger.error(f"Failed to fetch {data_type} data for {symbol}: {e}")
            raise

    async def _fetch_data_with_retry(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data_type: str,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Fetch data with retry logic.

        Args:
            symbol: Instrument symbol
            start_date: Start date
            end_date: End date
            data_type: Data type
            progress_callback: Progress callback

        Returns:
            DataFrame with data
        """
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                return await self._fetch_single_symbol_data(
                    symbol, start_date, end_date, data_type, progress_callback
                )
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_backoff ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol} {data_type}, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.retry_attempts} attempts failed for {symbol} {data_type}: {e}")

        raise last_exception

    def _store_partitioned_data(self, df: pd.DataFrame, symbol: str, date: Union[str, datetime]):
        """
        Store DataFrame in partitioned Parquet format.

        Args:
            df: DataFrame to store
            symbol: Instrument symbol
            date: Date for partitioning
        """
        if df.empty:
            return

        data_file = self._ensure_partition_directory(symbol, date)

        try:
            # Add partition columns
            df_storage = df.copy()
            df_storage['symbol'] = symbol

            if isinstance(date, str):
                date_obj = pd.to_datetime(date)
            else:
                date_obj = date

            df_storage['year'] = date_obj.year
            df_storage['month'] = date_obj.month
            df_storage['day'] = date_obj.day

            # Convert to PyArrow table
            table = pa.Table.from_pandas(df_storage)

            # Write with compression
            pq.write_table(
                table,
                str(data_file),
                compression=self.compression,
                row_group_size=self.row_group_size
            )

            logger.debug(f"Stored {len(df)} records for {symbol} on {date}")

        except Exception as e:
            logger.error(f"Failed to store data for {symbol} on {date}: {e}")
            raise

    def _get_incremental_date_range(
        self,
        symbol: str,
        data_type: str,
        start_date: str,
        end_date: str
    ) -> tuple[str, str]:
        """
        Get the date range for incremental updates.

        Args:
            symbol: Instrument symbol
            data_type: Data type
            start_date: Requested start date
            end_date: Requested end date

        Returns:
            Tuple of (actual_start_date, actual_end_date)
        """
        cache_key = f"{symbol}_{data_type}"
        metadata = self.metadata_cache.get(cache_key)

        if metadata and metadata.last_successful_fetch:
            # Start from the day after last successful fetch
            incremental_start = metadata.last_successful_fetch + timedelta(days=1)
            incremental_start_str = incremental_start.strftime('%Y-%m-%d')

            # Use the later of requested start and incremental start
            actual_start = max(start_date, incremental_start_str)
        else:
            actual_start = start_date

        return actual_start, end_date

    def _validate_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Validate and clean fetched data.

        Args:
            df: DataFrame to validate
            data_type: Type of data

        Returns:
            Validated and cleaned DataFrame
        """
        if df.empty or not self.enable_validation:
            return df

        original_len = len(df)

        # Basic validation based on data type
        if data_type == 'ohlcv':
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Remove rows with invalid OHLC values
            valid_mask = (
                (df['open'] > 0) &
                (df['high'] > 0) &
                (df['low'] > 0) &
                (df['close'] > 0) &
                (df['volume'] >= 0)
            )
            df = df[valid_mask]

        elif data_type == 'quotes':
            required_cols = ['bid_price', 'bid_size', 'ask_price', 'ask_size']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Remove invalid quotes
            valid_mask = (
                (df['bid_price'] > 0) &
                (df['ask_price'] > 0) &
                (df['bid_size'] >= 0) &
                (df['ask_size'] >= 0) &
                (df['bid_price'] <= df['ask_price'])
            )
            df = df[valid_mask]

        elif data_type == 'trades':
            required_cols = ['price', 'size']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Remove invalid trades
            valid_mask = (
                (df['price'] > 0) &
                (df['size'] > 0)
            )
            df = df[valid_mask]

        # Remove duplicates based on timestamp
        if not df.empty and 'timestamp' in df.index.names:
            df = df[~df.index.duplicated(keep='first')]

        cleaned_len = len(df)
        if cleaned_len < original_len:
            logger.info(f"Validation removed {original_len - cleaned_len} invalid records")

        return df

    async def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        data_types: Union[str, List[str]] = 'ohlcv',
        incremental: bool = False,
        dry_run: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Fetch historical data for multiple symbols and data types.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_types: Single data type or list of data types
            incremental: Whether to perform incremental update
            dry_run: If True, only simulate the operation
            progress_callback: Optional progress callback

        Returns:
            Dictionary with operation results
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        if isinstance(data_types, str):
            data_types = [data_types]

        # Validate inputs
        for data_type in data_types:
            if data_type not in self.SUPPORTED_DATA_TYPES:
                raise ValueError(f"Unsupported data type: {data_type}")

        results = {
            'total_symbols': len(symbols),
            'total_data_types': len(data_types),
            'successful_fetches': 0,
            'failed_fetches': 0,
            'total_records': 0,
            'errors': [],
            'symbol_results': {}
        }

        logger.info(f"Starting data fetch for {len(symbols)} symbols, {len(data_types)} data types")

        for symbol in tqdm(symbols, desc="Symbols", disable=dry_run):
            symbol_results = {
                'data_types_processed': 0,
                'records_fetched': 0,
                'errors': []
            }

            for data_type in data_types:
                try:
                    # Determine actual date range for incremental updates
                    if incremental:
                        actual_start, actual_end = self._get_incremental_date_range(
                            symbol, data_type, start_date, end_date
                        )
                    else:
                        actual_start, actual_end = start_date, end_date

                    if dry_run:
                        logger.info(f"[DRY RUN] Would fetch {data_type} data for {symbol} from {actual_start} to {actual_end}")
                        continue

                    # Fetch data
                    df = await self._fetch_data_with_retry(
                        symbol, actual_start, actual_end, data_type, progress_callback
                    )

                    # Validate data
                    df = self._validate_data(df, data_type)

                    if not df.empty:
                        # Store data by date partitions
                        date_groups = df.groupby(df.index.date)

                        for date, group_df in date_groups:
                            self._store_partitioned_data(group_df, symbol, date)

                        # Update metadata
                        cache_key = f"{symbol}_{data_type}"
                        metadata = self.metadata_cache.get(cache_key, IngestionMetadata(symbol, data_type))

                        metadata.last_update = datetime.now()
                        metadata.last_successful_fetch = pd.to_datetime(actual_end)
                        metadata.total_records += len(df)
                        metadata.error_count = 0
                        metadata.last_error = None

                        if not metadata.date_range_start or pd.to_datetime(actual_start) < metadata.date_range_start:
                            metadata.date_range_start = pd.to_datetime(actual_start)
                        if not metadata.date_range_end or pd.to_datetime(actual_end) > metadata.date_range_end:
                            metadata.date_range_end = pd.to_datetime(actual_end)

                        self._save_metadata(metadata)

                        symbol_results['records_fetched'] += len(df)
                        results['total_records'] += len(df)

                    symbol_results['data_types_processed'] += 1
                    results['successful_fetches'] += 1

                    logger.info(f"Successfully fetched {len(df)} {data_type} records for {symbol}")

                except Exception as e:
                    error_msg = f"Failed to fetch {data_type} data for {symbol}: {str(e)}"
                    logger.error(error_msg)

                    # Update metadata with error
                    cache_key = f"{symbol}_{data_type}"
                    metadata = self.metadata_cache.get(cache_key, IngestionMetadata(symbol, data_type))
                    metadata.error_count += 1
                    metadata.last_error = str(e)
                    self._save_metadata(metadata)

                    symbol_results['errors'].append(error_msg)
                    results['errors'].append(error_msg)
                    results['failed_fetches'] += 1

            results['symbol_results'][symbol] = symbol_results

        logger.info(f"Data fetch completed. Success: {results['successful_fetches']}, Failed: {results['failed_fetches']}")
        return results

    async def fetch_incremental_update(
        self,
        symbols: Optional[Union[str, List[str]]] = None,
        days_back: int = 7,
        data_types: Union[str, List[str]] = 'ohlcv'
    ) -> Dict[str, Any]:
        """
        Perform incremental update for symbols that need updating.

        Args:
            symbols: Specific symbols to update (None for all tracked symbols)
            days_back: Number of days to look back for updates
            data_types: Data types to update

        Returns:
            Dictionary with update results
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        elif symbols is None:
            # Get all symbols from metadata
            symbols = list(set(meta.symbol for meta in self.metadata_cache.values()))

        if isinstance(data_types, str):
            data_types = [data_types]

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        logger.info(f"Starting incremental update for {len(symbols)} symbols from {start_date} to {end_date}")

        return await self.fetch_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            incremental=True
        )

    def get_ingestion_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get ingestion status and metadata.

        Args:
            symbol: Optional symbol to get status for

        Returns:
            Dictionary with status information
        """
        if symbol:
            # Get status for specific symbol
            symbol_metadata = {
                k: v for k, v in self.metadata_cache.items()
                if v.symbol == symbol
            }

            return {
                'symbol': symbol,
                'data_types': list(symbol_metadata.keys()),
                'metadata': {k: {
                    'last_update': v.last_update.isoformat() if v.last_update else None,
                    'date_range': {
                        'start': v.date_range_start.isoformat() if v.date_range_start else None,
                        'end': v.date_range_end.isoformat() if v.date_range_end else None
                    },
                    'total_records': v.total_records,
                    'last_successful_fetch': v.last_successful_fetch.isoformat() if v.last_successful_fetch else None,
                    'error_count': v.error_count,
                    'last_error': v.last_error
                } for k, v in symbol_metadata.items()}
            }
        else:
            # Get overall status
            return {
                'total_symbols': len(set(meta.symbol for meta in self.metadata_cache.values())),
                'total_data_types': len(self.metadata_cache),
                'symbols': list(set(meta.symbol for meta in self.metadata_cache.values())),
                'data_types': list(set(meta.data_type for meta in self.metadata_cache.values())),
                'total_records': sum(meta.total_records for meta in self.metadata_cache.values()),
                'error_count': sum(meta.error_count for meta in self.metadata_cache.values())
            }

    def clear_metadata(self, symbol: Optional[str] = None, data_type: Optional[str] = None):
        """
        Clear ingestion metadata.

        Args:
            symbol: Optional symbol to clear metadata for
            data_type: Optional data type to clear metadata for
        """
        keys_to_remove = []

        for cache_key, metadata in self.metadata_cache.items():
            if symbol and metadata.symbol != symbol:
                continue
            if data_type and metadata.data_type != data_type:
                continue

            keys_to_remove.append(cache_key)

            # Remove metadata file
            metadata_file = self.metadata_dir / f"{cache_key}.json"
            if metadata_file.exists():
                metadata_file.unlink()

        for key in keys_to_remove:
            del self.metadata_cache[key]

        logger.info(f"Cleared metadata for {len(keys_to_remove)} entries")

    async def aclose(self):
        """Close the ingestor and cleanup resources."""
        await self.polygon_client.aclose()
        logger.info("PolygonDataIngestor closed")
