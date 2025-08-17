"""
Databento client for historical market data.

This module provides interfaces for accessing historical market data
from Databento with efficient data retrieval and caching.
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
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    logging.warning("databento not available. Databento client will be disabled.")

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DatabentoClient:
    """
    Databento client for historical market data retrieval.
    
    This class handles connection to Databento API and provides methods
    for retrieving historical market data with efficient caching.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Databento client.
        
        Args:
            settings: Configuration settings
        """
        if not DATABENTO_AVAILABLE:
            raise ImportError("databento is required for Databento client")
        
        self.settings = settings
        self.client = db.Live()
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_dir = Path("data/cache/databento")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API key from environment
        self.api_key = settings.get('data', 'databento_api_key')
        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY environment variable not set")
        
        # Configure client
        self.client.set_key(self.api_key)
        
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
        Retrieve historical market data from Databento.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            dataset: Dataset name (default: 'continuous')
            stype: Symbol type (default: 'raw_symbol')
            schema: Data schema (default: 'ohlcv-1s')
            freq: Frequency for resampling (default: '1min')
            
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
            cache_key = f"{symbol}_{start_date}_{end_date}_{schema}_{freq}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                logger.info(f"Loading cached data for {symbol} from {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Request data from Databento
            logger.info(f"Requesting historical data for {symbol} from {start_date} to {end_date}")
            
            data = self.client.timeseries.get_dataset(
                dataset=dataset,
                symbols=[symbol],
                stype=stype,
                schema=schema,
                start=start_date,
                end=end_date,
                ts_event=True,
                limit=1000000  # Large limit to get all data
            )
            
            # Convert to DataFrame
            df = self._timeseries_to_dataframe(data)
            
            # Resample if needed
            if freq != '1s' and schema == 'ohlcv-1s':
                df = self._resample_data(df, freq)
            
            # Cache the data
            df.to_parquet(cache_file)
            logger.info(f"Cached data to {cache_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise
    
    def _timeseries_to_dataframe(self, data: db.Timeseries) -> pd.DataFrame:
        """
        Convert Databento timeseries data to DataFrame.
        
        Args:
            data: Databento timeseries data
            
        Returns:
            DataFrame with market data
        """
        # Convert to pandas DataFrame
        df = data.to_df()
        
        # Set timestamp as index
        if 'ts_event' in df.columns:
            df.set_index('ts_event', inplace=True)
            df.index = pd.to_datetime(df.index)
        
        # Standardize column names
        column_mapping = {
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'bid_px_00': 'bid',
            'ask_px_00': 'ask',
            'bid_sz_00': 'bid_size',
            'ask_sz_00': 'ask_size'
        }
        
        # Rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)
        
        return df
    
    def _resample_data(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample data to specified frequency.
        
        Args:
            df: Input DataFrame
            freq: Resampling frequency
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        # Resample OHLCV data
        agg_dict = {}
        
        if 'open' in df.columns:
            agg_dict['open'] = 'first'
        if 'high' in df.columns:
            agg_dict['high'] = 'max'
        if 'low' in df.columns:
            agg_dict['low'] = 'min'
        if 'close' in df.columns:
            agg_dict['close'] = 'last'
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'
        if 'bid' in df.columns:
            agg_dict['bid'] = 'last'
        if 'ask' in df.columns:
            agg_dict['ask'] = 'last'
        
        if agg_dict:
            df = df.resample(freq).agg(agg_dict)
        
        return df
    
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
        return self.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            schema='topbook-1s',
            freq='1s'
        )
    
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
        # Get tick data and calculate VWAP
        tick_data = self.get_tick_data(symbol, start_date, end_date)
        
        if tick_data.empty:
            return pd.DataFrame()
        
        # Calculate VWAP for each minute
        tick_data['mid_price'] = (tick_data['bid'] + tick_data['ask']) / 2
        tick_data['price_volume'] = tick_data['mid_price'] * tick_data['bid_size']
        
        vwap_data = tick_data.resample('1min').agg({
            'price_volume': 'sum',
            'bid_size': 'sum'
        })
        
        vwap_data['vwap'] = vwap_data['price_volume'] / vwap_data['bid_size']
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
        book_data = self.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            schema='topbook-1s',
            freq='1s'
        )
        
        if book_data.empty:
            return pd.DataFrame()
        
        # Calculate liquidity metrics
        liquidity_metrics = pd.DataFrame(index=book_data.index)
        
        if 'bid' in book_data.columns and 'ask' in book_data.columns:
            # Bid-ask spread
            liquidity_metrics['spread'] = book_data['ask'] - book_data['bid']
            liquidity_metrics['spread_pct'] = liquidity_metrics['spread'] / book_data['mid']
        
        if 'bid_size' in book_data.columns and 'ask_size' in book_data.columns:
            # Order book depth
            liquidity_metrics['bid_depth'] = book_data['bid_size']
            liquidity_metrics['ask_depth'] = book_data['ask_size']
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
            pattern = f"{symbol}_*.parquet"
            cache_files = list(self.cache_dir.glob(pattern))
            for file in cache_files:
                file.unlink()
                logger.info(f"Cleared cache file: {file}")
        else:
            # Clear all cache
            for file in self.cache_dir.glob("*.parquet"):
                file.unlink()
            logger.info("Cleared all cache files")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.parquet"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_directory': str(self.cache_dir),
            'total_files': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'files': [f.name for f in cache_files]
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


class DatabentoDataPipeline:
    """
    Data pipeline for managing multiple instruments and data types.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize data pipeline.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.client = DatabentoClient(settings)
        self.data_buffer: Dict[str, pd.DataFrame] = {}
        self.buffer_size = 1000  # Number of bars to buffer
        
    def load_data_for_training(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_types: List[str] = ['ohlcv', 'liquidity']
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for training multiple instruments.
        
        Args:
            symbols: List of instrument symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_types: List of data types to load
            
        Returns:
            Dictionary with data for each symbol
        """
        results = {}
        
        for symbol in symbols:
            symbol_data = {}
            
            # Load OHLCV data
            if 'ohlcv' in data_types:
                try:
                    ohlcv_data = self.client.get_minute_bars(symbol, start_date, end_date)
                    symbol_data['ohlcv'] = ohlcv_data
                except Exception as e:
                    logger.error(f"Failed to load OHLCV data for {symbol}: {e}")
            
            # Load liquidity data
            if 'liquidity' in data_types:
                try:
                    liquidity_data = self.client.get_liquidity_metrics(symbol, start_date, end_date)
                    symbol_data['liquidity'] = liquidity_data
                except Exception as e:
                    logger.error(f"Failed to load liquidity data for {symbol}: {e}")
            
            results[symbol] = symbol_data
        
        return results
    
    def get_latest_data(self, symbol: str, lookback_minutes: int = 120) -> pd.DataFrame:
        """
        Get latest data for an instrument.
        
        Args:
            symbol: Instrument symbol
            lookback_minutes: Number of minutes to look back
            
        Returns:
            DataFrame with latest data
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Get recent data
        data = self.client.get_minute_bars(symbol, start_date, end_date)
        
        if data.empty:
            return pd.DataFrame()
        
        # Filter to lookback period
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_data = data[data.index >= cutoff_time]
        
        return recent_data