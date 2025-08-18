"""
VIX (Volatility Index) data loader.

This module provides interfaces for retrieving VIX data and volatility
measures from Cboe Global Markets.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from io import StringIO

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VIXDataLoader:
    def __init__(self, settings: Settings):
        """
        Initialize VIX data loader.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_dir = Path("data/cache/vix")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # VIX data sources
        self.cboe_base_url = "https://cdn.cboe.com/api/global/delayed_quotes"
        self.fred_base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        
    def get_historical_vix(self, start_date: Union[str, datetime], 
                          end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get historical VIX data.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with VIX historical data
        """
        try:
            # Convert dates to string if datetime objects
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Check cache first
            cache_key = f"vix_{start_date}_{end_date}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                logger.info(f"Loading cached VIX data from {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Try to get data from Cboe first
            vix_data = self._get_vix_from_cboe(start_date, end_date)
            
            if vix_data.empty:
                # Fallback to FRED
                logger.warning("Cboe data not available, falling back to FRED")
                vix_data = self._get_vix_from_fred(start_date, end_date)
            
            if vix_data.empty:
                raise ValueError("Failed to retrieve VIX data from all sources")
            
            # Cache the data
            vix_data.to_parquet(cache_file)
            logger.info(f"Cached VIX data to {cache_file}")
            
            return vix_data
            
        except Exception as e:
            logger.error(f"Failed to get historical VIX data: {e}")
            raise
    
    def _get_vix_from_cboe(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get VIX data from Cboe Global Markets API.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with VIX data
        """
        try:
            # Cboe API endpoint for VIX historical data
            url = f"{self.cboe_base_url}/vix/historical"
            
            # Add date parameters
            params = {
                'start_date': start_date,
                'end_date': end_date
            }
            
            # Make request
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            if 'data' not in data or not data['data']:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Convert date column and set as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Standardize column names
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get VIX data from Cboe: {e}")
            return pd.DataFrame()
    
    def _get_vix_from_fred(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get VIX data from FRED (Federal Reserve Economic Data).
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with VIX data
        """
        try:
            # FRED VIX series ID
            series_id = "VIXCLS"
            
            # Build URL
            url = f"{self.fred_base_url}?id={series_id}&from={start_date}&to={end_date}"
            
            # Make request
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV response
            df = pd.read_csv(StringIO(response.text))
            
            # Convert date column and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Standardize column names
            df = df.rename(columns={
                series_id: 'close',
                'vix_close': 'close'  # Alternative column name
            })
            
            # Create OHLC columns (FRED only provides close)
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']
            df['volume'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get VIX data from FRED: {e}")
            return pd.DataFrame()
    
    def get_vix_futures(self, start_date: Union[str, datetime], 
                       end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get VIX futures data.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with VIX futures data
        """
        try:
            # Convert dates to string if datetime objects
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Check cache first
            cache_key = f"vix_futures_{start_date}_{end_date}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                logger.info(f"Loading cached VIX futures data from {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Get data from Cboe
            vix_futures_data = self._get_vix_futures_from_cboe(start_date, end_date)
            
            if vix_futures_data.empty:
                logger.warning("VIX futures data not available")
                return pd.DataFrame()
            
            # Cache the data
            vix_futures_data.to_parquet(cache_file)
            logger.info(f"Cached VIX futures data to {cache_file}")
            
            return vix_futures_data
            
        except Exception as e:
            logger.error(f"Failed to get VIX futures data: {e}")
            return pd.DataFrame()
    
    def _get_vix_futures_from_cboe(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get VIX futures data from Cboe.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with VIX futures data
        """
        try:
            # This would require access to Cboe futures data API
            # For now, return empty DataFrame as placeholder
            logger.info("VIX futures data retrieval not implemented yet")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get VIX futures data from Cboe: {e}")
            return pd.DataFrame()
    
    def calculate_vix_term_structure(self, date: datetime) -> pd.DataFrame:
        """
        Calculate VIX term structure for a given date.
        
        Args:
            date: Date for term structure calculation
            
        Returns:
            DataFrame with VIX term structure
        """
        try:
            # Get VIX futures data for the date
            futures_data = self.get_vix_futures(
                date - timedelta(days=30),
                date + timedelta(days=30)
            )
            
            if futures_data.empty:
                return pd.DataFrame()
            
            # Filter data for the specific date
            date_data = futures_data[futures_data.index.date == date.date()]
            
            if date_data.empty:
                return pd.DataFrame()
            
            # Calculate term structure metrics
            term_structure = pd.DataFrame(index=[date])
            
            # Near-term and next-term VIX
            if len(date_data) >= 2:
                term_structure['vix_near'] = date_data['close'].iloc[0]
                term_structure['vix_next'] = date_data['close'].iloc[1]
                term_structure['vix_ratio'] = term_structure['vix_next'] / term_structure['vix_near']
                term_structure['vix_slope'] = term_structure['vix_next'] - term_structure['vix_near']
            else:
                term_structure['vix_near'] = date_data['close'].iloc[0]
                term_structure['vix_next'] = np.nan
                term_structure['vix_ratio'] = np.nan
                term_structure['vix_slope'] = np.nan
            
            return term_structure
            
        except Exception as e:
            logger.error(f"Failed to calculate VIX term structure: {e}")
            return pd.DataFrame()
    
    def get_vix_regime(self, vix_value: float) -> str:
        """
        Classify VIX level into volatility regime.
        
        Args:
            vix_value: VIX value
            
        Returns:
            Volatility regime classification
        """
        if vix_value < 13:
            return "low"
        elif vix_value < 20:
            return "normal"
        elif vix_value < 30:
            return "high"
        else:
            return "extreme"
    
    def calculate_vix_features(self, vix_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional VIX-based features.
        
        Args:
            vix_data: DataFrame with VIX data
            
        Returns:
            DataFrame with additional VIX features
        """
        if vix_data.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=vix_data.index)
        
        # Basic VIX levels
        features['vix_close'] = vix_data['close']
        features['vix_open'] = vix_data['open']
        features['vix_high'] = vix_data['high']
        features['vix_low'] = vix_data['low']
        
        # VIX returns
        features['vix_return'] = vix_data['close'].pct_change()
        features['vix_return_5d'] = vix_data['close'].pct_change(5)
        features['vix_return_20d'] = vix_data['close'].pct_change(20)
        
        # VIX volatility
        features['vix_volatility'] = vix_data['close'].rolling(window=20).std()
        features['vix_volatility_ratio'] = features['vix_volatility'] / features['vix_close']
        
        # VIX regime
        features['vix_regime'] = features['vix_close'].apply(self.get_vix_regime)
        
        # VIX momentum
        features['vix_momentum_5d'] = vix_data['close'].rolling(window=5).mean() / vix_data['close'] - 1
        features['vix_momentum_20d'] = vix_data['close'].rolling(window=20).mean() / vix_data['close'] - 1
        
        return features
    
    def get_vix_data_for_date_range(self, start_date: Union[str, datetime], 
                                   end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get VIX data for a date range with all features.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with VIX data and features
        """
        # Get raw VIX data
        vix_data = self.get_historical_vix(start_date, end_date)
        
        if vix_data.empty:
            return pd.DataFrame()
        
        # Calculate features
        vix_features = self.calculate_vix_features(vix_data)
        
        return vix_features
    
    def clear_cache(self) -> None:
        """Clear VIX data cache."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()
        logger.info("Cleared VIX data cache")
    
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


# Alias for compatibility
VIXLoader = VIXDataLoader