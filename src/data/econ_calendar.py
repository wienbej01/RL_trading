"""
Economic calendar data loader.

This module provides interfaces for retrieving economic events, 
earnings announcements, and market holidays.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from io import StringIO
import json

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EconomicCalendar:
    """
    Economic calendar for tracking market events and holidays.
    
    This class handles retrieval of economic events, earnings announcements,
    and market holidays that may impact trading.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize economic calendar.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_dir = Path("data/cache/econ_calendar")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Calendar data sources
        self.fred_base_url = "https://fred.stlouisfed.org"
        self.market_hours_url = "https://api.tradinghours.com"
        
    def get_market_holidays(self, year: int, country: str = "US") -> pd.DataFrame:
        """
        Get market holidays for a specific year and country.
        
        Args:
            year: Year for which to get holidays
            country: Country code (default: US)
            
        Returns:
            DataFrame with market holidays
        """
        try:
            # Check cache first
            cache_key = f"holidays_{country}_{year}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                logger.info(f"Loading cached holidays from {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Get holidays from FRED
            holidays_data = self._get_holidays_from_fred(year, country)
            
            if holidays_data.empty:
                logger.warning(f"No holidays found for {country} {year}")
                return pd.DataFrame()
            
            # Cache the data
            holidays_data.to_parquet(cache_file)
            logger.info(f"Cached holidays to {cache_file}")
            
            return holidays_data
            
        except Exception as e:
            logger.error(f"Failed to get market holidays: {e}")
            return pd.DataFrame()
    
    def _get_holidays_from_fred(self, year: int, country: str) -> pd.DataFrame:
        """
        Get market holidays from FRED.
        
        Args:
            year: Year for which to get holidays
            country: Country code
            
        Returns:
            DataFrame with holidays
        """
        try:
            # FRED holiday series IDs
            holiday_series = {
                "US": "FRED:DCHR1US",  # US Federal Reserve holidays
                "UK": "FRED:DCHR1GB",  # UK holidays
                "EU": "FRED:DCHR1EZ",  # Eurozone holidays
            }
            
            if country not in holiday_series:
                logger.warning(f"Holidays not available for country: {country}")
                return pd.DataFrame()
            
            series_id = holiday_series[country]
            
            # Build URL for the year
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            url = f"{self.fred_base_url}/graph/fredgraph.csv?id={series_id}&from={start_date}&to={end_date}"
            
            # Make request
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV response
            df = pd.read_csv(StringIO(response.text))
            
            # Convert date column and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Filter to the specific year
            df = df[df.index.year == year]
            
            # Standardize column names
            df = df.rename(columns={
                series_id: 'is_holiday'
            })
            
            # Create holiday name column (placeholder)
            df['holiday_name'] = 'Market Holiday'
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get holidays from FRED: {e}")
            return pd.DataFrame()
    
    def get_earnings_announcements(self, symbol: str, 
                                  start_date: Union[str, datetime], 
                                  end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get earnings announcements for a specific symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for search
            end_date: End date for search
            
        Returns:
            DataFrame with earnings announcements
        """
        try:
            # Convert dates to string if datetime objects
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Check cache first
            cache_key = f"earnings_{symbol}_{start_date}_{end_date}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                logger.info(f"Loading cached earnings data from {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Get earnings data (placeholder implementation)
            earnings_data = self._get_earnings_data(symbol, start_date, end_date)
            
            if earnings_data.empty:
                logger.warning(f"No earnings data found for {symbol}")
                return pd.DataFrame()
            
            # Cache the data
            earnings_data.to_parquet(cache_file)
            logger.info(f"Cached earnings data to {cache_file}")
            
            return earnings_data
            
        except Exception as e:
            logger.error(f"Failed to get earnings announcements: {e}")
            return pd.DataFrame()
    
    def _get_earnings_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get earnings data (placeholder implementation).
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with earnings data
        """
        try:
            # This would typically use an earnings API like:
            # - Alpha Vantage
            # - Financial Modeling Prep
            # - Earnings Calendar API
            
            # For now, return placeholder data
            logger.info(f"Earnings data retrieval not implemented for {symbol}")
            
            # Create sample data for demonstration
            dates = pd.date_range(start=start_date, end=end_date, freq='Q')
            earnings_data = pd.DataFrame({
                'symbol': symbol,
                'earnings_date': dates,
                'eps_estimate': np.random.uniform(1.0, 5.0, len(dates)),
                'eps_actual': np.random.uniform(0.5, 6.0, len(dates)),
                'revenue_estimate': np.random.uniform(1000, 10000, len(dates)),
                'revenue_actual': np.random.uniform(800, 12000, len(dates)),
                'surprise_pct': np.random.uniform(-20, 20, len(dates))
            })
            
            earnings_data.set_index('earnings_date', inplace=True)
            
            return earnings_data
            
        except Exception as e:
            logger.error(f"Failed to get earnings data: {e}")
            return pd.DataFrame()
    
    def get_fed_meetings(self, year: int) -> pd.DataFrame:
        """
        Get Federal Reserve meeting dates.
        
        Args:
            year: Year for which to get meetings
            
        Returns:
            DataFrame with Fed meeting dates
        """
        try:
            # Check cache first
            cache_key = f"fed_meetings_{year}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                logger.info(f"Loading cached Fed meetings from {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Get Fed meeting dates
            meetings_data = self._get_fed_meetings(year)
            
            if meetings_data.empty:
                logger.warning(f"No Fed meetings found for {year}")
                return pd.DataFrame()
            
            # Cache the data
            meetings_data.to_parquet(cache_file)
            logger.info(f"Cached Fed meetings to {cache_file}")
            
            return meetings_data
            
        except Exception as e:
            logger.error(f"Failed to get Fed meetings: {e}")
            return pd.DataFrame()
    
    def _get_fed_meetings(self, year: int) -> pd.DataFrame:
        """
        Get Federal Reserve meeting dates.
        
        Args:
            year: Year for which to get meetings
            
        Returns:
            DataFrame with Fed meeting dates
        """
        try:
            # Federal Reserve meeting dates (known schedule)
            # These are typically scheduled in advance
            meeting_dates = [
                f"{year}-01-31",  # January meeting
                f"{year}-03-21",  # March meeting
                f"{year}-05-02",  # May meeting
                f"{year}-06-14",  # June meeting
                f"{year}-07-26",  # July meeting
                f"{year}-09-20",  # September meeting
                f"{year}-11-01",  # November meeting
                f"{year}-12-13",  # December meeting
            ]
            
            # Create DataFrame
            meetings = pd.DataFrame({
                'meeting_date': pd.to_datetime(meeting_dates),
                'meeting_type': 'FOMC',
                'has_press_conference': [True, True, True, True, True, True, True, True],
                'rate_decision': np.nan  # Will be filled after meetings
            })
            
            meetings.set_index('meeting_date', inplace=True)
            
            return meetings
            
        except Exception as e:
            logger.error(f"Failed to get Fed meetings: {e}")
            return pd.DataFrame()
    
    def get_cpi_releases(self, year: int) -> pd.DataFrame:
        """
        Get CPI (Consumer Price Index) release dates.
        
        Args:
            year: Year for which to get CPI releases
            
        Returns:
            DataFrame with CPI release dates
        """
        try:
            # Check cache first
            cache_key = f"cpi_releases_{year}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                logger.info(f"Loading cached CPI releases from {cache_file}")
                return pd.read_parquet(cache_file)
            
            # Get CPI release dates
            cpi_data = self._get_cpi_releases(year)
            
            if cpi_data.empty:
                logger.warning(f"No CPI releases found for {year}")
                return pd.DataFrame()
            
            # Cache the data
            cpi_data.to_parquet(cache_file)
            logger.info(f"Cached CPI releases to {cache_file}")
            
            return cpi_data
            
        except Exception as e:
            logger.error(f"Failed to get CPI releases: {e}")
            return pd.DataFrame()
    
    def _get_cpi_releases(self, year: int) -> pd.DataFrame:
        """
        Get CPI release dates.
        
        Args:
            year: Year for which to get CPI releases
            
        Returns:
            DataFrame with CPI release dates
        """
        try:
            # CPI is typically released around the 10th-15th of each month
            # at 8:30 AM ET
            months = range(1, 13)
            release_dates = []
            
            for month in months:
                # Approximate release date (middle of month)
                release_date = pd.Timestamp(f"{year}-{month:02d}-10")
                release_dates.append(release_date)
            
            # Create DataFrame
            cpi_releases = pd.DataFrame({
                'release_date': release_dates,
                'indicator': 'CPI',
                'time': '08:30 ET',
                'frequency': 'Monthly'
            })
            
            cpi_releases.set_index('release_date', inplace=True)
            
            return cpi_releases
            
        except Exception as e:
            logger.error(f"Failed to get CPI releases: {e}")
            return pd.DataFrame()
    
    def get_trading_hours(self, symbol: str, date: datetime) -> Dict[str, Any]:
        """
        Get trading hours for a specific symbol and date.
        
        Args:
            symbol: Instrument symbol
            date: Date for which to get trading hours
            
        Returns:
            Dictionary with trading hours
        """
        try:
            # Get instrument configuration
            instruments = self.settings.get('instruments', {})
            
            if symbol not in instruments:
                logger.warning(f"Instrument not configured: {symbol}")
                return {}
            
            config = instruments[symbol]
            
            # Regular trading hours
            rth_start = config.get('rth_start', '09:30')
            rth_end = config.get('rth_end', '16:00')
            rth_tz = config.get('rth_tz', 'America/New_York')
            
            # Check if date is a holiday
            holidays = self.get_market_holidays(date.year, 'US')
            is_holiday = date.date() in holidays.index.date
            
            # Check if date is weekend
            is_weekend = date.weekday() >= 5
            
            trading_hours = {
                'symbol': symbol,
                'date': date.date(),
                'is_holiday': is_holiday,
                'is_weekend': is_weekend,
                'regular_trading_hours': {
                    'start': f"{date.date()} {rth_start}",
                    'end': f"{date.date()} {rth_end}",
                    'timezone': rth_tz,
                    'is_active': not (is_holiday or is_weekend)
                },
                'extended_trading_hours': {
                    'pre_market': {
                        'start': f"{date.date()} 04:00",
                        'end': f"{date.date()} {rth_start}",
                        'is_active': not (is_holiday or is_weekend)
                    },
                    'post_market': {
                        'start': f"{date.date()} {rth_end}",
                        'end': f"{date.date()} 20:00",
                        'is_active': not (is_holiday or is_weekend)
                    }
                }
            }
            
            return trading_hours
            
        except Exception as e:
            logger.error(f"Failed to get trading hours: {e}")
            return {}
    
    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a date is a trading day.
        
        Args:
            date: Date to check
            
        Returns:
            True if it's a trading day, False otherwise
        """
        try:
            # Check if it's a weekend
            if date.weekday() >= 5:
                return False
            
            # Check if it's a holiday
            holidays = self.get_market_holidays(date.year, 'US')
            if date.date() in holidays.index.date:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check trading day: {e}")
            return False
    
    def get_next_trading_day(self, date: datetime) -> datetime:
        """
        Get the next trading day after a given date.
        
        Args:
            date: Starting date
            
        Returns:
            Next trading day
        """
        next_day = date + timedelta(days=1)
        
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        
        return next_day
    
    def get_previous_trading_day(self, date: datetime) -> datetime:
        """
        Get the previous trading day before a given date.
        
        Args:
            date: Starting date
            
        Returns:
            Previous trading day
        """
        prev_day = date - timedelta(days=1)
        
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        
        return prev_day
    
    def get_event_impact(self, event_type: str, symbol: str = None) -> Dict[str, Any]:
        """
        Get expected impact of an economic event.
        
        Args:
            event_type: Type of event (e.g., 'FOMC', 'CPI', 'EARNINGS')
            symbol: Optional symbol for earnings events
            
        Returns:
            Dictionary with event impact information
        """
        event_impacts = {
            'FOMC': {
                'impact': 'high',
                'volatility': 'high',
                'liquidity': 'medium',
                'description': 'Federal Open Market Committee meetings can significantly impact markets'
            },
            'CPI': {
                'impact': 'high',
                'volatility': 'high',
                'liquidity': 'medium',
                'description': 'Consumer Price Index releases affect inflation expectations'
            },
            'NFP': {
                'impact': 'high',
                'volatility': 'high',
                'liquidity': 'high',
                'description': 'Non-Farm Payrolls report impacts currency and equity markets'
            },
            'EARNINGS': {
                'impact': 'medium',
                'volatility': 'high',
                'liquidity': 'medium',
                'description': 'Earnings announcements can impact individual stocks'
            },
            'GDP': {
                'impact': 'medium',
                'volatility': 'medium',
                'liquidity': 'medium',
                'description': 'Gross Domestic Product releases indicate economic growth'
            }
        }
        
        return event_impacts.get(event_type, {
            'impact': 'low',
            'volatility': 'low',
            'liquidity': 'low',
            'description': 'Unknown event type'
        })
    
    def clear_cache(self) -> None:
        """Clear economic calendar cache."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()
        logger.info("Cleared economic calendar cache")
    
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
EconCalendar = EconomicCalendar