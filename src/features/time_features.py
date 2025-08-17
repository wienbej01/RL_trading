"""
Time-based features for the RL trading system.

This module implements time-based features including intraday seasonality,
time-of-day encoding, and session-based features.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, time

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimeConfig:
    """Configuration for time-based features."""
    session_start: str
    session_end: str
    timezone: str
    intraday_windows: List[int]
    day_of_week_encoding: bool
    month_encoding: bool
    holiday_encoding: bool


class TimeFeatures:
    """
    Time-based features implementation.
    
    This class provides comprehensive time-based features including
    intraday seasonality, time-of-day encoding, and session-based features.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize time features.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.config = settings.get('time_features', {})
        self.session_start = self.config.get('session_start', '09:30')
        self.session_end = self.config.get('session_end', '16:00')
        self.timezone = self.config.get('timezone', 'America/New_York')
        self.intraday_windows = self.config.get('intraday_windows', [15, 30, 60, 120])
        self.day_of_week_encoding = self.config.get('day_of_week_encoding', True)
        self.month_encoding = self.config.get('month_encoding', True)
        self.holiday_encoding = self.config.get('holiday_encoding', True)
        
    def calculate_all_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all time-based features.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            DataFrame with all time-based features
        """
        if data.empty:
            return pd.DataFrame()
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        
        # Calculate intraday features
        result = self._add_intraday_features(result, data)
        
        # Calculate session features
        result = self._add_session_features(result, data)
        
        # Calculate calendar features
        result = self._add_calendar_features(result, data)
        
        # Calculate time-based volatility features
        result = self._add_time_volatility_features(result, data)
        
        # Calculate time-based momentum features
        result = self._add_time_momentum_features(result, data)
        
        return result
    
    def _add_intraday_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add intraday time-based features."""
        # Time of day features
        result['hour'] = data.index.hour
        result['minute'] = data.index.minute
        result['second'] = data.index.second
        
        # Time of day in minutes since session start
        session_start_time = time.fromisoformat(self.session_start)
        result['minutes_since_open'] = (
            (data.index - data.index.normalize() + pd.Timedelta(hours=session_start_time.hour, minutes=session_start_time.minute))
            .dt.total_seconds() / 60
        )
        
        # Time of day in minutes until session end
        session_end_time = time.fromisoformat(self.session_end)
        result['minutes_until_close'] = (
            (data.index.normalize() + pd.Timedelta(hours=session_end_time.hour, minutes=session_end_time.minute) - data.index)
            .dt.total_seconds() / 60
        )
        
        # Time of day sine/cosine encoding
        result['time_sin'] = np.sin(2 * np.pi * result['minutes_since_open'] / (24 * 60))
        result['time_cos'] = np.cos(2 * np.pi * result['minutes_since_open'] / (24 * 60))
        
        # Intraday windows
        for window in self.intraday_windows:
            result[f'intraday_return_{window}'] = data['close'].pct_change(window)
            result[f'intraday_volatility_{window}'] = data['close'].pct_change().rolling(window=window).std()
        
        # Intraday momentum
        result['intraday_momentum'] = (
            data['close'] - data['close'].rolling(window=30).mean()
        ) / data['close'].rolling(window=30).mean()
        
        # Intraday acceleration
        result['intraday_acceleration'] = (
            result['intraday_momentum'] - result['intraday_momentum'].shift(1)
        )
        
        # Intraday regime
        result['intraday_regime'] = pd.cut(
            result['minutes_since_open'],
            bins=[0, 60, 120, 180, 240, 300, 360, 420, 480],
            labels=['opening', 'early_morning', 'mid_morning', 'late_morning', 
                   'early_afternoon', 'mid_afternoon', 'late_afternoon', 'closing']
        )
        
        return result
    
    def _add_session_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add session-based features."""
        # Session time flags
        session_start_time = time.fromisoformat(self.session_start)
        session_end_time = time.fromisoformat(self.session_end)
        
        result['is_session_time'] = (
            (data.index.time >= session_start_time) & 
            (data.index.time <= session_end_time)
        )
        
        result['is_pre_market'] = data.index.time < session_start_time
        result['is_post_market'] = data.index.time > session_end_time
        
        # Session phase
        result['session_phase'] = pd.cut(
            result['minutes_since_open'],
            bins=[0, 60, 180, 300, 480],
            labels=['opening', 'morning', 'afternoon', 'closing']
        )
        
        # Session volatility
        result['session_volatility'] = (
            data['close'].pct_change().rolling(window=60).std()
        )
        
        # Session liquidity
        result['session_liquidity'] = (
            data['volume'].rolling(window=60).mean()
        )
        
        # Session momentum
        result['session_momentum'] = (
            data['close'] - data['close'].rolling(window=120).mean()
        ) / data['close'].rolling(window=120).mean()
        
        # Session pressure
        result['session_pressure'] = (
            (data['close'] - data['open']) / 
            (data['high'] - data['low'] + 1e-6)
        )
        
        return result
    
    def _add_calendar_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features."""
        # Day of week
        if self.day_of_week_encoding:
            result['day_of_week'] = data.index.dayofweek
            result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
            result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
            
            # Weekend flags
            result['is_weekend'] = data.index.dayofweek >= 5
            
            # Day type
            result['day_type'] = pd.cut(
                result['day_of_week'],
                bins=[-1, 0, 4, 6],
                labels=['monday', 'weekday', 'weekend']
            )
        
        # Month features
        if self.month_encoding:
            result['month'] = data.index.month
            result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
            result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
            
            # Quarter features
            result['quarter'] = data.index.quarter
            result['quarter_sin'] = np.sin(2 * np.pi * result['quarter'] / 4)
            result['quarter_cos'] = np.cos(2 * np.pi * result['quarter'] / 4)
            
            # Month end flags
            result['is_month_end'] = data.index.is_month_end
            result['is_month_start'] = data.index.is_month_start
            
            # Quarter end flags
            result['is_quarter_end'] = data.index.is_quarter_end
            result['is_quarter_start'] = data.index.is_quarter_start
        
        # Year features
        result['year'] = data.index.year
        result['year_progress'] = (data.index.dayofyear - 1) / 365
        
        # Holiday flags (placeholder - would need holiday calendar)
        if self.holiday_encoding:
            result['is_holiday'] = False  # Placeholder - would need holiday calendar
            result['holiday_distance'] = 0  # Placeholder
        
        # Earnings season flags (placeholder)
        result['is_earnings_season'] = False  # Placeholder
        
        # Economic event flags (placeholder)
        result['is_event_day'] = False  # Placeholder
        
        return result
    
    def _add_time_volatility_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based volatility features."""
        # Time-weighted volatility
        result['time_weighted_volatility'] = (
            data['close'].pct_change().rolling(window=20).std() * 
            np.sqrt(result['minutes_since_open'] / 480)
        )
        
        # Session volatility clustering
        result['session_volatility_clustering'] = (
            data['close'].pct_change().rolling(window=60).std() /
            data['close'].pct_change().rolling(window=480).std()
        )
        
        # Time-based volatility regime
        result['time_volatility_regime'] = pd.cut(
            result['time_weighted_volatility'],
            bins=[-np.inf, 0.1, 0.2, np.inf],
            labels=['low', 'medium', 'high']
        )
        
        # Volatility time patterns
        result['morning_volatility'] = data['close'].pct_change().rolling(window=60).std()
        result['afternoon_volatility'] = data['close'].pct_change().rolling(window=60).std()
        
        # Volatility time asymmetry
        result['volatility_asymmetry'] = (
            result['morning_volatility'] - result['afternoon_volatility']
        )
        
        return result
    
    def _add_time_momentum_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based momentum features."""
        # Time-weighted momentum
        result['time_weighted_momentum'] = (
            (data['close'] - data['close'].rolling(window=20).mean()) /
            data['close'].rolling(window=20).mean() *
            np.sqrt(result['minutes_since_open'] / 480)
        )
        
        # Session momentum patterns
        result['opening_momentum'] = (
            data['close'] - data['open']
        ) / data['open']
        
        result['closing_momentum'] = (
            data['close'] - data['close'].shift(30)
        ) / data['close'].shift(30)
        
        # Time-based momentum regime
        result['time_momentum_regime'] = pd.cut(
            result['time_weighted_momentum'],
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=['negative', 'neutral', 'positive']
        )
        
        # Momentum time patterns
        result['morning_momentum'] = (
            data['close'] - data['close'].rolling(window=60).mean()
        ) / data['close'].rolling(window=60).mean()
        
        result['afternoon_momentum'] = (
            data['close'] - data['close'].rolling(window=60).mean()
        ) / data['close'].rolling(window=60).mean()
        
        # Momentum time asymmetry
        result['momentum_asymmetry'] = (
            result['morning_momentum'] - result['afternoon_momentum']
        )
        
        return result
    
    def calculate_intraday_seasonality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday seasonality patterns.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with seasonality features
        """
        if data.empty:
            return pd.DataFrame()
        
        seasonality = pd.DataFrame(index=data.index)
        
        # Group by time of day
        time_groups = data.groupby(data.index.time)
        
        # Calculate average returns by time of day
        avg_returns = time_groups['close'].apply(
            lambda x: x.pct_change().mean()
        )
        
        # Calculate average volatility by time of day
        avg_volatility = time_groups['close'].apply(
            lambda x: x.pct_change().std()
        )
        
        # Calculate average volume by time of day
        avg_volume = time_groups['volume'].apply(
            lambda x: x.mean()
        )
        
        # Map back to original index
        seasonality['time_of_day_return'] = data.index.time.map(avg_returns)
        seasonality['time_of_day_volatility'] = data.index.time.map(avg_volatility)
        seasonality['time_of_day_volume'] = data.index.time.map(avg_volume)
        
        # Time of day deviation
        seasonality['return_deviation'] = (
            data['close'].pct_change() - seasonality['time_of_day_return']
        )
        
        seasonality['volatility_deviation'] = (
            data['close'].pct_change().rolling(window=20).std() - 
            seasonality['time_of_day_volatility']
        )
        
        return seasonality
    
    def calculate_session_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate session-based patterns.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with session patterns
        """
        if data.empty:
            return pd.DataFrame()
        
        patterns = pd.DataFrame(index=data.index)
        
        # Session grouping
        session_start = time.fromisoformat(self.session_start)
        session_end = time.fromisoformat(self.session_end)
        
        # Define session phases
        def get_session_phase(dt):
            t = dt.time()
            if t < session_start:
                return 'pre_market'
            elif t < time(10, 30):
                return 'opening'
            elif t < time(12, 0):
                return 'morning'
            elif t < time(14, 0):
                return 'midday'
            elif t < time(16, 0):
                return 'afternoon'
            else:
                return 'post_market'
        
        patterns['session_phase'] = data.index.map(get_session_phase)
        
        # Session performance
        session_performance = data.groupby(patterns['session_phase'])['close'].apply(
            lambda x: x.pct_change().mean()
        )
        
        patterns['session_performance'] = patterns['session_phase'].map(session_performance)
        
        # Session volatility
        session_volatility = data.groupby(patterns['session_phase'])['close'].apply(
            lambda x: x.pct_change().std()
        )
        
        patterns['session_volatility'] = patterns['session_phase'].map(session_volatility)
        
        # Session volume
        session_volume = data.groupby(patterns['session_phase'])['volume'].apply(
            lambda x: x.mean()
        )
        
        patterns['session_volume'] = patterns['session_phase'].map(session_volume)
        
        return patterns
    
    def calculate_calendar_effects(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate calendar-based effects.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with calendar effects
        """
        if data.empty:
            return pd.DataFrame()
        
        calendar = pd.DataFrame(index=data.index)
        
        # Day of week effects
        dow_performance = data.groupby(data.index.dayofweek)['close'].apply(
            lambda x: x.pct_change().mean()
        )
        
        calendar['dow_performance'] = data.index.dayofweek.map(dow_performance)
        
        # Month effects
        month_performance = data.groupby(data.index.month)['close'].apply(
            lambda x: x.pct_change().mean()
        )
        
        calendar['month_performance'] = data.index.month.map(month_performance)
        
        # Quarter effects
        quarter_performance = data.groupby(data.index.quarter)['close'].apply(
            lambda x: x.pct_change().mean()
        )
        
        calendar['quarter_performance'] = data.index.quarter.map(quarter_performance)
        
        # Year effects
        year_performance = data.groupby(data.index.year)['close'].apply(
            lambda x: x.pct_change().mean()
        )
        
        calendar['year_performance'] = data.index.year.map(year_performance)
        
        # Month-end effects
        calendar['month_end_effect'] = np.where(
            data.index.is_month_end,
            data['close'].pct_change(),
            np.nan
        )
        
        # Quarter-end effects
        calendar['quarter_end_effect'] = np.where(
            data.index.is_quarter_end,
            data['close'].pct_change(),
            np.nan
        )
        
        return calendar
    
    def calculate_time_based_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate time-based market regime.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with market regime
        """
        if data.empty:
            return pd.Series()
        
        # Combine multiple time-based signals
        volatility_signal = data['close'].pct_change().rolling(window=60).std()
        momentum_signal = (data['close'] - data['close'].rolling(window=120).mean()) / data['close'].rolling(window=120).mean()
        volume_signal = data['volume'] / data['volume'].rolling(window=120).mean()
        
        # Normalize signals
        volatility_norm = (volatility_signal - volatility_signal.rolling(window=480).mean()) / volatility_signal.rolling(window=480).std()
        momentum_norm = (momentum_signal - momentum_signal.rolling(window=480).mean()) / momentum_signal.rolling(window=480).std()
        volume_norm = (volume_signal - volume_signal.rolling(window=480).mean()) / volume_signal.rolling(window=480).std()
        
        # Combine signals
        regime_score = (
            volatility_norm * 0.4 +
            momentum_norm * 0.3 +
            volume_norm * 0.3
        )
        
        # Classify regime
        regime = pd.cut(
            regime_score,
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=['low_activity', 'normal', 'high_activity']
        )
        
        return regime