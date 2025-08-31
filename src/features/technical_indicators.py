"""
Technical indicators for the RL trading system.

This module implements a comprehensive set of technical indicators
including trend, momentum, volatility, and volume-based metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    name: str
    window: int
    params: Dict[str, Union[int, float]]


class TechnicalIndicators:
    """
    Comprehensive technical indicators implementation.
    
    This class provides a wide range of technical indicators commonly
    used in algorithmic trading and reinforcement learning.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize technical indicators.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.indicator_configs = settings.get('indicators', default={})
        
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all configured technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators
        """
        if data.empty:
            return pd.DataFrame()
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        
        # Calculate trend indicators
        result = self._add_trend_indicators(result, data)
        
        # Calculate momentum indicators
        result = self._add_momentum_indicators(result, data)
        
        # Calculate volatility indicators
        result = self._add_volatility_indicators(result, data)
        
        # Calculate volume indicators
        result = self._add_volume_indicators(result, data)
        
        # Calculate price-based indicators
        result = self._add_price_indicators(result, data)
        
        # Calculate pattern-based indicators
        result = self._add_pattern_indicators(result, data)
        
        return result
    
    def _add_trend_indicators(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators."""
        # Simple Moving Average
        result['sma_5'] = self.sma(data['close'], 5)
        result['sma_10'] = self.sma(data['close'], 10)
        result['sma_20'] = self.sma(data['close'], 20)
        result['sma_50'] = self.sma(data['close'], 50)
        
        # Exponential Moving Average
        result['ema_5'] = self.ema(data['close'], 5)
        result['ema_10'] = self.ema(data['close'], 10)
        result['ema_20'] = self.ema(data['close'], 20)
        result['ema_50'] = self.ema(data['close'], 50)
        
        # Weighted Moving Average
        result['wma_10'] = self.wma(data['close'], 10)
        result['wma_20'] = self.wma(data['close'], 20)
        
        # MACD
        macd_data = self.macd(data['close'])
        result['macd'] = macd_data['macd']
        result['macd_signal'] = macd_data['signal']
        result['macd_histogram'] = macd_data['histogram']
        
        # ADX (Average Directional Index)
        adx_data = self.adx(data)
        result['adx'] = adx_data['adx']
        result['di_plus'] = adx_data['di_plus']
        result['di_minus'] = adx_data['di_minus']
        
        # Parabolic SAR
        result['sar'] = self.parabolic_sar(data)
        
        # Ichimoku Cloud
        ichimoku_data = self.ichimoku(data)
        result.update(ichimoku_data)
        
        return result
    
    def _add_momentum_indicators(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        # RSI (Relative Strength Index)
        result['rsi_14'] = self.rsi(data['close'], 14)
        result['rsi_21'] = self.rsi(data['close'], 21)
        
        # Stochastic Oscillator
        stoch_data = self.stochastic(data)
        result['stoch_k'] = stoch_data['k']
        result['stoch_d'] = stoch_data['d']
        
        # CCI (Commodity Channel Index)
        result['cci_20'] = self.cci(data, 20)
        
        # Williams %R
        result['williams_r'] = self.williams_r(data)
        
        # ROC (Rate of Change)
        result['roc_10'] = self.roc(data['close'], 10)
        result['roc_20'] = self.roc(data['close'], 20)
        
        # Momentum
        result['momentum_10'] = self.momentum(data['close'], 10)
        result['momentum_20'] = self.momentum(data['close'], 20)
        
        return result
    
    def _add_volatility_indicators(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        # Bollinger Bands
        bb_data = self.bollinger_bands(data['close'])
        result.update(bb_data)
        
        # ATR (Average True Range)
        result['atr_14'] = self.atr(data, 14)
        result['atr_20'] = self.atr(data, 20)
        
        # Keltner Channels
        kc_data = self.keltner_channels(data)
        result.update(kc_data)
        
        # Standard Deviation
        result['std_10'] = self.std(data['close'], 10)
        result['std_20'] = self.std(data['close'], 20)
        
        # Historical Volatility
        result['hv_20'] = self.historical_volatility(data['close'], 20)
        result['hv_50'] = self.historical_volatility(data['close'], 50)
        
        # Donchian Channel
        dc_data = self.donchian_channel(data)
        result.update(dc_data)
        
        return result
    
    def _add_volume_indicators(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume Moving Average
        result['volume_sma_10'] = self.sma(data['volume'], 10)
        result['volume_sma_20'] = self.sma(data['volume'], 20)
        
        # Volume Weighted Moving Average
        result['vwma_20'] = self.vwma(data)
        
        # On-Balance Volume
        result['obv'] = self.obv(data)
        
        # Accumulation/Distribution
        result['ad'] = self.accumulation_distribution(data)
        
        # Money Flow Index
        result['mfi_14'] = self.money_flow_index(data, 14)
        
        # Volume Rate of Change
        result['volume_roc_10'] = self.roc(data['volume'], 10)
        
        # Chaikin Money Flow
        result['cmf'] = self.chaikin_money_flow(data, 20)
        
        return result
    
    def _add_price_indicators(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based indicators."""
        # Price Position
        result['price_position'] = self.price_position(data)
        
        # Price Change
        result['price_change'] = data['close'].pct_change()
        result['price_change_abs'] = data['close'].diff()
        
        # High-Low Range
        result['hl_range'] = data['high'] - data['low']
        result['hl_pct'] = (data['high'] - data['low']) / data['close']
        
        # Open-Close Range
        result['oc_range'] = data['close'] - data['open']
        result['oc_pct'] = (data['close'] - data['open']) / data['open']
        
        # True Range
        result['true_range'] = self.true_range(data)
        
        # Close Position
        result['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        return result
    
    def _add_pattern_indicators(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern-based indicators."""
        # Candlestick patterns
        result['doji'] = self.detect_doji(data)
        result['hammer'] = self.detect_hammer(data)
        result['shooting_star'] = self.detect_shooting_star(data)
        result['engulfing'] = self.detect_engulfing(data)
        
        # Support/Resistance
        result['support'] = self.find_support(data)
        result['resistance'] = self.find_resistance(data)
        
        # Price action patterns
        result['breakout'] = self.detect_breakout(data)
        result['reversal'] = self.detect_reversal(data)
        
        return result
    
    # Basic indicator functions
    def sma(self, series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=window).mean()
    
    def ema(self, series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=window, adjust=False).mean()
    
    def wma(self, series: pd.Series, window: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, window + 1)
        return series.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    def macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.ema(series, fast)
        ema_slow = self.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """RSI (Relative Strength Index)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def stochastic(self, data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        low_min = data['low'].rolling(window=k_window).min()
        high_max = data['high'].rolling(window=k_window).max()
        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def adx(self, data: pd.DataFrame, window: int = 14) -> Dict[str, pd.Series]:
        """ADX (Average Directional Index)."""
        # Calculate True Range
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = data['high'] - data['high'].shift()
        down_move = data['low'].shift() - data['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the values
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / tr.rolling(window=window).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / tr.rolling(window=window).mean())
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return {
            'adx': adx,
            'di_plus': plus_di,
            'di_minus': minus_di
        }
    
    def bollinger_bands(self, series: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        sma = self.sma(series, window)
        std = series.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bb_width = (upper_band - lower_band) / sma
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_width': bb_width
        }
    
    def atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """ATR (Average True Range)."""
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    def cci(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """CCI (Commodity Channel Index)."""
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    def williams_r(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Williams %R."""
        high_max = data['high'].rolling(window=window).max()
        low_min = data['low'].rolling(window=window).min()
        wr = -100 * (high_max - data['close']) / (high_max - low_min)
        return wr
    
    def roc(self, series: pd.Series, window: int) -> pd.Series:
        """ROC (Rate of Change)."""
        return series.pct_change(window) * 100
    
    def momentum(self, series: pd.Series, window: int) -> pd.Series:
        """Momentum."""
        return series - series.shift(window)
    
    def keltner_channels(self, data: pd.DataFrame, window: int = 20, multiplier: float = 2) -> Dict[str, pd.Series]:
        """Keltner Channels."""
        ema = self.ema(data['close'], window)
        atr = self.atr(data, window)
        
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        
        return {
            'kc_upper': upper,
            'kc_middle': ema,
            'kc_lower': lower
        }
    
    def std(self, series: pd.Series, window: int) -> pd.Series:
        """Standard Deviation."""
        return series.rolling(window=window).std()
    
    def historical_volatility(self, series: pd.Series, window: int) -> pd.Series:
        """Historical Volatility."""
        returns = series.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252) * 100
    
    def donchian_channel(self, data: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
        """Donchian Channel."""
        upper = data['high'].rolling(window=window).max()
        lower = data['low'].rolling(window=window).min()
        middle = (upper + lower) / 2
        
        return {
            'dc_upper': upper,
            'dc_middle': middle,
            'dc_lower': lower
        }
    
    def vwma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Volume Weighted Moving Average."""
        tp = (data['high'] + data['low'] + data['close']) / 3
        return (tp * data['volume']).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
    
    def obv(self, data: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        obv = pd.Series(index=data.index, dtype=float)
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    def accumulation_distribution(self, data: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution Line."""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        ad = (clv * data['volume']).cumsum()
        return ad
    
    def money_flow_index(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Money Flow Index."""
        tp = (data['high'] + data['low'] + data['close']) / 3
        rmf = tp * data['volume']
        
        positive_flow = rmf.where(tp > tp.shift(), 0).rolling(window=window).sum()
        negative_flow = rmf.where(tp < tp.shift(), 0).rolling(window=window).sum()
        
        money_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + money_ratio))
    
    def chaikin_money_flow(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Chaikin Money Flow."""
        adl = self.accumulation_distribution(data)
        return adl.rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
    
    def parabolic_sar(self, data: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """Parabolic SAR."""
        length = len(data)
        sar = pd.Series(index=data.index, dtype=float)
        ep = pd.Series(index=data.index, dtype=float)
        af_series = pd.Series(index=data.index, dtype=float)
        
        # Initialize
        sar.iloc[0] = data['low'].iloc[0]
        ep.iloc[0] = data['high'].iloc[0]
        af_series.iloc[0] = af
        
        for i in range(1, length):
            if (data['close'].iloc[i] > data['close'].iloc[i-1] and data['close'].iloc[i-1] > data['close'].iloc[i-2]):
                # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af_series.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                if data['high'].iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = data['high'].iloc[i]
                    af_series.iloc[i] = min(af_series.iloc[i-1] + af, max_af)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af_series.iloc[i] = af_series.iloc[i-1]
                    
                if data['low'].iloc[i] < sar.iloc[i]:
                    sar.iloc[i] = data['high'].iloc[i]
                    ep.iloc[i] = data['low'].iloc[i]
                    af_series.iloc[i] = af
                    
            elif (data['close'].iloc[i] < data['close'].iloc[i-1] and data['close'].iloc[i-1] < data['close'].iloc[i-2]):
                # Downtrend
                sar.iloc[i] = sar.iloc[i-1] - af_series.iloc[i-1] * (sar.iloc[i-1] - ep.iloc[i-1])
                
                if data['low'].iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = data['low'].iloc[i]
                    af_series.iloc[i] = min(af_series.iloc[i-1] + af, max_af)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af_series.iloc[i] = af_series.iloc[i-1]
                    
                if data['high'].iloc[i] > sar.iloc[i]:
                    sar.iloc[i] = data['low'].iloc[i]
                    ep.iloc[i] = data['high'].iloc[i]
                    af_series.iloc[i] = af
            else:
                # Continue trend
                sar.iloc[i] = sar.iloc[i-1] + af_series.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                if data['high'].iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = data['high'].iloc[i]
                    af_series.iloc[i] = min(af_series.iloc[i-1] + af, max_af)
                elif data['low'].iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = data['low'].iloc[i]
                    af_series.iloc[i] = min(af_series.iloc[i-1] + af, max_af)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af_series.iloc[i] = af_series.iloc[i-1]
        
        return sar
    
    def ichimoku(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Ichimoku Cloud."""
        # Conversion Line (9-period)
        nine_period_high = data['high'].rolling(window=9).max()
        nine_period_low = data['low'].rolling(window=9).min()
        conversion_line = (nine_period_high + nine_period_low) / 2
        
        # Base Line (26-period)
        twenty_six_period_high = data['high'].rolling(window=26).max()
        twenty_six_period_low = data['low'].rolling(window=26).min()
        base_line = (twenty_six_period_high + twenty_six_period_low) / 2
        
        # Leading Span A
        leading_span_a = (conversion_line + base_line) / 2
        
        # Leading Span B (52-period)
        fifty_two_period_high = data['high'].rolling(window=52).max()
        fifty_two_period_low = data['low'].rolling(window=52).min()
        leading_span_b = (fifty_two_period_high + fifty_two_period_low) / 2
        
        # Lagging Span
        lagging_span = data['close'].shift(-26)
        
        return {
            'ichimoku_conversion': conversion_line,
            'ichimoku_base': base_line,
            'ichimoku_span_a': leading_span_a,
            'ichimoku_span_b': leading_span_b,
            'ichimoku_lagging': lagging_span
        }
    
    def true_range(self, data: pd.DataFrame) -> pd.Series:
        """True Range."""
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def price_position(self, data: pd.DataFrame) -> pd.Series:
        """Price position within daily range."""
        return (data['close'] - data['low']) / (data['high'] - data['low'])
    
    def detect_doji(self, data: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """Detect Doji candlestick pattern."""
        body_size = abs(data['close'] - data['open'])
        range_size = data['high'] - data['low']
        return (body_size / range_size) < threshold
    
    def detect_hammer(self, data: pd.DataFrame, threshold: float = 0.6) -> pd.Series:
        """Detect Hammer candlestick pattern."""
        body_size = abs(data['close'] - data['open'])
        range_size = data['high'] - data['low']
        
        # Use vectorized operations instead of conditional statements
        is_bullish = data['close'] > data['open']
        lower_shadow_bullish = data['open'] - data['low']
        lower_shadow_bearish = data['close'] - data['low']
        lower_shadow = np.where(is_bullish, lower_shadow_bullish, lower_shadow_bearish)
        
        upper_shadow_bullish = data['high'] - data['open']
        upper_shadow_bearish = data['high'] - data['close']
        upper_shadow = np.where(is_bullish, upper_shadow_bullish, upper_shadow_bearish)
        
        is_hammer = (lower_shadow > threshold * body_size) & (upper_shadow < 0.2 * body_size)
        return is_hammer
    
    def detect_shooting_star(self, data: pd.DataFrame, threshold: float = 0.6) -> pd.Series:
        """Detect Shooting Star candlestick pattern."""
        body_size = abs(data['close'] - data['open'])
        
        # Use vectorized operations instead of conditional statements
        is_bullish = data['close'] > data['open']
        upper_shadow_bullish = data['high'] - data['open']
        upper_shadow_bearish = data['high'] - data['close']
        upper_shadow = np.where(is_bullish, upper_shadow_bullish, upper_shadow_bearish)
        
        lower_shadow_bullish = data['open'] - data['low']
        lower_shadow_bearish = data['close'] - data['low']
        lower_shadow = np.where(is_bullish, lower_shadow_bullish, lower_shadow_bearish)
        
        is_shooting_star = (upper_shadow > threshold * body_size) & (lower_shadow < 0.2 * body_size)
        return is_shooting_star
    
    def detect_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect Engulfing candlestick pattern."""
        prev_body_size = abs(data['close'].shift(1) - data['open'].shift(1))
        current_body_size = abs(data['close'] - data['open'])
        
        bullish_engulfing = (data['close'] > data['open']) & (data['open'].shift(1) > data['close'].shift(1)) & \
                           (current_body_size > prev_body_size)
        
        bearish_engulfing = (data['close'] < data['open']) & (data['open'].shift(1) < data['close'].shift(1)) & \
                           (current_body_size > prev_body_size)
        
        return bullish_engulfing | bearish_engulfing
    
    def find_support(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Find support levels."""
        return data['low'].rolling(window=window).min()
    
    def find_resistance(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Find resistance levels."""
        return data['high'].rolling(window=window).max()
    
    def detect_breakout(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect price breakout."""
        resistance = self.find_resistance(data, window)
        support = self.find_support(data, window)
        
        breakout_up = data['close'] > resistance.shift(1)
        breakout_down = data['close'] < support.shift(1)
        
        return breakout_up | breakout_down
    
    def detect_reversal(self, data: pd.DataFrame, window: int = 5) -> pd.Series:
        """Detect potential reversal patterns."""
        # Simple reversal detection based on price action
        price_change = data['close'].pct_change(window)
        
        reversal_up = (price_change < -0.02) & (data['close'] > data['open'])
        reversal_down = (price_change > 0.02) & (data['close'] < data['open'])
        
        return reversal_up | reversal_down


# Standalone functions for direct use
def calculate_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate returns for a given series.
    
    Args:
        series: Price series
        periods: Number of periods to look back
        
    Returns:
        Returns series
    """
    return series.pct_change(periods, fill_method=None)


def calculate_log_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate log returns for a given series.
    
    Args:
        series: Price series
        periods: Number of periods to look back
        
    Returns:
        Log returns series
    """
    return np.log(series / series.shift(periods))


def calculate_sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: Price series
        window: Lookback window
        
    Returns:
        SMA series
    """
    return prices.rolling(window=window).mean()


def calculate_ema(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: Price series
        window: Lookback window
        
    Returns:
        EMA series
    """
    return prices.ewm(span=window, adjust=False).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Lookback window for ATR
        
    Returns:
        ATR series
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(window=window).mean()
    
    return atr


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        window: Lookback window for RSI
        
    Returns:
        RSI series
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()
    
    # Calculate RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period

    Returns:
        Dictionary with MACD line, signal line, and histogram
    """
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period).mean()

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Price series
        window: Lookback window
        num_std: Number of standard deviations

    Returns:
        Dictionary with upper band, middle band, lower band, and width
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    bb_width = (upper_band - lower_band) / rolling_mean

    return {
        'upper': upper_band,
        'middle': rolling_mean,
        'lower': lower_band,
        'width': bb_width
    }


def calculate_stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                                  k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: Period for %K calculation
        d_period: Period for %D calculation

    Returns:
        Dictionary with %K and %D series
    """
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()

    return {
        'k': k_percent,
        'd': d_percent
    }


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, 
                        window: int = 14) -> pd.Series:
    """
    Calculate Williams %R.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Lookback window
        
    Returns:
        Williams %R series
    """
    high_max = high.rolling(window=window).max()
    low_min = low.rolling(window=window).min()
    wr = -100 * (high_max - close) / (high_max - low_min)
    return wr
