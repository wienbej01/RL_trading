import pandas as pd
import numpy as np

def calculate_fvg(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Calculate Fair Value Gaps (FVG).

    A bullish FVG is formed when the low of the current candle is higher than the high of the previous candle.
    A bearish FVG is formed when the high of the current candle is lower than the low of the previous candle.

    Args:
        high: High price series
        low: Low price series

    Returns:
        A series with 1 for bullish FVG, -1 for bearish FVG, and 0 otherwise.
    """
    fvg = pd.Series(0, index=high.index)
    
    # Bullish FVG
    bullish_fvg = high.shift(1) < low.shift(-1)
    fvg[bullish_fvg] = 1
    
    # Bearish FVG
    bearish_fvg = low.shift(1) > high.shift(-1)
    fvg[bearish_fvg] = -1
    
    return fvg

def calculate_spread(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Calculate the spread between two series (e.g., ask/bid or high/low).
    """
    return series1 - series2

def calculate_microprice(bid_price: pd.Series, bid_size: pd.Series, ask_price: pd.Series, ask_size: pd.Series) -> pd.Series:
    """
    Calculate microprice.
    """
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)

def calculate_queue_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    """
    Calculate queue imbalance.
    """
    return bid_size / (bid_size + ask_size)

def calculate_order_flow_imbalance(bid_price: pd.Series, bid_size: pd.Series, ask_price: pd.Series, ask_size: pd.Series) -> pd.Series:
    """
    Calculate order flow imbalance.
    This is a simplified version.
    """
    price_change = bid_price.diff()
    buy_pressure = (price_change > 0) * bid_size
    sell_pressure = (price_change < 0) * ask_size
    return buy_pressure - sell_pressure

def calculate_vwap(close: pd.Series, volume: pd.Series, vwap_col: pd.Series = None, window: int = 20) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    If vwap_col is provided, it will be returned.
    """
    if vwap_col is not None:
        return vwap_col
    return (close * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()

def calculate_twap(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Time Weighted Average Price (TWAP).
    """
    return close.rolling(window=window).mean()

def calculate_price_impact(close: pd.Series, volume: pd.Series, bid_price: pd.Series, ask_price: pd.Series) -> pd.Series:
    """
    Calculate price impact.
    A simple measure of price impact.
    """
    spread = ask_price - bid_price
    return spread / volume
