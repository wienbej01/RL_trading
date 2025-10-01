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
    Calculate the spread as second minus first (e.g., ask - bid), expected positive.
    """
    return (series2 - series1)

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

def calculate_ofi_best(bid_price: pd.Series, bid_size: pd.Series, ask_price: pd.Series, ask_size: pd.Series) -> pd.Series:
    """
    Best-quote Order Flow Imbalance (OFI_best).

    positive = buy pressure

    Implements the standard best-quote OFI definition using best bid/ask prices
    (p_b, p_a) and sizes (q_b, q_a):

      OFI_t = 1{Δp_b>0} Δq_b - 1{Δp_b<0} q_{b,t-1} - 1{Δp_a<0} Δq_a + 1{Δp_a>0} q_{a,t-1}

    where Δq denotes first difference of queue sizes at the respective side.
    """
    dpb = bid_price.diff()
    dpa = ask_price.diff()
    dqb = bid_size.diff()
    dqa = ask_size.diff()

    term_bid_up = (dpb > 0).astype(float) * dqb.fillna(0.0)
    term_bid_dn = (dpb < 0).astype(float) * bid_size.shift(1).fillna(0.0)
    term_ask_dn = (dpa < 0).astype(float) * dqa.fillna(0.0)
    term_ask_up = (dpa > 0).astype(float) * ask_size.shift(1).fillna(0.0)

    ofi = term_bid_up - term_bid_dn - term_ask_dn + term_ask_up
    # First point undefined due to diffs
    try:
        ofi.iloc[0] = np.nan
    except Exception:
        pass
    return ofi


def calculate_order_flow_imbalance(bid_price: pd.Series, bid_size: pd.Series, ask_price: pd.Series, ask_size: pd.Series) -> pd.Series:
    """
    Order Flow Imbalance at best quote, z-scored per day.

    positive = buy pressure
    """
    raw = calculate_ofi_best(bid_price, bid_size, ask_price, ask_size)
    # Daily z-score normalization to avoid regime drift while preserving sign
    if not isinstance(raw.index, pd.DatetimeIndex):
        return raw
    by_day = raw.groupby(raw.index.normalize())
    def _z(s: pd.Series) -> pd.Series:
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if not np.isfinite(sd) or sd == 0:
            return (s - mu) * 0.0
        return (s - mu) / sd
    return by_day.apply(_z)


def compute_bar_imbalance(open_: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Bar imbalance proxy: (close - open) * volume."""
    try:
        return (pd.to_numeric(close, errors='coerce') - pd.to_numeric(open_, errors='coerce')) * pd.to_numeric(volume, errors='coerce')
    except Exception:
        return pd.Series(index=close.index, dtype=float)


def compute_signed_vol_delta(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Signed volume delta: sign(close change) * Δvolume."""
    r = pd.to_numeric(close, errors='coerce').pct_change()
    dv = pd.to_numeric(volume, errors='coerce').diff()
    return np.sign(r).fillna(0.0) * dv.fillna(0.0)


def compute_spread_bps(bid: pd.Series = None, ask: pd.Series = None, high: pd.Series = None, low: pd.Series = None, close: pd.Series = None) -> pd.Series:
    """
    Spread in bps: prefer (ask-bid)/mid*1e4; fallback to (high-low)/close*1e4 if quotes unavailable.
    """
    if bid is not None and ask is not None:
        mid = (pd.to_numeric(bid, errors='coerce') + pd.to_numeric(ask, errors='coerce')) / 2.0
        spr = pd.to_numeric(ask, errors='coerce') - pd.to_numeric(bid, errors='coerce')
        return (spr / (mid.replace(0.0, np.nan))) * 1e4
    if high is not None and low is not None and close is not None:
        return ((pd.to_numeric(high, errors='coerce') - pd.to_numeric(low, errors='coerce')) / pd.to_numeric(close, errors='coerce').replace(0.0, np.nan)) * 1e4
    return pd.Series(index=(close.index if close is not None else (high.index if high is not None else None)), dtype=float)


def compute_quote_intensity(transactions: pd.Series = None, volume: pd.Series = None) -> pd.Series:
    """Quote/trade intensity proxy: prefer transactions per bar; fallback to volume."""
    if transactions is not None:
        return pd.to_numeric(transactions, errors='coerce')
    if volume is not None:
        return pd.to_numeric(volume, errors='coerce')
    return pd.Series(dtype=float)

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
