from __future__ import annotations

import numpy as np
import pandas as pd
from warnings import warn


# ---------- OHLCV proxy microstructure ----------

def _z(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)


def ofi_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Tick-rule flow proxy using returns and volume.
    ofi_proxy = sign(close_t − close_{t−1}) * (volume_t − volume_{t−1})
    Daily z-scored to avoid regime drift. Positive = buy pressure.
    """
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("ofi_proxy requires columns: close, volume")
    sgn = np.sign(pd.to_numeric(df["close"], errors="coerce").diff().fillna(0.0))
    dvol = pd.to_numeric(df["volume"], errors="coerce").diff().fillna(0.0)
    out = (sgn * dvol).astype("float32")
    try:
        return out.groupby(df.index.date).transform(_z).astype("float32")
    except Exception:
        return out.astype("float32")


def bar_imbalance(df: pd.DataFrame) -> pd.Series:
    """Close location within the bar: (close − open)/(high − low), clipped to [-1, 1]."""
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError("bar_imbalance requires columns: open, high, low, close")
    rng = (pd.to_numeric(df["high"], errors="coerce") - pd.to_numeric(df["low"], errors="coerce")).replace(0, np.nan)
    bi = (pd.to_numeric(df["close"], errors="coerce") - pd.to_numeric(df["open"], errors="coerce")) / rng
    return bi.clip(-1, 1).fillna(0.0).astype("float32")


def spread_bps_hl(df: pd.DataFrame, win: int = 5) -> pd.Series:
    """HL-based spread proxy: median rolling (high-low)/close * 1e4."""
    need = {"high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError("spread_bps_hl requires columns: high, low, close")
    sp = (pd.to_numeric(df["high"], errors="coerce") - pd.to_numeric(df["low"], errors="coerce")).rolling(win, min_periods=1).median()
    mid = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan)
    out = (sp / mid * 1e4).fillna(0.0).astype("float32")
    return out


def quote_intensity_proxy(df: pd.DataFrame, w: int = 5) -> pd.Series:
    """Cheap activity proxy: zscore of rolling root-sum-squared returns over window w."""
    if "close" not in df.columns:
        raise ValueError("quote_intensity_proxy requires column: close")
    ret = pd.to_numeric(df["close"], errors="coerce").pct_change().fillna(0.0)
    rv = ret.rolling(w, min_periods=1).apply(lambda x: float(np.sqrt((x**2).sum())), raw=True)
    try:
        out = rv.groupby(df.index.date).transform(_z).fillna(0.0).astype("float32")
    except Exception:
        out = rv.fillna(0.0).astype("float32")
    return out


def signed_vol_delta(df: pd.DataFrame) -> pd.Series:  # type: ignore[override]
    """OHLCV signed volume delta with daily z-score normalization."""
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("signed_vol_delta requires columns: close, volume")
    sgn = np.sign(pd.to_numeric(df["close"], errors="coerce").diff().fillna(0.0))
    dvol = pd.to_numeric(df["volume"], errors="coerce").diff().fillna(0.0)
    out = (sgn * dvol).astype("float32")
    try:
        return out.groupby(df.index.date).transform(_z).astype("float32")
    except Exception:
        return out.astype("float32")


def queue_imbalance_proxy(df: pd.DataFrame, w: int = 10) -> pd.Series:
    """Volume-tilt proxy over window w: (vol_up - vol_down)/(vol_up + vol_down)."""
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("queue_imbalance_proxy requires columns: close, volume")
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    up = (close > close.shift(1)).astype(int)
    dn = (close < close.shift(1)).astype(int)
    vol_up = (vol * up).rolling(w, min_periods=1).sum()
    vol_dn = (vol * dn).rolling(w, min_periods=1).sum()
    qi = (vol_up - vol_dn) / (vol_up + vol_dn + 1e-9)
    return qi.fillna(0.0).astype("float32")


OHLCV_MICRO = {
  "ofi_proxy": ofi_proxy,
  "bar_imbalance": bar_imbalance,
  "spread_bps_hl": spread_bps_hl,
  "quote_intensity_proxy": quote_intensity_proxy,
  "signed_vol_delta": signed_vol_delta,
  "queue_imbalance_proxy": queue_imbalance_proxy,
}



def spread_bps(df: pd.DataFrame, bid_col: str = "bid", ask_col: str = "ask", close_col: str = "close") -> pd.Series:
    """Bid-ask spread in basis points with sensible fallback.

    If bid/ask are available: ((ask - bid) / mid) * 1e4 where mid=(bid+ask)/2.
    Else fallback to (rolling-median(high-low)/close) * 1e4 and WARN.
    """
    if bid_col in df.columns and ask_col in df.columns:
        mid = (pd.to_numeric(df[bid_col], errors="coerce") + pd.to_numeric(df[ask_col], errors="coerce")) / 2.0
        sp = (pd.to_numeric(df[ask_col], errors="coerce") - pd.to_numeric(df[bid_col], errors="coerce")) / mid.replace(0.0, np.nan)
        return (sp * 1e4).astype("float32")
    # Fallback: HL proxy
    warn("spread_bps: using HL proxy (quotes unavailable)")
    if not {"high", "low", close_col}.issubset(df.columns):
        raise ValueError("spread_bps fallback requires columns: high, low, and close")
    hlm = (pd.to_numeric(df["high"], errors="coerce") - pd.to_numeric(df["low"], errors="coerce")).rolling(5).median()
    sp = (hlm / pd.to_numeric(df[close_col], errors="coerce").replace(0.0, np.nan)) * 1e4
    return sp.astype("float32")


def signed_vol_delta(df: pd.DataFrame) -> pd.Series:
    """Signed volume delta: sign(close - close.shift(1)) * (volume - volume.shift(1))."""
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("signed_vol_delta requires columns: close, volume")
    up = (pd.to_numeric(df["close"], errors="coerce") > pd.to_numeric(df["close"], errors="coerce").shift(1)).astype(int) - (
        pd.to_numeric(df["close"], errors="coerce") < pd.to_numeric(df["close"], errors="coerce").shift(1)
    ).astype(int)
    dv = (pd.to_numeric(df["volume"], errors="coerce") - pd.to_numeric(df["volume"], errors="coerce").shift(1)).fillna(0)
    return (up * dv).astype("float32")


def ofi_best(
    df: pd.DataFrame,
    bid: str = "bid",
    ask: str = "ask",
    bsz: str = "bid_size",
    asz: str = "ask_size",
) -> pd.Series:
    """Best-level Order Flow Imbalance (positive = buy pressure) with a coarse fallback.

    If bid/ask and their sizes are available, uses standard best-quote OFI. Otherwise falls back to
    sign(close change) * volume change as a coarse proxy.
    """
    need = {bid, ask, bsz, asz}
    if not need.issubset(df.columns):
        warn("ofi_best: missing L1 quotes; falling back to trade imbalance")
        if not {"close", "volume"}.issubset(df.columns):
            raise ValueError("ofi_best fallback requires columns: close, volume")
        tick = (pd.to_numeric(df["close"], errors="coerce") - pd.to_numeric(df["close"], errors="coerce").shift(1)).fillna(0)
        vi = (pd.to_numeric(df["volume"], errors="coerce") - pd.to_numeric(df["volume"], errors="coerce").shift(1)).fillna(0)
        return (np.sign(tick) * vi).astype("float32")

    db = pd.to_numeric(df[bid], errors="coerce").diff().fillna(0)
    qb = pd.to_numeric(df[bsz], errors="coerce").diff().fillna(0)
    da = pd.to_numeric(df[ask], errors="coerce").diff().fillna(0)
    qa = pd.to_numeric(df[asz], errors="coerce").diff().fillna(0)

    term_bid = (db.gt(0).astype(int) * qb) - (db.lt(0).astype(int) * pd.to_numeric(df[bsz], errors="coerce").shift(1).fillna(method="bfill"))
    term_ask = -(da.lt(0).astype(int) * qa) + (da.gt(0).astype(int) * pd.to_numeric(df[asz], errors="coerce").shift(1).fillna(method="bfill"))
    return (term_bid + term_ask).astype("float32")
