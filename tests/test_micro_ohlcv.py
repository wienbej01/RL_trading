import numpy as np
import pandas as pd

from src.features.microstructure import OHLCV_MICRO


def make_ohlcv(n_per_day: int = 120, days: int = 2) -> pd.DataFrame:
    idx = []
    for d in range(days):
        start = pd.Timestamp("2024-01-02", tz="America/New_York") + pd.Timedelta(days=d)
        rng = pd.date_range(start=start, periods=n_per_day, freq="1min")
        idx.append(rng)
    idx = idx[0].append(idx[1:]) if len(idx) > 1 else idx[0]
    # simple random walk for close
    rs = np.random.RandomState(42)
    ret = rs.normal(0, 0.0005, size=len(idx))
    close = 100 * (1 + pd.Series(ret, index=idx)).cumprod()
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([
        open_, close
    ], axis=1).max(axis=1) + pd.Series(rs.rand(len(idx)) * 0.05, index=idx)
    low = pd.concat([
        open_, close
    ], axis=1).min(axis=1) - pd.Series(rs.rand(len(idx)) * 0.05, index=idx)
    vol = pd.Series(rs.randint(100, 1000, size=len(idx)), index=idx).astype(float)
    df = pd.DataFrame({
        "open": open_.astype(float),
        "high": high.astype(float),
        "low": low.astype(float),
        "close": close.astype(float),
        "volume": vol,
    }, index=idx)
    return df


def test_ohlcv_micro_outputs_and_zscore_behavior():
    df = make_ohlcv()

    # Compute each OHLCV micro feature and validate
    out = {}
    for name, func in OHLCV_MICRO.items():
        ser = func(df.copy())
        out[name] = ser
        # Not all NaN and finite where available
        assert ser.notna().any(), f"{name} produced all NaN"
        assert np.isfinite(ser.fillna(0)).all(), f"{name} has non-finite values"

    # Range checks for bounded signals
    if "bar_imbalance" in out:
        bi = out["bar_imbalance"].dropna()
        assert ((bi >= -1) & (bi <= 1)).all()

    if "spread_bps_hl" in out:
        sp = out["spread_bps_hl"].dropna()
        assert (sp >= 0).all()

    if "queue_imbalance_proxy" in out:
        qi = out["queue_imbalance_proxy"].dropna()
        assert ((qi >= -1) & (qi <= 1)).all()

    # Day z-scored features should have ~0 mean per day
    def assert_day_zscore_zero_mean(ser: pd.Series):
        by_day = ser.groupby(ser.index.date).mean()
        # Allow small numeric noise
        assert (by_day.abs() < 1e-2).all(), f"daily mean not ~0: {by_day.to_dict()}"

    if "ofi_proxy" in out:
        assert_day_zscore_zero_mean(out["ofi_proxy"])
    if "quote_intensity_proxy" in out:
        assert_day_zscore_zero_mean(out["quote_intensity_proxy"])
    if "signed_vol_delta" in out:
        assert_day_zscore_zero_mean(out["signed_vol_delta"])

