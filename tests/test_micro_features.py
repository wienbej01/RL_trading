import numpy as np
import pandas as pd
import pytest

from src.features.microstructure_features import calculate_ofi_best
from src.features.pipeline import FeaturePipeline


def _make_df(with_quotes=True):
    idx = pd.date_range("2024-05-01 09:30", periods=6, freq="1min", tz="America/New_York")
    close = pd.Series([100.0, 100.05, 100.04, 100.06, 100.10, 100.08], index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.02
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.02
    volume = pd.Series([1000, 1100, 1050, 1200, 1300, 1250], index=idx)
    df = pd.DataFrame({
        'open': open_.values,
        'high': high.values,
        'low': low.values,
        'close': close.values,
        'volume': volume.values,
    }, index=idx)
    if with_quotes:
        # Build a scenario where bid ticks up with size add and ask size depletes
        bid = pd.Series([99.98, 99.99, 100.00, 100.01, 100.02, 100.03], index=idx, name='bid_price')
        ask = pd.Series([100.02, 100.03, 100.04, 100.05, 100.06, 100.07], index=idx, name='ask_price')
        bid_size = pd.Series([500, 520, 550, 600, 650, 700], index=idx, name='bid_size')
        ask_size = pd.Series([700, 680, 660, 640, 620, 600], index=idx, name='ask_size')
        df['bid_price'] = bid
        df['ask_price'] = ask
        df['bid_size'] = bid_size
        df['ask_size'] = ask_size
    return df


def test_ofi_best_sign_positive_on_bid_up_ask_deplete():
    df = _make_df(with_quotes=True)
    ofi = calculate_ofi_best(df['bid_price'], df['bid_size'], df['ask_price'], df['ask_size'])
    # Skip initial NaN; last value should be positive under buy pressure
    assert float(ofi.iloc[-1]) > 0.0


def test_spread_bps_uses_quotes_then_fallback_warn(caplog):
    df = _make_df(with_quotes=True)
    cfg = {
        'microstructure': {
            'calculate_order_flow_imbalance': True,
        }
    }
    pipe = FeaturePipeline(cfg)
    feats = pipe.fit_transform(df)
    assert 'ofi_best' in feats.columns
    assert 'spread_bps' in feats.columns
    # Remove quotes to trigger HL fallback and WARN
    df2 = df.drop(columns=['bid_price','ask_price','bid_size','ask_size'])
    caplog.clear()
    feats2 = pipe.fit_transform(df2)
    assert 'spread_bps' in feats2.columns
    assert any("HL proxy for spread_bps" in rec.message for rec in caplog.records)


def test_pipeline_returns_requested_micro_columns():
    df = _make_df(with_quotes=True)
    cfg = {
        'microstructure': {
            'calculate_order_flow_imbalance': True,
        }
    }
    pipe = FeaturePipeline(cfg)
    feats = pipe.fit_transform(df)
    for col in ['ofi_best', 'signed_vol_delta', 'spread_bps']:
        assert col in feats.columns, f"missing expected micro column: {col}"

