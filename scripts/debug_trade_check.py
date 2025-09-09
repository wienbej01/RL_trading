#!/usr/bin/env python3
"""
Quick env sanity check: ensure trades occur with non-hold actions.

Loads cached Polygon OHLCV via the unified loader, builds features with the
example pipeline, runs a simple deterministic policy that alternates actions,
and reports the number of trades recorded by the environment.
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import UnifiedDataLoader
from src.features.pipeline import FeaturePipeline
from src.sim.env_intraday_rl import IntradayRLEnv


def create_feature_pipeline():
    cfg = {
        'data_source': 'polygon',
        'technical': {
            'calculate_returns': True,
            'sma_windows': [5, 10],
            'ema_windows': [5, 10],
            'calculate_atr': True,
            'atr_window': 14,
        },
        'time': {
            'extract_time_of_day': True,
        },
        'polygon': {'features': {'use_vwap_column': True}}
    }
    return FeaturePipeline(cfg)


def main():
    # Load a small slice to keep it quick (prefer cache file if present)
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    cache_try = root / 'data' / 'cache' / 'SPY_20240101_20240701_ohlcv_1min.parquet'
    if cache_try.exists():
        df = pd.read_parquet(cache_try)
    else:
        loader = UnifiedDataLoader(data_source='polygon', config_path=str(root / 'configs' / 'settings.yaml'))
        df = loader.load_ohlcv(symbol='SPY', start=pd.Timestamp('2024-01-03'), end=pd.Timestamp('2024-01-10'), timeframe='1min')
    if df.empty:
        print('No data loaded')
        return
    pipe = create_feature_pipeline()
    X = pipe.fit_transform(df)

    env = IntradayRLEnv(ohlcv=df, features=X, cash=100_000.0, point_value=5.0)
    obs, _ = env.reset()

    # Alternating policy: long for 5 bars, hold for 5 bars repeatedly
    step = 0
    done = False
    while not done:
        phase = (step // 5) % 2
        # SB3-compatible actions: 0=short, 1=hold, 2=long for IntradayRLEnv
        action = 2 if phase == 0 else 1
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    trades = env.get_trades()
    print(f"Steps={step}, trades={len(trades)}")
    # Show first few trade records if available
    for t in trades[:3]:
        print({k: t[k] for k in ['action', 'ts', 'pos', 'price'] if k in t})


if __name__ == '__main__':
    main()
