#!/usr/bin/env python3
"""
Quick CPU sanity training run:
 - Generates tiny synthetic OHLCV + features (2 days of 1-min bars)
 - Trains RecurrentPPO for a small number of steps on CPU
 - Prints elapsed time and approx samples/sec

Usage:
  PYTHONPATH=. python scripts/sanity_train_cpu.py --steps 10000
"""
import argparse
import time
from pathlib import Path
import sys

# Add src to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils.config_loader import Settings
from src.rl.train import train_ppo_lstm, TrainingConfig


def make_synth_data() -> tuple[Path, Path]:
    data_dir = Path('data/raw'); data_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path('data/features'); feat_dir.mkdir(parents=True, exist_ok=True)

    idx_parts = []
    for day in [pd.Timestamp('2024-06-03'), pd.Timestamp('2024-06-04')]:
        base = pd.date_range(day.tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=30),
                             day.tz_localize('America/New_York') + pd.Timedelta(hours=16),
                             freq='1min', inclusive='left')
        idx_parts.append(base)
    idx = idx_parts[0].append(idx_parts[1])

    rng = np.random.default_rng(0)
    price = 500 + np.cumsum(rng.normal(0, 0.1, size=len(idx)))
    high = price + np.abs(rng.normal(0, 0.05, size=len(idx)))
    low  = price - np.abs(rng.normal(0, 0.05, size=len(idx)))
    openp = price + rng.normal(0, 0.02, size=len(idx))
    close = price
    vol = (rng.random(size=len(idx))*1000+1000).astype(int)

    df = pd.DataFrame({'open': openp, 'high': high, 'low': low, 'close': close, 'volume': vol}, index=idx)
    p_data = Path('data/raw/SPY_1min.parquet')
    df.to_parquet(p_data)

    feat = pd.DataFrame(index=df.index)
    feat['atr'] = (df['high']-df['low']).rolling(14, min_periods=1).mean().fillna(0.01)
    feat['spread'] = 0.01
    feat['rvol'] = (df['volume']/df['volume'].rolling(60, min_periods=1).mean()).fillna(1.0)
    feat['dist_last_swing_low'] = np.abs(rng.random(len(feat)))*0.2
    feat['dist_last_swing_high'] = np.abs(rng.random(len(feat)))*0.2

    p_feat = Path('data/features/SPY_features.parquet')
    feat.to_parquet(p_feat)

    return p_data, p_feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=10000, help='Total training timesteps')
    args = ap.parse_args()

    data_path, feat_path = make_synth_data()
    settings = Settings.from_paths('configs/settings.yaml')

    # Training config overrides for CPU saturation on 8 cores
    cfg = TrainingConfig()
    cfg.total_steps = int(args.steps)
    cfg.n_steps = 1024
    cfg.batch_size = 4096
    cfg.ent_coef = 0.01
    cfg.verbose = 1
    cfg.device = 'cpu'

    t0 = time.time()
    model = train_ppo_lstm(settings, str(data_path), str(feat_path), 'models/sanity_model', cfg)
    dt = time.time() - t0

    steps = int(args.steps)
    sps = steps / dt if dt > 0 else 0.0
    print(f"Sanity run complete in {dt:.2f}s | ~{sps:.1f} steps/sec | n_envs={settings.get('train','n_envs',default=1)}")


if __name__ == '__main__':
    main()

