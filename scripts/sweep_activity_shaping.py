#!/usr/bin/env python3
"""
Small sweep over activity-shaping hyperparameters to break "hold-only" behavior.

For each combo, trains a short run and evaluates on the given backtest window,
printing total_trades and saving a tiny summary JSON per run.

Example:
  PYTHONPATH=$(pwd) venv/bin/python scripts/sweep_activity_shaping.py \
    --config configs/settings.yaml \
    --train-data data/raw/BBVA_1min.parquet \
    --train-features data/features/BBVA_features.parquet \
    --eval-data data/raw/BBVA_20250101_20250630_1min.parquet \
    --start 2025-01-01 --end 2025-06-30 \
    --steps 20000 \
    --out runs/sweep_activity

Notes:
- Forces n_envs=1 for fast CPU training.
- Applies manual VecNormalize during evaluation via BacktestEvaluator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import json

from src.utils.config_loader import Settings
from src.rl.train import train_ppo_lstm, TrainingConfig
from src.evaluation.backtest_evaluator import BacktestEvaluator


def make_settings(base_cfg: Settings,
                  open_bonus: float,
                  hold_penalty: float,
                  trade_target: int,
                  activity_bonus: float,
                  activity_penalty: float,
                  ent_coef: float) -> Settings:
    cfg = deepcopy(base_cfg)
    # Reward block
    rb = cfg._cfg.setdefault('env', {}).setdefault('reward', {})
    rb['open_bonus'] = float(open_bonus)
    rb['hold_penalty'] = float(hold_penalty)
    rb['trade_target_per_day'] = int(trade_target)
    rb['trade_activity_bonus'] = float(activity_bonus)
    rb['trade_activity_penalty'] = float(activity_penalty)
    # Training block
    tb = cfg._cfg.setdefault('train', {})
    tb['ent_coef'] = float(ent_coef)
    tb['n_envs'] = 1  # single-env for speed and Gymnasium-friendly
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Small sweep to avoid hold-only policies")
    ap.add_argument('--config', default='configs/settings.yaml')
    ap.add_argument('--train-data', required=True)
    ap.add_argument('--train-features', required=True)
    ap.add_argument('--eval-data', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--out', default='runs/sweep_activity')
    args = ap.parse_args()

    base = Settings.from_paths(args.config)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Candidate grid (kept small for quick iteration)
    open_bonuses = [0.02, 0.05]
    hold_penalties = [0.0, 0.02]
    trade_targets = [4, 8]
    ent_coefs = [0.01, 0.02]
    activity_bonus = 0.02
    activity_penalty = 0.01

    results = []
    for ob in open_bonuses:
        for hp in hold_penalties:
            for tt in trade_targets:
                for ec in ent_coefs:
                    tag = f"ob{ob:g}_hp{hp:g}_tt{tt}_ec{ec:g}"
                    sweep_dir = out_root / tag
                    sweep_dir.mkdir(parents=True, exist_ok=True)

                    cfg = make_settings(base, ob, hp, tt, activity_bonus, activity_penalty, ec)

                    # Train short run
                    tconf = TrainingConfig()
                    tconf.total_steps = int(args.steps)
                    tconf.device = 'cpu'
                    model_path = str(sweep_dir / 'model.zip')
                    try:
                        train_ppo_lstm(cfg, args.train_data, args.train_features, model_path, tconf)
                    except Exception as e:
                        print(f"[SKIP TRAIN] {tag} due to error: {e}")
                        continue

                    # Evaluate on the requested window
                    # override evaluation window
                    eval_cfg = deepcopy(cfg)
                    eval_block = eval_cfg._cfg.setdefault('evaluation', {})
                    eval_block['start_date'] = args.start
                    eval_block['end_date'] = args.end
                    be = BacktestEvaluator(eval_cfg)
                    res = be.run_backtest(model_path=model_path, data_path=args.eval_data)
                    total_trades = int(getattr(res, 'total_trades', 0) if res else 0)

                    row = {
                        'tag': tag,
                        'open_bonus': ob,
                        'hold_penalty': hp,
                        'trade_target_per_day': tt,
                        'ent_coef': ec,
                        'total_trades': total_trades
                    }
                    results.append(row)
                    with open(sweep_dir / 'summary.json', 'w') as f:
                        json.dump(row, f, indent=2)
                    print(f"{tag}: total_trades={total_trades}")

    # Save global summary
    results = sorted(results, key=lambda r: (-r['total_trades'], r['tag']))
    with open(out_root / 'sweep_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nTop results:")
    for r in results[:5]:
        print(r)


if __name__ == '__main__':
    main()

