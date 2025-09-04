#!/usr/bin/env python3
"""
Walk-forward training and evaluation CLI with embargo and optional per-regime reporting.

Usage:
  PYTHONPATH=. python scripts/walkforward_train_eval.py \
    --config configs/settings.yaml \
    --data data/raw/BBVA_1min.parquet \
    --features data/features/BBVA_features.parquet \
    --output runs/wf_bbva \
    --train-days 60 --test-days 20 --embargo-minutes 60
"""
import argparse
from pathlib import Path
from src.utils.config_loader import Settings
from src.rl.train import walk_forward_training


def main():
    ap = argparse.ArgumentParser(description="Walk-forward training + evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-days", type=int, default=60)
    ap.add_argument("--test-days", type=int, default=20)
    ap.add_argument("--embargo-minutes", type=int, default=60)
    args = ap.parse_args()

    settings = Settings.from_paths(args.config)
    # override WF params
    settings._config['walkforward'] = {
        'train_days': args.train_days,
        'test_days': args.test_days,
        'embargo_minutes': args.embargo_minutes,
    }

    # run WF
    res = walk_forward_training(settings, args.data, args.features, args.output, training_config=None)  # training_config built inside train
    print("Walk-forward completed. Results in:", args.output)


if __name__ == "__main__":
    main()

