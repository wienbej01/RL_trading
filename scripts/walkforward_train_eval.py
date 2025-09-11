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
import sys

# Make the project importable when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import Settings
from src.rl.train import walk_forward_training, TrainingConfig
from src.utils.logging import setup_logging


def main():
    ap = argparse.ArgumentParser(description="Walk-forward training + evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-days", type=int, default=60)
    ap.add_argument("--test-days", type=int, default=20)
    ap.add_argument("--embargo-minutes", type=int, default=60)
    ap.add_argument("--total-steps", type=int, default=None, help="Override total PPO timesteps per fold (e.g., 5000 for a quick run)")
    ap.add_argument("--n-envs", type=int, default=None, help="Override number of parallel envs (set 1 to avoid subprocesses)")
    args = ap.parse_args()

    settings = Settings.from_paths(args.config)
    # override WF params
    settings._config['walkforward'] = {
        'train_days': args.train_days,
        'test_days': args.test_days,
        'embargo_minutes': args.embargo_minutes,
    }

    # Setup logging
    log_settings = settings.get('logging', default={})
    file_settings = log_settings.get('file', {})
    setup_logging(
        level=log_settings.get('level', 'INFO'),
        log_file=file_settings.get('path', 'logs/walkforward.log'),
        max_bytes=10485760, # Default 10MB, will fix parsing later
        backup_count=file_settings.get('backup_count', 5)
    )

    training_config = TrainingConfig()
    if args.total_steps is not None:
        training_config.total_steps = int(args.total_steps)

    # Optional override of n_envs to speed up or avoid subprocess teardown issues
    if args.n_envs is not None:
        try:
            settings._config.setdefault('train', {})['n_envs'] = int(args.n_envs)
        except Exception:
            pass

    # run WF
    res = walk_forward_training(settings, args.data, args.features, args.output, training_config=training_config)
    print("Walk-forward completed. Results in:", args.output)


if __name__ == "__main__":
    main()
