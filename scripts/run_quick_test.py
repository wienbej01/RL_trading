#!/usr/bin/env python3
"""
Quick, bounded multi-ticker training + backtest runner.

Why: Avoids shell heredoc issues and ensures a bounded number of PPO steps with
live progress and reduced noisy env-step logs.

Usage example:
  ./venv/bin/python -u scripts/run_quick_test.py \
    --tickers AAPL MSFT NVDA \
    --train-start 2024-01-02 --train-end 2024-01-09 \
    --test-start 2024-01-10 --test-end 2024-01-12 \
    --timesteps 1024 \
    --output-dir results/quick_multi_polygon_all3_1024

Notes:
  - Requires that data already exists in data/polygon/historical (use the
    download_with_polygon_module.py beforehand if needed).
  - Writes a temporary config at configs/quick_multiticker_autogen.yaml.
  - Silences chatter from src.sim.env_intraday_rl at WARNING level.
  - Prints a heartbeat every 15 seconds while the pipeline runs.
"""
from __future__ import annotations

import argparse
import logging
import runpy
import sys
import threading
import time
from pathlib import Path

import yaml


def make_autogen_config(base_cfg: Path, out_cfg: Path, timesteps: int) -> None:
    cfg = yaml.safe_load(base_cfg.read_text())
    # Ensure both keys exist — different code paths read either block
    cfg.setdefault('train', {})['total_steps'] = int(timesteps)
    cfg.setdefault('training', {})['total_timesteps'] = int(timesteps)
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))


def heartbeat(stop_event: threading.Event, interval_sec: int = 15) -> None:
    while not stop_event.is_set():
        print(f"[heartbeat] {time.strftime('%H:%M:%S')} running...", flush=True)
        stop_event.wait(interval_sec)


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick bounded multi-ticker run")
    ap.add_argument('--tickers', nargs='+', required=True)
    ap.add_argument('--train-start', required=True)
    ap.add_argument('--train-end', required=True)
    ap.add_argument('--test-start', required=True)
    ap.add_argument('--test-end', required=True)
    ap.add_argument('--timesteps', type=int, default=1024)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--portfolio-env', action='store_true', help='Force portfolio environment')
    ap.add_argument('--base-config', default='configs/quick_multiticker.yaml')
    args = ap.parse_args()

    base_cfg = Path(args.base_config)
    if not base_cfg.exists():
        print(f"[error] Base config not found: {base_cfg}", flush=True)
        return 2

    # Prepare output directory early (used by tee if caller wants)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build an auto-generated config with bounded total steps
    auto_cfg = base_cfg.parent / 'quick_multiticker_autogen.yaml'
    make_autogen_config(base_cfg, auto_cfg, int(args.timesteps))
    print(f"[info] Wrote bounded config: {auto_cfg}", flush=True)

    # Silence very chatty logs while keeping other modules informative
    logging.getLogger('src.sim.env_intraday_rl').setLevel(logging.WARNING)
    logging.getLogger('src.sim.risk').setLevel(logging.WARNING)

    # Heartbeat while the pipeline runs so there is always console activity
    stop_evt = threading.Event()
    t = threading.Thread(target=heartbeat, args=(stop_evt, 15), daemon=True)
    t.start()

    # Build argv for the pipeline script and dispatch it
    sys.argv = [
        'scripts/run_multiticker_pipeline.py',
        '--config', str(auto_cfg),
        '--tickers', *args.tickers,
        '--train-start', args.train_start, '--train-end', args.train_end,
        '--test-start', args.test_start, '--test-end', args.test_end,
        '--output-dir', str(out_dir)
    ]
    if args.portfolio_env:
        sys.argv.append('--portfolio-env')
    try:
        runpy.run_path('scripts/run_multiticker_pipeline.py', run_name='__main__')
        return 0
    finally:
        stop_evt.set()
        t.join(timeout=1)


if __name__ == '__main__':
    raise SystemExit(main())
