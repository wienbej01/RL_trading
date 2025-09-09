#!/usr/bin/env python3
"""
Generate an extensive trading performance report.

Usage:
  python scripts/generate_performance_report.py \
    --model runs/walkforward_spy/fold_01/model.zip \
    --data path/to/data.parquet \
    --out reports/backtest_report.json

Outputs:
  - JSON report with performance, risk, and trade analytics
  - Plots in the same output directory
"""
import argparse
import sys
from pathlib import Path

# Add repo root to path for package-style imports (src.*)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import Settings
from src.evaluation.backtest_evaluator import BacktestEvaluator
import numpy as np
from sb3_contrib import RecurrentPPO
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained SB3 model')
    parser.add_argument('--data', required=True, help='Path to backtest data (parquet)')
    parser.add_argument('--out', required=True, help='Output JSON path for report')
    parser.add_argument('--start', help='Override start date (YYYY-MM-DD) for backtest window')
    parser.add_argument('--end', help='Override end date (YYYY-MM-DD) for backtest window')
    parser.add_argument('--seeds', type=int, default=1, help='Number of seeds for multi-seed validation (default: 1)')
    args = parser.parse_args()

    settings = Settings.from_paths('configs/settings.yaml')
    # Optionally override evaluation date window in settings for this run
    if args.start or args.end:
        from copy import deepcopy
        cfg = deepcopy(settings._cfg)
        eval_block = cfg.setdefault('evaluation', {})
        if args.start:
            eval_block['start_date'] = args.start
        if args.end:
            eval_block['end_date'] = args.end
        settings._cfg = cfg

    evaluator = BacktestEvaluator(settings)
    if args.seeds > 1:
        results = []
        with tqdm(total=args.seeds, desc="Running backtests") as pbar:
            for seed in range(args.seeds):
                np.random.seed(seed)
                model = RecurrentPPO.load(args.model)
                result = evaluator.run_backtest(model_path=None, data_path=args.data, model=model, progress_bar=pbar)
                if result is None:
                    print(f"Seed {seed} backtest failed; skipping")
                    continue
                results.append(result)
                pbar.update(1)
        if results:
            avg_result = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
            result = avg_result
        else:
            raise SystemExit('All backtests failed; check logs')
    else:
        result = evaluator.run_backtest(model_path=args.model, data_path=args.data)
        if result is None:
            raise SystemExit('Backtest failed; check logs')

    out_path = Path(args.out)
    evaluator.save_backtest_report(str(out_path))
    evaluator.generate_backtest_plots(str(out_path.parent))
    print(f"Report saved: {out_path}")
    print(f"Plots saved to: {out_path.parent}")


if __name__ == '__main__':
    main()
