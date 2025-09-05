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

# Add src to path for in-repo execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_loader import Settings
from evaluation.backtest_evaluator import BacktestEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained SB3 model')
    parser.add_argument('--data', required=True, help='Path to backtest data (parquet)')
    parser.add_argument('--out', required=True, help='Output JSON path for report')
    args = parser.parse_args()

    settings = Settings.from_paths('configs/settings.yaml')
    evaluator = BacktestEvaluator(settings)
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
