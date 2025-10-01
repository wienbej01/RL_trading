# User Guide

This guide shows how to install, prepare data, generate features, train/backtest, and run embargoed walk-forward with CPCV.

## Install

- Python 3.11+
- Install deps and dev extras:
  - `pip install -r requirements.txt && pip install -e .[dev]`
- Format, lint, type-check, tests:
  - `black .`
  - `ruff check . --fix`
  - `mypy src scripts tests`
  - `pytest -m "not slow"`

## Data

- Use `UnifiedDataLoader` (Polygon-compatible) or existing parquet files.
- Timezone is normalized to America/New_York; minutes data expected.

## Features

- Build features via `FeaturePipeline` (preserves `ticker`).
- Optional feature packs: curated lists or package-defined packs.
- Screening artifacts live under `results/features/<run>/`:
  - `consensus_importance.parquet`, `curated_topN.txt`.

## Single-Split Training & Backtest

- Script: `scripts/run_multiticker_pipeline.py`
- Example:
```
python scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --train-start 2024-06-01 --train-end 2024-08-31 \
  --test-start  2024-09-01 --test-end  2024-09-30 \
  --tickers SPY AAPL MSFT \
  --output-dir results/demo_mt --portfolio-env
```
- Outputs (under `<output-dir>/backtest/`):
  - `summary.json`, `trades.csv`, `portfolio_history.csv`, `daily_report.csv`.

## Walk-Forward + CPCV (Sprint 1)

- Dry-run (list windows):
```
python scripts/run_wf.py --config configs/settings.yaml --run-name demo \
  --tickers AAPL MSFT --train-start 2024-01-01 --train-end 2024-03-31 --dry-run
```

- Full run:
```
python scripts/run_wf.py --config configs/settings.yaml --run-name demo \
  --tickers AAPL MSFT --train-start 2024-01-01 --train-end 2024-03-31 \
  --wf-train-days 60 --wf-valid-days 10 --wf-test-days 10 --wf-step-days 10 \
  --embargo-min 15 --feature-pack curated --timesteps 100000
```

- Per-window artifacts: `results/wf/<run-name>/window_<k>/`
  - `models/`, `backtest/summary.json`, `backtest/trades.csv`, `backtest/portfolio_history.csv`.
- Aggregate: `results/wf/<run-name>/aggregate.json`

## Registry & Summaries

- Registry CSV: `results/_registry/runs.csv` receives appends from pipeline and WF runs.
- Summarize recent runs:
```
python scripts/summarize_runs.py --registry results/_registry/runs.csv --sort sharpe --desc --limit 20
```

## Troubleshooting

- NaN Sharpe/Calmar warnings: occur when returns have zero variance in short windows; metrics default to 0.0 where guarded.
- pct_change FutureWarning: benign; will be updated to `pct_change(fill_method=None)` progressively.
- Empty test window: use `--strict-test-window` or adjust bounds to ensure data coverage.

