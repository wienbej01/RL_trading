# Ticker Source Summary

## Training Tickers
- Single-asset training (`src/rl/train.py`): Defined implicitly by the dataset you pass.
  - Command example:
    - `PYTHONPATH=. python -m src.rl.train --config configs/settings.yaml --data data/raw/SPY_1m.parquet --features data/features/SPY_features.parquet --ticker SPY --total-steps 100000`
  - Notes: `--ticker` only names outputs; no ticker list in config is read here.

- Multi-ticker training (`scripts/run_multiticker_pipeline.py`): Defined explicitly via CLI `--tickers`.
  - Args (excerpt): `parser.add_argument('--tickers', nargs='+', default=[...])`
  - Command example:
    - `python scripts/run_multiticker_pipeline.py --config configs/settings.yaml --tickers AAPL MSFT NVDA --output-dir results/multiticker_pipeline`
  - The pipeline passes these tickers into the loader and trainer; observation shape is fixed using this universe.

## Testing/Backtest Tickers
- Walk-forward CLI (`scripts/walkforward_train_eval.py`): Backtests the same instrument implied by `--data/--features` per fold.
  - Command example:
    - `PYTHONPATH=. python scripts/walkforward_train_eval.py --config configs/settings.yaml --data data/raw/SPY_1m.parquet --features data/features/SPY_features.parquet --output runs/wf_spy --train-days 60 --test-days 20`

- Multi-ticker backtest (`scripts/run_multiticker_pipeline.py`):
  - Preferred: `--test-tickers` to restrict evaluation universe; otherwise a subset of training tickers is used by default.
  - Portfolio env supports `allowed_trade_tickers` and `fixed_tickers` to enforce parity and restrictions.

## Dependency Flow (ASCII)
```
CLI (--tickers / --data+--features)
  → Data Loader (UnifiedDataLoader / parquet)
    → FeaturePipeline (mapping, engineering, selection)
      → Env (IntradayRLEnv or PortfolioRLEnv)
        → Trainer (RecurrentPPO)
          → Evaluators (walk-forward / backtest)
```

## Config Notes
- `configs/settings.yaml` includes curriculum examples with `tickers: ["AAPL"]`, but single-asset training still follows the CLI-provided data paths.
- Multi-ticker pipeline reads CLI `--tickers` into `config['data']['tickers']` before loading.
