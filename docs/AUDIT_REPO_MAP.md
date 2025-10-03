# Repository Audit Map

## Training Entrypoints
- Single-asset: `src/rl/train.py` parses data/feature paths and builds SB3 `RecurrentPPO`.
```
# src/rl/train.py (argparse)
--config configs/settings.yaml
--data <OHLCV.parquet> --features <features.parquet>
--ticker <optional> --walkforward --total-steps
```
- Multi-ticker pipeline: `scripts/run_multiticker_pipeline.py` orchestrates download → features → train → backtest with `--tickers ...` and optional `--test-tickers ...`.
```
# scripts/run_multiticker_pipeline.py (args)
parser.add_argument('--tickers', nargs='+', default=[...])
parser.add_argument('--test-tickers', nargs='+', default=None)
```

## RL Loop & Model
- Policy/Algo: SB3 `RecurrentPPO('MlpLstmPolicy', vec_env, ...)` configured in `src/rl/train.py`.
- Vec env: `DummyVecEnv`/`SubprocVecEnv` with optional `VecNormalize`.
- Optional callbacks include `EntropyAnnealCallback` and custom logging.

## Feature Pipeline
- `src/features/pipeline.py` detects source (Polygon/Databento), maps columns, engineers technical/microstructure/time features, and supports feature selection.
```
# COLUMN_MAPPINGS[...] and _map_columns(...)
if 'timestamp' in mapped_data.columns:
    mapped_data.set_index('timestamp', inplace=True)
```

## Simulation Envs
- Single-asset: `src/sim/env_intraday_rl.py` (`IntradayRLEnv`).
- Portfolio: `src/sim/portfolio_env.py` (`PortfolioRLEnv`) aggregates per-ticker streams; supports `allowed_trade_tickers` and `fixed_tickers`.

## Backtesting & Evaluation
- Walk-forward CLI: `scripts/walkforward_train_eval.py` (wraps `walk_forward_training`).
- Evaluator utilities: `src/evaluation/backtest_evaluator.py`, `src/rl/evaluate.py`.

## Configuration
- Primary: `configs/settings.yaml` (data.multiticker, features, training, walkforward, hpo, monitoring).
- Quick presets: `configs/quick_multiticker*.yaml`, `configs/active_trader.yaml`.
- Instruments metadata: `configs/instruments.yaml`.

## Tests & Markers
- Tests in `tests/` and `bin/` utilities; markers configured in `pytest.ini` (`unit`, `integration`, `slow`, `data`, `ibkr`).

## Where Tickers Come From
- Single-asset path: the dataset given via `--data/--features` implies the ticker; `--ticker` only names outputs.
- Multi-ticker path: training tickers from CLI `--tickers`; backtest tickers via `--test-tickers` or default subset of training tickers.
