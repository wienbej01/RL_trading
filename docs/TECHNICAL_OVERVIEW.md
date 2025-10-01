# Technical Overview

This document explains the RL Intraday Trading System architecture, data flow, training/evaluation stack, execution cost modeling, and the new Walk-Forward + CPCV utilities.

## System Architecture

- Data Layer (`src/data`):
  - `UnifiedDataLoader` loads OHLCV with canonical timestamp handling, optional RTH resample, and caching.
  - Timezone normalization to America/New_York is standard across the stack.

- Features (`src/features`):
  - `FeaturePipeline` builds feature groups (technical, microstructure, time, etc.) while preserving the `ticker` column.
  - Packs (e.g., curated, price_vol, microstructure) provide curated subsets.

- Simulation Environments (`src/sim`):
  - `IntradayRLEnv`: single-ticker environment with triple-barrier exits, risk sizing, EOD flatten, and realistic transaction costs via `ExecutionEngine`.
  - `PortfolioRLEnv`: multi-ticker environment with per-ticker unit sizing, gross exposure cap, turnover/exposure penalties, and per-step history.

- Execution (`src/sim/execution.py`):
  - Commission, spread, slippage, and impact modeled in `ExecutionEngine` (dataclass `ExecParams`).
  - Utility `estimate_tc` computes per-trade transaction costs.

- RL Training (`src/rl`):
  - `MultiTickerRLTrainer` orchestrates PPO-LSTM training (sb3_contrib) across multiple tickers, with callbacks (KL guards, LR adaptation) and VecNormalize compatibility.
  - Backtest exports portfolio/equity metrics, trades, and daily reports.

- Walk-Forward + CPCV (`src/utils/wf_cv.py`, `scripts/run_wf.py`, `scripts/feature_screen.py`):
  - `EmbargoedWalkForward` produces (train, valid, test) windows with minute embargo between segments.
  - `CombinatorialPurgedKFold` yields purged, embargoed train/test splits for leakage-aware feature scoring.
  - `run_cpcv_feature_screen` computes consensus absolute-correlation importance on training data only.

## Data Flow

1. Load OHLCV for requested tickers and outer time bounds via `UnifiedDataLoader`.
2. Generate features via `FeaturePipeline` (aligned and ticker-aware).
3. (Walk-forward): Slice frames per window using `EmbargoedWalkForward`.
4. (Training-only) Run CPCV feature screening; persist `consensus_importance.parquet` and `curated_topN.txt`.
5. Train PPO using `MultiTickerRLTrainer` on the train split with curated features.
6. Backtest on test split; write `summary.json`, `trades.csv`, `portfolio_history.csv`, `daily_report.csv`.

## Execution Costs and Diagnostics

- Environments track cumulative transaction costs at the env level (`tx_costs_total`).
- `PortfolioRLEnv` history includes `turnover`, `gross_exposure`, `exposure_pct` per step.
- `get_action_counts()` provides `long_steps`, `short_steps`, `flat_steps`, and `flips`.

## Metrics

- Portfolio summary includes: total_return, annual_return/volatility, sharpe_ratio, max_drawdown, sortino_ratio, profit_factor.
- Extended metrics (walk-forward aware): turnover, exposure_pct, long/short/flat steps, flips, tx_costs_total/tx_costs_per_trade.

## Extensibility Hooks

- Feature packs and curated lists for quick experimentation.
- Reward mix (configs `reward.mix`) to combine dsr, turnover, inventory, cvar penalties.
- Tuning placeholder (`configs.tuning.optuna_trials`) to integrate Optuna later.

