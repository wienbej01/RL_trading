# RL Intraday Trading System

Comprehensive multi-ticker intraday trading platform built around a recurrent PPO agent, strict risk controls, and production-grade data pipelines. This README consolidates the current implementation so that new contributors and operators can understand the system without digging through historical patch plans.

---

## 1. Platform Architecture & Data Flow

### 1.1 High-Level Topology
```
[GCS / Polygon] ──► Data Loader ──► Feature Pipeline ──► RL Env/Sim ──► PPO-LSTM Trainer
                                            │                         │
                                            └────► Backtest & WF ─────┘
                                                    │
                                           Reporting & Broker IO
```

### 1.2 Data Sources & Loading
- **Primary store:** Google Cloud Storage bucket `jwss_data_store` (prefix `stocks/<SYMBOL>/<YEAR>/<SYMBOL>_<YEAR>-<MONTH>.parquet`).
- **Fallback:** Legacy Polygon partition layout under `data/polygon/historical/`.
- **Loader:** `src/data/data_loader.py`
  - Autodetects `data.source.provider` (`configs/settings.yaml`).
  - Validates column headers (`open/high/low/close/volume`) and canonicalises timestamps to `America/New_York`.
  - Optional RTH resampling, sparse-day pruning, caching (`data/cache`).
- **Artifacts:** Raw parquet (per symbol), merged training sets in `results/<run>/data/*.parquet`.

### 1.3 Feature Engineering
- Pipeline: `src/features/pipeline.py` with modular packs
  - Technical (SMA/EMA/ATR/ADX/Bollinger), microstructure (OFI, bid/ask proxies), volatility, regime labelling, ticker identity.
  - External signals: optional VIX merge (`features.volatility.external_vix_path`).
  - Data hygiene prior to feature calc: dedupe `(timestamp,ticker)`, forward-fill ≤2 bars, drop isolated islands (<5 bars).
- Outputs: `results/<run>/features/multiticker_features_<train>_to_<test>.parquet` plus pack summary reports.

### 1.4 Simulation & Evaluation Surfaces
- **IntradayRLEnv (`src/sim/env_intraday_rl.py`):** single-ticker environment used inside portfolio wrapper when `len(tickers)==1`. Adds position/unrealized P&L channels.
- **PortfolioRLEnv (`src/sim/portfolio_env.py`):** fixed-universe multi-ticker env with unit positions, intraday discipline (min/max hold, entries/day caps), allowed ticker gating.
- **Backtest / Evaluation:** `src/rl/train.py::evaluate_model` and `src/rl/multiticker_trainer.py::backtest` emit:
  - `summary.json`, `portfolio_history.csv`, `daily_report.csv`, `trades.csv`.
  - Normalizes obs via saved `VecNormalize` stats when present.
- **Walk-Forward & CPCV:** `scripts/run_wf.py` handles embargoed sliding windows, feature screening, and per-window PPO training.

### 1.5 Reporting & Monitoring
- JSON reports stored in `reports/` (see `scripts/generate_performance_report.py`).
- Optional Slack hooks (`configs/settings.yaml:notifications.slack`).
- TensorBoard logs under `models/logs/tensorboard/` during training.

### 1.6 Broker / Execution Integration
- **Execution engine:** `src/sim/execution.py` with configurable commission, slippage, impact, spread parameters.
- **Broker interface:** `src/trading/paper_trading.py` (Interactive Brokers via IB-insync), with mirrored risk checks.
- **Live migration:** Gate behind IBKR credentials in `.env`; paper trading only by default.

---

## 2. Trading Methodology & Risk Controls

### 2.1 Feature Set & Signal Inputs
- Technical trend / momentum (SMA, EMA, MACD, RSI, Bollinger, ADX).
- Volatility and regime tags (ATR, historical volatility, VIX-z, session tags).
- Microstructure (order-flow imbalance proxies, volume delta, queue imbalance).
- Contextual / cross-sectional packs (liquidity, sector classification) configurable via `features.packs`.

### 2.2 Policy Constraints (Rulebook)
- Symmetric long/short entries, single unit exposure per ticker, no pyramiding.
- Entry thresholds derived from policy logits; soft gating via reward penalties.
- Triple-barrier exits: stop = 1R, take-profit = 2R, timeout ≈ 30 minutes.
- RTH-only trading, skip first/last N minutes, VIX/liquidity filters (configurable).

### 2.3 Risk Management
- Per-trade risk fraction (`risk.risk_per_trade_frac`, default 2%).
- Portfolio limits: `env.portfolio.max_entries_per_day`, `min_hold_minutes`, kill-switch on daily drawdown (`risk.max_daily_loss_r`).
- Cost modelling baked into environment reward (commission, spread, slippage, impact).
- Day-end flatten enforcement and inventory penalties.

### 2.4 Reward System (Composite Intraday Objective)
The default reward is a weighted composite configured in `env.reward`:

| Component     | Description                                                      | Config Key            | Default Weight |
|---------------|------------------------------------------------------------------|-----------------------|----------------|
| Return (PnL)  | Realized+unrealized gain per bar, inclusive of transaction costs | `w_ret`               | 1.0            |
| Turnover Pen. | Penalises excessive flips / churn                                | `w_turnover`          | 0.2            |
| Inventory Pen.| Penalises holding inventory to encourage flat end-of-day         | `w_inv`               | 0.05           |
| DSR Component | Differential Sharpe proxy promoting smooth equity curve          | `w_dsr`               | 0.0 (opt-in)   |

Additional guardrails:
- Optional reward scaling (`env.reward_scaling`) before clipping.
- Activity shaping (force-open epsilon, warmup fraction) to ensure exploration.
- Support for hybrid & Sharpe-based rewards retained for experimentation (`docs/design/advanced_hybrid2_reward.md`).

---

## 3. Model Specification & Training Regime

### 3.1 Core Algorithm
- **Agent:** Recurrent PPO (sb3-contrib `RecurrentPPO`).
- **Policy:** Two-layer MLP heads (256 units) feeding 256-sized LSTM; orthogonal init.
- **Vectorisation:** `rl.n_envs` parallel environments (SubprocVecEnv for >1) with optional VecNormalize (`rl.vecnormalize`).

### 3.2 Hyperparameters (current defaults)
| Parameter             | Value (core) | Notes |
|-----------------------|--------------|-------|
| `ppo.n_steps`         | 2048         | rollout horizon per env |
| `ppo.batch_size`      | 2048         | larger (3072) in legacy block for stability |
| `ppo.gamma`           | 0.99         | discount |
| `ppo.gae_lambda`      | 0.95         | GAE smoothing |
| `ppo.ent_coef`        | 0.015        | entropy bonus; scheduled (0.02 → 0.005) |
| `ppo.vf_coef`         | 0.7          | critic loss weight |
| `ppo.target_kl`       | 0.01         | early stop guard; adaptive callbacks halving LR |
| `rl.ppo.lr_schedule`  | cosine       | 1.5e-4 → 1e-5 |
| `normalize.reward_scale` | 0.5       | ensures numeric stability |

Training callbacks (`src/rl/callbacks.py`): early-stop on high KL, adaptive LR bump, rolling evaluation with best-model checkpoints and TensorBoard logging.

### 3.3 Observation / Reward Normalization
- VecNormalize (obs + reward) persisted to `models/checkpoints/vecnorm.pkl`.
- Evaluation reloads stats and disables reward updates to ensure parity with training.

### 3.4 Data Splits & Testing
- `scripts/run_multiticker_pipeline.py`: orchestrates download (GCS), feature generation, training, backtest.
- `scripts/run_wf.py`: embargoed walk-forward with CPCV feature screening.
- Targets (100k steps smoke): median explained variance ≥0.15, KL ≈ 0.03–0.07.

---

## 4. User Guide

### 4.1 Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
export GCS_PROJECT=llm-behavior-trading  # if not set in config
auth application-default login           # ensure GCS credentials available
```
Optional: set `POLYGON_API_KEY` if using polygon fallback scripts.

### 4.2 Quick Start: Multi-Ticker Pipeline
```bash
PYTHONPATH=. python -X faulthandler scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --train-start 2024-06-01 --train-end 2024-07-31 \
  --test-start  2024-08-01 --test-end  2024-08-31 \
  --tickers TSLA AAPL MSFT \
  --test-tickers SPY \
  --output-dir results/tsla_aapl_msft_train_spy_test
```
Outputs include `data/`, `features/`, `models/`, `backtest/`, and a `pipeline_config.json` snapshot.

### 4.3 Walk-Forward & Feature Screening
```bash
PYTHONPATH=. python scripts/run_wf.py \
  --config configs/settings.yaml \
  --run-name wf_core \
  --tickers AAPL MSFT \
  --train-start 2024-01-01 --train-end 2024-04-30 \
  --wf-train-days 60 --wf-valid-days 10 --wf-test-days 10 --wf-step-days 10 \
  --feature-pack curated --timesteps 100000
```
Use `--dry-run` to list windows, `--no-cache` to force feature recomputation.

### 4.4 Performance Reporting
```bash
PYTHONPATH=. python scripts/generate_performance_report.py \
  --model results/tsla_aapl_msft_train_spy_test/models/checkpoints/model_last.zip \
  --data  results/tsla_aapl_msft_train_spy_test/data/multiticker_data_2024-06-01_to_2024-08-31.parquet \
  --features results/tsla_aapl_msft_train_spy_test/features/multiticker_features_2024-06-01_to_2024-08-31.parquet \
  --start 2024-08-01 --end 2024-08-31 \
  --out reports/tsla_aapl_msft_august_report.json
```
This aligns features to the training schema, reloads VecNormalize, and produces JSON summaries for downstream analytics.

### 4.5 Paper Trading (IBKR)
```bash
export IBKR_USERNAME=...
export IBKR_PASSWORD=...
PYTHONPATH=. python scripts/run_paper_trading.py --config configs/settings.yaml --model-path models/latest.zip
```
- Simulates orders via `src/trading/paper_trading.py`.
- Enforces same risk rules as backtest (daily limits, EOD flatten).

### 4.6 Housekeeping Commands
- Remove stale summaries: `find results -name '*summary*' -mtime +2 -delete`
- Clear audits: `rm -rf results/audits/*`
- Update dependencies: `pip install -r requirements.txt && pip install -e .[dev]`

---

## 5. Additional References
- **Technical Overview:** `docs/TECH_OVERVIEW.md`
- **Data Loader & Sources:** `docs/technical_guide/data_sources.md`
- **Trading Rulebook:** `docs/rulebook.md`
- **Reward Design Details:** `docs/design/advanced_hybrid2_reward.md`
- **Training Stability Notes:** `docs/training_stability.md`
- **Walk-Forward Playbook:** `docs/USER_GUIDE.md` & `docs/walkforward/*.md`

For questions or issues, open a ticket referencing the relevant module (data, features, rl, sim, trading).
