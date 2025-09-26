# RL Intraday Trading System

A comprehensive reinforcement learning-based intraday trading system for futures markets, specifically designed for E-mini S&P 500 (MES) trading with professional-grade risk management, execution, and monitoring capabilities.

## 🎯 Overview

This system implements a sophisticated RL trading agent using PPO-LSTM architecture with triple-barrier labeling, comprehensive feature engineering, and institutional-quality risk management. The system supports both paper trading and live trading through Interactive Brokers (IBKR) integration.

### Key Features

- **Advanced RL Architecture**: PPO-LSTM policy with memory for sequential decision making
- **Professional Risk Management**: Position sizing, drawdown limits, VaR controls, emergency stops
- **Triple-Barrier Exits**: Profit targets, stop losses, and time-based exits
- **Comprehensive Feature Engineering**: Technical indicators, microstructure features, time encoding
- **Walk-Forward Validation**: Rigorous backtesting with temporal splits
- **Real-Time Monitoring**: Live performance dashboards and risk analytics
- **Paper & Live Trading**: Seamless transition from simulation to live markets
- **Extensive Testing**: 99%+ test coverage with unit, integration, and performance tests

## 🏗️ Architecture

```
rl-intraday/
├── src/
│   ├── data/           # Data acquisition and management
│   ├── features/       # Feature engineering pipeline
│   ├── sim/           # Trading simulation environment
│   ├── rl/            # Reinforcement learning components
│   ├── trading/       # Paper and live trading engines
│   ├── monitoring/    # Risk monitoring and analytics
│   ├── evaluation/    # Performance evaluation and reporting
│   └── utils/         # Core utilities and configurations
├── tests/             # Comprehensive test suite
├── configs/           # Configuration files
├── scripts/           # Utility scripts
├── notebooks/         # Research and analysis notebooks
└── models/           # Trained model storage
```
## 📈 US Stocks (Polygon) — Data + Training

This repo now supports high‑throughput US stocks collection for upgraded Polygon subscriptions while keeping the original downloader for other asset classes (forex, futures, indices, options).

Key scripts and usage:
- Collect US stocks (minute aggregates, reference, fundamentals, corporate actions, snapshots):
  - `export POLYGON_API_KEY=YOUR_KEY`
  - `python scripts/collect_polygon_us_stocks.py --tickers BBVA --start-date 2020-01-01 --end-date 2025-09-04 --types aggregates fundamentals reference corp_actions snapshot --concurrency 20`
- Aggregate a single ticker to one Parquet for training:
  - `python scripts/aggregate_us_stock_data.py --ticker BBVA --start-date 2020-01-01 --end-date 2025-09-04`
  - Output: `data/raw/BBVA_1min.parquet`
- Generate features (generic, works for any ticker):
  - `python scripts/generate_spy_features.py --data data/raw/BBVA_1min.parquet --output data/features/BBVA_features.parquet --config configs/settings.yaml`
- Train with ticker‑aware model naming and CPU parallelism:
  - `PYTHONPATH=. python -m src.rl.train --ticker BBVA --config configs/settings.yaml --data data/raw/BBVA_1min.parquet --features data/features/BBVA_features.parquet`
  - Saves model to `models/BBVA_trained_model` and `vecnormalize.pkl` alongside.
- Backtest (continuous mode shown):
  - `python examples/polygon_re_backtest_continous.py --symbol BBVA --start-date 2025-01-01 --end-date 2025-06-30 --model-path models/BBVA_trained_model --features-path data/features/BBVA_features.parquet --continuous --plot`

Performance (CPU‑only):
- Set `train.n_envs: 8`, `train.n_steps: 512–1024`, `train.batch_size: 1024`, `train.device: cpu`, `train.torch_threads: 1` in `configs/settings.yaml`.
- Progress bars are enabled for both training and backtesting.

See `docs/rl_improvement_plan.md` for the staged roadmap.

## 🧪 Stability & Normalization (Multi‑Ticker)

We added a stability pass for multi‑ticker PPO training:

- PPO tuning: lr=1e-4, batch_size=4096, vf_coef=0.7, ent_coef=0.015, target_kl=0.075, n_steps=2048
- Normalization: `normalize:` block enables VecNormalize (obs+reward); stats are saved with the model
- Data hygiene pre‑features: strict de‑dup by (timestamp,ticker), bounded ffill (≤2 bars), drop tiny islands; aligned masks for train/test
- Evaluation: always emits `trades.csv` and expanded trade stats in `evaluation_results.json`

Quick run with local partitioned data:

```
PYTHONPATH=. venv/bin/python scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --train-start 2024-06-01 --train-end 2024-08-31 \
  --test-start  2024-09-23 --test-end  2024-09-30 \
  --tickers "LYFT RUN SNAP PLUG SQ UBER ROKU ENPH CRWD DDOG OKTA ETSY NET SPY" \
  --test-tickers "GOOGL CHPT UAL DOCU CROX FSLR PINS ZM TWLO RBLX SOFI" \
--output-dir results/local_multi_norm
```

## ✅ Profitability Improvements Applied

- Per‑ticker features: pipeline preserves `ticker` and computes features per name.
- Normalization: VecNormalize enabled for observations and rewards.
- PPO tuning: n_steps=1536, batch_size=3072, ent_coef=0.005, target_kl=0.04, total_timesteps=300k.
- Portfolio controls: min_hold_minutes=10, max_entries_per_day=2, small holding penalty.
- Regime/ticker conditioning: lightweight regime tags and optional ticker identity columns.

### Train/Test (clean OOS split)

Train (Apr → mid‑Aug), Test (mid‑Aug → Sep) on a core, liquid cluster:

```
PYTHONPATH=. python scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --tickers SPY AAPL MSFT NVDA \
  --train-start 2024-04-01 --train-end 2024-08-15 \
  --test-start  2024-08-16 --test-end  2024-09-30 \
  --output-dir results/mt_cluster_core \
  --portfolio-env --skip-download
```

OOS evaluation across rolling windows:

```
PYTHONPATH=. python scripts/oos_eval.py \
  --config configs/settings.yaml \
  --data     results/mt_cluster_core/data/multiticker_data_2024-04-01_to_2024-09-30.parquet \
  --features results/mt_cluster_core/features/multiticker_features_2024-04-01_to_2024-09-30.parquet \
  --model    results/mt_cluster_core/models/model \
  --test-tickers SPY AAPL MSFT NVDA \
  --windows 3 \
  --output results/oos_eval/mt_cluster_core
```

### Optional: External VIX as a Feature

- Polygon (5 calls/min):

```
export POLYGON_API_KEY=...
PYTHONPATH=. python scripts/download_vix_polygon.py --start 2024-01-01 --end 2024-09-30 --symbol I:VIX
```

- Configure Features: `configs/settings.yaml`

```
features:
  volatility:
    external_vix_path: data/external/vix.parquet
  regime:
    enabled: true
  ticker_identity:
    enabled: true
```

The feature pipeline will merge `vix` and `vix_z` (z‑score) lagged by 1 bar to avoid lookahead.

## 📊 Backtest Reports & Trades

When a backtest completes, the following artifacts are written under `<run>/backtest/`:

- summary.json: portfolio summary with total_return, drawdown, Sharpe-like metrics, win_rate, profit_factor.
- portfolio_history.csv: per-bar portfolio equity, open positions, turnover, and per-ticker units.
- daily_report.csv: compact day-level PnL with a proxy daily Sharpe and intraday max drawdown.
- trades.csv: one row per closed trade with enriched fields:
  - ticker, direction, entry_time, exit_time
  - entry_price, exit_price, units, duration_bars, duration_minutes, pnl
  - mfe, mae: max favorable/adverse excursion in PnL terms
  - commission, spread_cost, slippage_cost, impact_cost, total_cost_est
  - return_pct: pnl / |entry_price * units|
  - trade_id, run_seed, window_start, window_end

Flags & tips:
- Use `--strict-test-window` to avoid fallback to a different test slice when the requested window is empty.
- Limit evaluation to tickers with coverage via `--test-tickers`.

## 🧪 PPO Stabilization (Obs/Reward Norm + Schedules)

Stabilization changes are configurable under `rl:` in `configs/settings.yaml` and include:

- VecNormalize (obs+reward) with persistent stats (`checkpoints/vecnorm.pkl`).
- SubprocVecEnv for de-correlated rollouts (`rl.n_envs`).
- Linear schedules for learning rate (3e-4 → 1e-5) and clip range (0.2 → 0.1).
- Evaluation callback saving `best_model.zip` and halving LR on plateau.
- TB logging under `models/logs/tensorboard/` and `metrics/training_summary.json`.

### KL Guards, Adaptive LR, and Live LR Bump

- Healthy ranges during training:
  - approx_kl ≈ 0.003–0.010
  - clip_fraction ≈ 3–8%
  - explained_variance ≥ 0.8 (later in training)
- Callbacks wired into PPO learn:
  - KLStopCallback (early stops an update when KL > target)
  - AdaptiveLRByKL (nudges LR up/down based on KL)
  - LiveLRBump (touch a flag file to multiply LR once without restart)
- Nudge LR mid‑run:

```
./scripts/lr_bump.sh /path/to/run_dir   # creates /path/to/run_dir/.lr_bump
```

## 🧰 Low‑Price ($10–$20) Universe Runner

We include a static low‑price ticker universe and a convenience script to collect, aggregate, train, and backtest a portfolio.

- Universe: `scripts/universe_lowpx_10_20.txt` (edit as needed)
- Runner: `scripts/run_lowpx_portfolio.sh`

Run (requires `POLYGON_API_KEY`):

```
export POLYGON_API_KEY=... \
UNIVERSE_FILE=scripts/universe_lowpx_10_20.txt \
OUT_DIR=results/mt_lowpx_core \
MAX_STEPS=3000000 SEED=123 \
bash scripts/run_lowpx_portfolio.sh
```

Notes:
- Strict test window is enforced by the runner (ingest first to avoid fallback).
- Artifacts: `${OUT_DIR}/models/{checkpoints,logs,metrics}` and `${OUT_DIR}/backtest/{summary.json,portfolio_history.csv,daily_report.csv,trades.csv}`.

Bulk Polygon ingest (partitioned parquet):

```
export POLYGON_API_KEY=... 
PYTHONPATH=. venv/bin/python scripts/polygon_bulk_ingest.py \
  --config configs/settings.yaml \
  --tickers SPY,QQQ \
  --start 2022-01-01 --end 2024-12-31 \
  --cpm 60
```

## 🧠 Feature Engineering Highlights

- ICT: prior‑day levels and equilibrium (PDM), opening‑range (OR) distances, displacement bars (+ density), fair‑value gaps (+ density), equal highs/lows proximity.
- VPA: relative volume (RVOL), climax volume flags, churn (+ z‑score), imbalance persistence, direction EMA, intrabar volatility.
- SMT (intermarket divergence): instrument vs SPY short‑horizon momentum; SPY vs QQQ momentum divergence for regime context.
- VIX term structure: 9D/1M ratio, 1M/3M ratio; base VIX returns/levels normalized to minute cadence.
- Time‑of‑day: cyclic encodings (hour/minute/day), market‑open flags; can be toggled off.
- Pruning: low variance filter and correlation threshold.

Feature generation prints a summary so you can verify VIX/SMT/ICT/VPA coverage.

## ⚙️ Training Settings (CPU‑only optimized)

- Parallel envs: `train.n_envs: 8`
- Rollouts: `n_steps: 1024`, `batch_size: 1024`
- PPO: `gamma: 0.995`, `gae_lambda: 0.97`, `target_kl: 0.03`, `ent_coef: 0.005`
- Device: `train.device: cpu`, `torch_threads: 1`
- Reward (blend): `alpha * DSR + beta * raw_pnl - micro_penalties - risk/drawdown penalties`
  - No‑trade windows (first/last 5m) and widened micro‑penalties during first/last 15m.

## 🧰 Utility Scripts

- `scripts/collect_polygon_us_stocks.py` — upgraded US stocks downloader (parallel, per‑day aggregates, reference, fundamentals, CA, snapshots)
- `scripts/aggregate_us_stock_data.py` — aggregate per‑day minute files to `data/raw/<TICKER>_1min.parquet`
- `scripts/download_vix_data.py` — Yahoo (^VIX, ^VIX9D, ^VIX3M) to `data/external/vix.parquet`
- `scripts/download_vix_term_structure.py` — Polygon indices with fallback (may be restricted)
- `scripts/ingest_external_vix.py` — ingest Databento/FRED/CBOE VIX files into unified parquet
- `scripts/prepare_context_with_fallback.py` — orchestrates SPY/QQQ/VIX context with fallbacks
- `scripts/audit_data_inventory.py` — inventory report of available data, features, and models
- `scripts/audit_collect.py` — read‑only audit of a run directory; writes JSON + Markdown reports
- `scripts/audit_collect.sh` — wrapper to run the audit easily

## 🧾 Audit

Quickly gather all relevant facts about a run to judge stability and correctness.

Run:

```
./scripts/audit_collect.sh results/mt_lowpx_core configs/settings.yaml
```

Outputs are written to `results/audits/<run_basename>_<timestamp>/`:

- `audit_report.json` — structured facts (data/splits, features, normalization, execution, PPO params, callbacks, backtest metrics, derived heuristics)
- `audit_report.md` — human‑readable Markdown summary

Paste `audit_report.json` in review threads for fast triage.

## 🔍 Feature Screening (Leak‑Proof)

Screen features OOS using purged, embargoed, expanding walk‑forward CV and cheap baselines (Ridge, RF). Labels are net‑of‑cost returns over a horizon.

Run:

```
PYTHONPATH=. python scripts/feature_screen.py \
  --config configs/settings.yaml \
  --run-name feature_screen_mt_core \
  --horizon-min 10 \
  --embargo-min 15 \
  --n_slices 6 \
  --rf-n 200 \
  --rf-depth 7 \
  --ridge-alpha 5.0 \
  --seed 123
```

Outputs under `results/features/<run_name>/`:

- `screen_config.json` — parameters used
- `importance_slice<N>_ridge.parquet`, `importance_slice<N>_rf.parquet` — OOS permutation importance per slice/model
- `consensus_importance.parquet` — aggregated consensus score across slices/models
- `summary.md` — human‑readable recap with top 50 features

Notes:
- Uses the repo’s FeaturePipeline (no duplication). Normalization is fit on TRAIN folds only by virtue of the split and re‑fit per slice.
- Costs/slippage are read from `configs/settings.yaml` execution block to compute net‑of‑cost labels.

## 🚀 Quick Commands (Makefile + Scripts)

Use the provided Makefile targets or call the scripts directly.

- Generate features (example for SPY):
  - `make gen_features DATA=data/cache/SPY_20240101_20240701_ohlcv_1min.parquet FEATURES=results/oos_eval_demo/spy_features.parquet`
- Train a small RL model:
  - `make rl_train DATA=data/cache/SPY_20240101_20240701_ohlcv_1min.parquet FEATURES=results/oos_eval_demo/spy_features.parquet MODEL=results/oos_eval_demo/model`
- OOS evaluation (held-out windows/tickers):
  - `make backtest_oos DATA=data/cache/SPY_20240101_20240701_ohlcv_1min.parquet FEATURES=results/oos_eval_demo/spy_features.parquet MODEL=results/oos_eval_demo/model OOS_OUT=results/oos_eval_demo TICKERS="SPY"`
- Small tuning sweep over PPO hyperparams:
  - `make tune DATA=data/cache/SPY_20240101_20240701_ohlcv_1min.parquet FEATURES=results/oos_eval_demo/spy_features.parquet`

Direct scripts (if you prefer):
- OOS evaluator: `PYTHONPATH=. venv/bin/python scripts/oos_eval.py --config configs/settings.yaml --data <parquet> --features <parquet> --model <sb3_model> --test-tickers SPY AAPL --output results/oos_eval_demo`
- Tuning orchestrator: `PYTHONPATH=. venv/bin/python scripts/tune_orchestrator.py --config configs/settings.yaml --data <parquet> --features <parquet> --output results/tuning_run --test-tickers SPY --trials 4 --total-steps 5000`

See `docs/FULL_RUNBOOK.md` for an end‑to‑end sequence and `docs/DRY_RUNS.md` for lightweight sanity runs.

## 🕒 DatetimeIndex Fix and Data Pipeline Enhancements

### Issue Summary
The system encountered critical issues with timestamp handling and DatetimeIndex management that affected data consistency across the entire trading pipeline:

1. **Inconsistent Timestamp Detection**: Different data sources used varying column names for timestamps ("timestamp", "datetime", "time", "dt", "ts")
2. **Timezone Handling Problems**: Mixed timezone-aware and timezone-naive data caused errors in feature engineering and time-based calculations
3. **DatetimeIndex Setup Issues**: Improper index configuration led to sorting problems and duplicate handling failures
4. **Data Pipeline Inconsistencies**: Downstream components (features, RL environment, risk management) received inconsistent timestamp formats

### Fix Implementation

#### 1. Timestamp Column Detection Helper
**File**: `src/data/data_loader.py`
```python
def _detect_ts_col(df):
    for c in ("timestamp", "datetime", "time", "dt", "ts"):
        if c in df.columns:
            return c
    return None
```
- Automatically detects timestamp columns by checking common naming patterns
- Ensures consistent column identification across different data sources

#### 2. DataFrame Postprocessing Method
**File**: `src/data/data_loader.py`
```python
def _postprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
    # Detect timestamp column
    ts_col = _detect_ts_col(df)
    
    # Convert to UTC-aware datetime
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.loc[ts.notna()].copy()
    df[ts_col] = ts
    
    # Set DatetimeIndex with proper timezone conversion
    df = df.sort_values(ts_col)
    df = df.set_index(ts_col)
    try:
        df.index = df.index.tz_convert("America/New_York")
    except Exception:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]
    
    return df
```

#### 3. Timezone Handling Strategy
- **Input Processing**: All timestamps converted to UTC-aware format first
- **Market Time Conversion**: Converted to America/New_York timezone for consistency with market hours
- **Error Handling**: Graceful handling of timezone-naive data with fallback localization
- **Validation**: Ensures all data entering the pipeline has consistent timezone information

### Impact on Data Pipeline

#### Feature Engineering
- **Time Features**: Consistent DatetimeIndex enables reliable time-of-day, day-of-week, and session features
- **Microstructure Features**: Proper timestamp ordering ensures accurate spread and imbalance calculations
- **Technical Indicators**: Sorted data prevents calculation errors in moving averages and other indicators

#### RL Environment
- **Temporal Consistency**: Proper DatetimeIndex ensures correct episode timing and market session handling
- **Feature Alignment**: Consistent timestamps align OHLCV data with engineered features
- **Walk-Forward Validation**: Reliable temporal splits for robust backtesting

#### Risk Management
- **Time-Based Calculations**: Accurate timestamp handling for position sizing and risk limits
- **Market Hours Validation**: Proper timezone conversion ensures correct trading hours identification
- **Performance Attribution**: Consistent time indexing enables accurate P&L attribution

### Technical Benefits
- **Data Integrity**: Eliminates timestamp-related data corruption
- **Performance**: Efficient duplicate removal and sorting operations
- **Scalability**: Consistent data format across different data sources
- **Maintainability**: Centralized timestamp handling logic
- **Robustness**: Comprehensive error handling for edge cases

### Usage Examples
```python
from src.data.data_loader import UnifiedDataLoader

# Load data with automatic DatetimeIndex setup
loader = UnifiedDataLoader(data_source="polygon")
data = loader.load("SPY", "2024-01-01", "2024-01-31")

# Data now has:
# - Proper DatetimeIndex in America/New_York timezone
# - No duplicate timestamps
# - Consistent column naming
# - Ready for feature engineering pipeline
```

This fix ensures that all data flowing through the RL trading system maintains temporal consistency and proper timezone handling, which is critical for reliable trading signals and risk management.


## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd rl-intraday

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Configuration

1. **Copy and customize settings:**
```bash
cp configs/settings.yaml configs/my_settings.yaml
# Edit my_settings.yaml with your parameters
```

2. **Configure data sources:**
```yaml
data:
  databento:
    api_key: "your_databento_key"
  ibkr:
    host: "127.0.0.1"
    port: 7497
    client_id: 1
```

### Basic Usage

#### 1. Data Preparation
```python
from src.data.databento_client import DatabentoClient

client = DatabentoClient(api_key="your_key")
data = client.fetch_historical_data(
    dataset="GLBX.MDP3",
    symbols="MES",
    schema="ohlcv-1m",
    start="2023-01-01",
    end="2023-12-31"
)
client.save_to_parquet(data, "data/raw/mes_1min.parquet")
```

#### 2. Feature Engineering
```python
from src.features.pipeline import FeaturePipeline

config = {
    'technical': {
        'sma_windows': [10, 20, 50],
        'rsi_window': 14,
        'atr_window': 14
    },
    'microstructure': {
        'calculate_spread': True,
        'calculate_imbalance': True
    },
    'time': {
        'time_of_day': True,
        'session_features': True
    }
}

pipeline = FeaturePipeline(config)
features = pipeline.transform(market_data)
```

#### 3. RL Training
```python
from src.rl.train import RLTrainer
from src.sim.env_intraday_rl import IntradayRLEnvironment

# Setup environment
env = IntradayRLEnvironment(
    market_data=data,
    config=training_config
)

# Train agent
trainer = RLTrainer(env=env, config=training_config)
trainer.train(num_episodes=10000)
```

#### 4. Paper Trading
```python
from src.trading.paper_trading import PaperTradingEngine

config = {
    'initial_capital': 100000.0,
    'commission_per_trade': 2.5,
    'max_position_size': 5
}

engine = PaperTradingEngine(config=config)
engine.connect()

# Place orders
order_id = engine.place_order(
    symbol='MES',
    action='BUY',
    quantity=1,
    order_type='MARKET'
)
```

#### 5. Live Trading (IBKR)
```python
from src.trading.ibkr_client import IBKRTradingClient

config = {
    'host': '127.0.0.1',
    'port': 7497,
    'client_id': 1,
    'account': 'YOUR_ACCOUNT'
}

client = IBKRTradingClient(config=config)
client.connect()

# Place live orders
order_id = client.place_market_order(
    symbol='MES',
    action='BUY',
    quantity=1
)
```

## 🧪 Testing

The system includes comprehensive testing with 99%+ coverage:

```bash
# Run all tests
python scripts/run_tests.py --all --coverage

# Run specific test categories
python scripts/run_tests.py --unit           # Unit tests only
python scripts/run_tests.py --integration    # Integration tests only
python scripts/run_tests.py --fast          # Fast tests only

# Generate test report
python scripts/run_tests.py --report

# Check code quality
python scripts/run_tests.py --lint          # Linting
python scripts/run_tests.py --security      # Security checks
```

## 📊 Monitoring

### Real-Time Dashboard
```python
# Start monitoring dashboard
python scripts/run_monitoring_dashboard.py --config configs/settings.yaml
```

Access the dashboard at `http://localhost:8050`

### Risk Monitoring
```python
from src.monitoring.risk_monitor import RiskMonitor

monitor = RiskMonitor(config=risk_config)
risk_status = monitor.assess_portfolio_risk(portfolio_state)

if risk_status['emergency_controls']['flatten_all_positions']:
    # Emergency position flattening triggered
    pass
```

## 🎛️ Configuration

### Core Settings (`configs/settings.yaml`)
```yaml
project: "rl-intraday"
seed: 42

data:
  instrument: "MES"
  minute_file: "data/raw/mes_1min.parquet"
  session:
    tz: "America/New_York"
    rth_start: "09:30"
    rth_end: "16:00"

rl:
  policy:
    obs_dim: 50
    action_dim: 3  # [hold, buy, sell]
    hidden_dim: 128
    sequence_length: 20
  training:
    learning_rate: 3e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    num_epochs: 4
    batch_size: 64

simulation:
  initial_cash: 100000.0
  max_position_size: 5
  transaction_cost: 2.5
  slippage: 0.01
  triple_barrier:
    profit_target: 0.01
    stop_loss: 0.005
    time_limit: 60

risk:
  max_position_size: 5
  max_daily_loss: -5000
  max_drawdown: -0.10
  var_limit: -2000
  leverage_limit: 2.0
```

### Instrument Configuration (`configs/instruments.yaml`)
```yaml
instruments:
  MES:
    description: "E-mini S&P 500 Futures"
    exchange: "CME"
    currency: "USD"
    tick_size: 0.25
    tick_value: 1.25
    margin_requirement: 1320
    trading_hours:
      start: "17:00"
      end: "16:00"
    contract_months: ["MAR", "JUN", "SEP", "DEC"]
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Trading Metrics
- **Returns**: Total, annual, risk-adjusted returns
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown**: Maximum drawdown, recovery time
- **Trade Statistics**: Win rate, profit factor, average win/loss
- **Risk Measures**: VaR, CVaR, volatility

### Example Output
```
=== Performance Summary ===
Total Return: 15.8%
Annual Return: 12.4%
Sharpe Ratio: 1.85
Sortino Ratio: 2.12
Calmar Ratio: 1.34
Max Drawdown: -3.2%
Win Rate: 58.3%
Profit Factor: 1.67
Total Trades: 1,247
```

## 🛡️ Risk Management

### Position Sizing
- Maximum position size limits
- Dynamic position sizing based on volatility
- Concentration limits per instrument

### Risk Limits
- Daily loss limits with automatic position flattening
- Maximum drawdown limits
- VaR-based position sizing
- Leverage constraints

### Emergency Controls
- Automatic position flattening on breach of limits
- Trading halt mechanisms
- Real-time risk monitoring and alerts

## 🔄 Walk-Forward Validation

Rigorous backtesting with temporal awareness:

```python
from src.rl.walkforward import WalkForwardValidator

validator = WalkForwardValidator(
    data=historical_data,
    config={
        'train_window_months': 6,
        'test_window_months': 1,
        'step_size_months': 1
    }
)

results = validator.run_validation()
print(f"Out-of-sample Sharpe: {results['summary_stats']['mean_test_sharpe']:.2f}")
```

## 📚 Research & Development

### Jupyter Notebooks
- `notebooks/data_exploration.ipynb`: Data analysis and feature exploration
- `notebooks/strategy_research.ipynb`: Strategy development and backtesting
- `notebooks/model_analysis.ipynb`: Model performance analysis

### Feature Engineering Research
- Technical indicator optimization
- Microstructure feature development
- Alternative data integration
- Feature selection and dimensionality reduction

## 🚀 Production Deployment

### Paper Trading
```bash
# Start paper trading
python scripts/run_paper_trading.py --config configs/paper_trading.yaml
```

### Live Trading Checklist
1. ✅ Extensive backtesting completed
2. ✅ Paper trading validation successful
3. ✅ Risk management parameters configured
4. ✅ IBKR connection established and tested
5. ✅ Monitoring dashboard operational
6. ✅ Emergency stop procedures documented

### Monitoring & Alerting
- Real-time P&L monitoring
- Risk limit breach alerts
- System health monitoring
- Trade execution monitoring

## 📋 System Requirements

### Software Requirements
- Python 3.8+
- PyTorch 1.9+
- Interactive Brokers TWS/Gateway
- Redis (for caching)
- PostgreSQL (optional, for trade storage)

### Hardware Requirements
- **Minimum**: 8GB RAM, 4 CPU cores, 50GB storage
- **Recommended**: 16GB RAM, 8 CPU cores, 100GB SSD storage
- **Production**: 32GB RAM, 16 CPU cores, 500GB NVMe storage

### Network Requirements
- Stable internet connection (low latency preferred)
- Direct market data feeds (Databento, IBKR)
- VPN access for secure trading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- All code must have corresponding tests
- Maintain 95%+ test coverage
- Follow PEP 8 style guidelines
- Document all public APIs
- Run full test suite before committing

## 🧺 Portfolio Environment (Multi‑Ticker)

A minimal portfolio RL environment with per‑ticker actions and ATR‑based unit sizing is available at `src/sim/portfolio_env.py`.

Quick start:
```python
from src.sim.portfolio_env import PortfolioRLEnv, PortfolioEnvConfig

# Prepare aligned OHLCV/feature maps keyed by ticker (minute bars, America/New_York tz)
ohlcv_map = {
    'SPY': spy_df,  # columns: open, high, low, close, volume, vwap (optional)
    'QQQ': qqq_df,
}
features_map = {
    'SPY': spy_features,  # numeric columns only
    'QQQ': qqq_features,
}

env = PortfolioRLEnv(
    ohlcv_map=ohlcv_map,
    features_map=features_map,
    env_cfg=PortfolioEnvConfig(
        units_per_ticker=100,
        risk_budget_per_ticker=1000.0,  # used with ATR for unit sizing
        max_gross_exposure=1.0,         # cap as fraction of equity
        turnover_penalty=0.0,
        position_holding_penalty=0.0,
        fixed_tickers=['SPY', 'QQQ'],   # fixes obs/action order
    )
)

obs, _ = env.reset()
action = [1, 2]  # MultiDiscrete per‑ticker: 0→short, 1→flat, 2→long
obs, reward, terminated, truncated, info = env.step(action)
```

Notes:
- Per‑ticker ATR determines unit magnitude: units ≈ risk_budget / (ATR * point_value), capped by `units_per_ticker`.
- Gross exposure is capped by `max_gross_exposure * equity`; units are scaled down if exceeded.
- EOD flatten: positions are closed at session boundaries.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. The authors and contributors are not responsible for any financial losses incurred through the use of this software.**

## 📞 Support

- **Documentation**: See `docs/` directory for detailed documentation
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join discussions in GitHub Discussions
- **Email**: [contact@example.com](mailto:contact@example.com)

## 🙏 Acknowledgments

- Interactive Brokers for market access APIs
- Databento for high-quality market data
- PyTorch team for deep learning framework
- OpenAI Gym for reinforcement learning environment standards
- The quantitative finance community for research and insights

---

**Built with ❤️ by the RL Trading Team**
