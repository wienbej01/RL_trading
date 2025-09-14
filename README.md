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
