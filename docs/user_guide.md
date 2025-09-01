# RL Intraday Trading System User Guide

## Overview

The RL Intraday Trading System is a comprehensive reinforcement learning-based trading platform designed for intraday futures trading, specifically optimized for E-mini S&P 500 (MES) contracts. The system integrates advanced RL algorithms, professional risk management, and real-time monitoring capabilities.

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 50GB free space (100GB recommended for data)
- **Network**: Stable internet connection for market data

### Software Dependencies

Install the required packages using the provided requirements file:

```bash
cd rl-intraday
pip install -r requirements.txt
```

Key dependencies include:
- `pandas>=2.2.2` - Data manipulation
- `numpy>=1.26.4` - Numerical computing
- `torch>=2.1` - Deep learning framework
- `stable-baselines3>=2.3.2` - RL algorithms
- `gymnasium>=0.29.1` - RL environment
- `ib-insync>=0.9.86` - Interactive Brokers integration
- `polygon-api-client>=1.0.0` - Polygon.io data API

### API Keys and Credentials

#### Polygon.io API (Required for data collection)
```bash
export POLYGON_API_KEY="your_polygon_api_key_here"
```

Get your API key from [Polygon.io](https://polygon.io/)

#### Databento API (Alternative data source)
```bash
export DATABENTO_API_KEY="your_databento_api_key_here"
```

#### Interactive Brokers (Required for live/paper trading)
- TWS (Trader Workstation) or IB Gateway installed and running
- IBKR account credentials configured in `configs/settings.yaml`

### Data Requirements

The system supports multiple data sources:
- **Polygon.io**: High-frequency data with millisecond precision
- **Databento**: Alternative data provider with comprehensive market data
- **Interactive Brokers**: Real-time market data and execution

## Ticker Selection and Period Setup

### Supported Instruments

The system is pre-configured for major futures contracts. Edit `configs/instruments.yaml` to add or modify instruments:

```yaml
MES:
  exchange: CME
  symbol: MES
  tick_size: 0.25
  point_value: 5.0
  tick_value: 1.25
  rth_tz: America/New_York
  rth_start: "09:30"
  rth_end: "16:00"
  contract_multiplier: 5.0
  min_tick: 0.25
  currency: USD
```

### Trading Hours Configuration

Configure regular trading hours in `configs/settings.yaml`:

```yaml
data:
  session:
    tz: America/New_York
    rth_start: "09:30"
    rth_end: "16:00"
```

### Data Collection Period

Set the data collection parameters:

```yaml
data:
  lookback_bars: 120  # ~2 hours context for LSTM
  resample: "1min"     # 1-minute bars
  drop_na_policy: "forward_fill"
```

### Advanced Period Configuration

For custom timeframes and extended hours trading:

```yaml
data:
  provider: polygon  # 'polygon', 'databento', or 'auto'
  polygon_timeout: 30
  polygon:
    rate_limit:
      calls_per_minute: 5
      burst_limit: 10
```

## Backtest Execution

### Data Preparation

1. **Collect Historical Data**:
```bash
# Using Polygon data
python scripts/collect_polygon_data.py \
  --symbols SPY,QQQ,AAPL,MSFT,TSLA \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --include-quotes

# Using predefined portfolio
python scripts/collect_polygon_data.py --preset pilot
```

2. **Validate Collected Data**:
```bash
python scripts/validate_collected_data.py --data-dir data/polygon/historical
```

### Feature Engineering

Configure feature extraction in `configs/settings.yaml`:

```yaml
features:
  technical:
    returns_horizons: [1, 5, 15]
    atr_window: 14
    range_vol_windows: [30,60,90]
    use_polygon_vwap: true
  microstructure:
    enable: true
    ofi_levels: 1
    microprice: true
    spread_regime: true
  seasonality:
    tod_encoding: true
  regimes:
    vix: true
    macro_flags: true
```

### RL Model Training

#### Basic Training
```bash
python -m src.rl.train \
  --config configs/settings.yaml \
  --data data/polygon/historical/SPY_20240101_20240630_ohlcv_1min.parquet \
  --features data/features/SPY_features.parquet \
  --output models/trained_model
```

#### Walk-Forward Validation
```bash
python -m src.rl.train \
  --config configs/settings.yaml \
  --data data/polygon/historical/SPY_20240101_20240630_ohlcv_1min.parquet \
  --features data/features/SPY_features.parquet \
  --walkforward \
  --wf-output runs/walkforward_results
```

#### Advanced Training Options
```bash
python -m src.rl.train \
  --config configs/settings.yaml \
  --data data/polygon/historical/SPY_20240101_20240630_ohlcv_1min.parquet \
  --features data/features/SPY_features.parquet \
  --output models/trained_model \
  --total-steps 2000000 \
  --learning-rate 0.0003 \
  --batch-size 16384
```

### Backtest Simulation

Run backtest with trained model:

```python
from src.sim.env_intraday_rl import IntradayRLEnv
from src.features.pipeline import FeaturePipeline
from stable_baselines3 import PPO
import pandas as pd

# Load data and features
ohlcv = pd.read_parquet('data/polygon/historical/SPY_20240101_20240630_ohlcv_1min.parquet')
features = pd.read_parquet('data/features/SPY_features.parquet')

# Initialize environment
env = IntradayRLEnv(
    ohlcv=ohlcv,
    features=features,
    cash=100000.0,
    point_value=5.0
)

# Load trained model
model = PPO.load('models/trained_model.zip')

# Run backtest
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

# Get results
equity_curve = env.get_equity_curve()
performance = env.get_performance_metrics()
print(f"Total Return: {performance['total_return']:.2%}")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
```

### Example Backtest Script

Use the provided example script:

```bash
python examples/polygon_rl_backtest_example.py \
  --symbol SPY \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --episodes 10 \
  --plot
```

## Paper Trading

### Configuration Setup

Configure paper trading settings in `configs/settings.yaml`:

```yaml
trading:
  mode: paper
  initial_capital: 100000.0
  max_position_size: 5
  risk_per_trade_frac: 0.02
  stop_loss_r_multiple: 1.0
  take_profit_r_multiple: 1.5
  max_daily_loss_r: 3.0
```

### IBKR Connection Setup

1. **Install and Configure TWS/Gateway**:
   - Download TWS or IB Gateway from Interactive Brokers
   - Enable API connections in TWS settings
   - Set API port (default: 7497 for live, 7496 for paper)

2. **Configure Connection**:
```yaml
ibkr:
  host: "127.0.0.1"
  port: 7496  # Paper trading port
  client_id: 1
  account: "YOUR_PAPER_ACCOUNT_ID"
```

### Running Paper Trading

#### Basic Paper Trading Session
```bash
python scripts/run_paper_trading.py \
  --config configs/settings.yaml \
  --model models/trained_model.zip \
  --symbol MES \
  --capital 100000 \
  --duration 480  # 8 hours
```

#### Advanced Paper Trading Options
```bash
python scripts/run_paper_trading.py \
  --config configs/settings.yaml \
  --model models/trained_model.zip \
  --symbol MES \
  --exchange CME \
  --currency USD \
  --capital 100000 \
  --max-position-size 10 \
  --output paper_trading_results \
  --dry-run
```

### Paper Trading with Custom Risk Parameters

```python
from src.trading.paper_trading import PaperTradingEngine, PaperTradingConfig
from src.utils.config_loader import Settings

# Load configuration
settings = Settings.from_paths('configs/settings.yaml')

# Create paper trading configuration
config = PaperTradingConfig(
    model_path='models/trained_model.zip',
    trading_symbol='MES',
    trading_exchange='CME',
    trading_currency='USD',
    initial_capital=100000.0,
    max_position_size=5,
    risk_per_trade_frac=0.02,
    output_dir='paper_trading_results'
)

# Initialize and run
engine = PaperTradingEngine(settings, config)
await engine.initialize()
await engine.run_trading_session(duration_minutes=480)
```

## Monitoring Dashboard

### Starting the Dashboard

#### Basic Dashboard
```bash
python scripts/run_monitoring_dashboard.py \
  --config configs/settings.yaml \
  --output monitoring_results
```

#### Advanced Dashboard Configuration
```bash
python scripts/run_monitoring_dashboard.py \
  --config configs/settings.yaml \
  --output monitoring_results \
  --duration 3600 \
  --interval 5 \
  --max-alerts 20 \
  --max-trades 100 \
  --enable-real-time \
  --enable-alerts \
  --enable-performance \
  --enable-risk \
  --export-format json \
  --export-interval 300
```

### Dashboard Features

The monitoring dashboard provides:

- **Real-time Equity Tracking**: Live P&L and equity curve visualization
- **Position Monitoring**: Current positions and unrealized P&L
- **Risk Metrics**: VaR, drawdown, and risk limits
- **Trade History**: Recent trades with execution details
- **Performance Analytics**: Sharpe ratio, win rate, profit factor
- **Alert System**: Risk limit breaches and system alerts

### Accessing the Dashboard

Once started, access the dashboard at: `http://localhost:8050`

### Dashboard API

```python
from src.monitoring.dashboard import MonitoringDashboard, DashboardConfig
from src.utils.config_loader import Settings

# Initialize dashboard
settings = Settings.from_paths('configs/settings.yaml')
config = DashboardConfig(
    update_interval=5,
    enable_real_time=True,
    enable_alerts=True,
    enable_performance=True,
    enable_risk=True
)

dashboard = MonitoringDashboard(settings, config)
dashboard.start_dashboard()

# Add real-time data
dashboard.add_real_time_data('equity', {
    'timestamp': datetime.now(),
    'equity': 100500.0,
    'pnl': 500.0
})

# Get dashboard summary
summary = dashboard.get_dashboard_summary()
print(f"Current Equity: ${summary['equity']:,.2f}")
```

## Testing

### Running Test Suite

#### All Tests with Coverage
```bash
python scripts/run_tests.py --all --coverage
```

#### Unit Tests Only
```bash
python scripts/run_tests.py --unit --verbose
```

#### Integration Tests Only
```bash
python scripts/run_tests.py --integration --verbose
```

#### Fast Tests (Excluding Slow/External)
```bash
python scripts/run_tests.py --fast --verbose
```

### Specific Test Execution

#### Run Specific Test File
```bash
python scripts/run_tests.py --specific tests/test_rl_environment.py --verbose
```

#### Run Tests with Profiling
```bash
python scripts/run_tests.py --profile --unit
```

### Code Quality Checks

#### Linting
```bash
python scripts/run_tests.py --lint
```

#### Security Checks
```bash
python scripts/run_tests.py --security
```

#### Generate Test Report
```bash
python scripts/run_tests.py --report
```

This generates:
- HTML coverage report: `htmlcov/index.html`
- XML coverage report: `coverage.xml`
- JUnit test results: `test-results.xml`
- HTML test report: `test-report.html`

### Test Configuration

Tests are configured in `pytest.ini`:

```ini
[tool:pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml"
]
```

### Writing Custom Tests

Example test structure:

```python
import pytest
import pandas as pd
from src.sim.env_intraday_rl import IntradayRLEnv

class TestIntradayRLEnv:
    @pytest.fixture
    def sample_data(self):
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01 09:30:00', periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 100),
            'high': np.random.uniform(4000, 4100, 100),
            'low': np.random.uniform(4000, 4100, 100),
            'close': np.random.uniform(4000, 4100, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data

    def test_environment_initialization(self, sample_data):
        features = pd.DataFrame(index=sample_data.index)
        env = IntradayRLEnv(ohlcv=sample_data, features=features, cash=100000.0)
        assert env.cash == 100000.0
        assert env.pos == 0
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Data Loading Issues

**Problem**: "No data found for symbol"
```bash
# Check data directory structure
ls -la data/polygon/historical/

# Verify API key
echo $POLYGON_API_KEY

# Re-collect data
python scripts/collect_polygon_data.py --symbols SPY --start-date 2024-01-01 --end-date 2024-01-31
```

**Problem**: Timestamp format errors
```python
# Check data format
df = pd.read_parquet('data/polygon/historical/symbol=SPY/year=2024/month=01/day=01/data.parquet')
print(df.head())
print(df.index)
```

#### 2. Model Training Issues

**Problem**: Training doesn't converge
```yaml
# Adjust training parameters in settings.yaml
train:
  learning_rate: 0.0001  # Reduce learning rate
  batch_size: 4096       # Reduce batch size
  n_steps: 1024         # Reduce n_steps
```

**Problem**: Memory errors during training
```bash
# Reduce environment parameters
export CUDA_VISIBLE_DEVICES=0  # Use GPU if available
# Or reduce n_envs in training config
```

#### 3. Paper Trading Connection Issues

**Problem**: IBKR connection fails
```bash
# Check TWS/Gateway is running
ps aux | grep tws

# Verify connection settings
netstat -tlnp | grep 7496

# Test connection manually
python -c "from ib_insync import IB; ib = IB(); ib.connect('127.0.0.1', 7496, clientId=1)"
```

**Problem**: Contract not found
```python
# Check contract details
from src.trading.ibkr_client import IBKRClient
client = IBKRClient(settings)
contract = await client.get_contract('MES', 'CME', 'USD')
print(contract)
```

#### 4. Performance Issues

**Problem**: Slow backtesting
```yaml
# Optimize data loading
data:
  cache_enabled: true
  provider: polygon  # Use faster data source

# Reduce feature complexity
features:
  technical:
    sma_windows: [10, 20]  # Reduce windows
```

**Problem**: Memory usage high
```python
# Use data chunking
from src.data.data_loader import UnifiedDataLoader
loader = UnifiedDataLoader()
# Process data in chunks
for chunk in pd.read_parquet(file, chunksize=10000):
    process_chunk(chunk)
```

#### 5. Feature Engineering Issues

**Problem**: NaN values in features
```python
# Check for NaN values
features = pipeline.transform(data)
print(features.isna().sum())

# Configure fill policy
data:
  drop_na_policy: "forward_fill"
```

**Problem**: Feature scaling issues
```yaml
# Configure normalization
features:
  normalization:
    method: "standardize"  # or "minmax"
    fit_on_train: true
```

### Logging and Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or configure in settings.yaml
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### Check System Health
```bash
# Monitor system resources
top -p $(pgrep -f "python.*rl-intraday")

# Check disk space
df -h data/

# Monitor network connectivity
ping -c 3 api.polygon.io
```

### Performance Optimization

#### Data Pipeline Optimization
```yaml
# Enable caching
data:
  cache_enabled: true
  polygon:
    cache:
      directory: data/cache/polygon
      format: parquet

# Use parallel processing
features:
  parallel_processing: true
  n_jobs: -1
```

#### Model Optimization
```yaml
# Reduce model complexity
rl:
  policy:
    hidden_dim: 64  # Reduce from 128
    sequence_length: 10  # Reduce from 20

# Use mixed precision
train:
  device: "cuda"  # Use GPU
  mixed_precision: true
```

### Getting Help

1. **Check Documentation**: Review this guide and inline code documentation
2. **Run Diagnostics**: Use the test suite to identify issues
3. **Check Logs**: Examine log files in `logs/trading_system.log`
4. **Community Support**: Check GitHub issues for similar problems
5. **Professional Support**: Contact the development team for enterprise support

### Emergency Procedures

#### Stop Trading Immediately
```bash
# Kill all trading processes
pkill -f "python.*paper_trading"
pkill -f "python.*monitoring"

# Manual position closure via TWS
# 1. Open TWS
# 2. Go to Account > Positions
# 3. Close all open positions manually
```

#### Data Recovery
```bash
# Backup current data
cp -r data/ data_backup_$(date +%Y%m%d_%H%M%S)/

# Re-download missing data
python scripts/collect_polygon_data.py --symbols SPY --start-date 2024-01-01 --end-date 2024-12-31
```

#### System Reset
```bash
# Clear cache
rm -rf data/cache/*
rm -rf __pycache__/
rm -rf *.pyc

# Reset environment
pip install -r requirements.txt --force-reinstall
```

---

**Note**: This system is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly in paper trading before deploying live strategies.