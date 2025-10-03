# Module Tree & Roles

## Overview
This document provides a comprehensive overview of the RL trading system's module structure, their roles, and relationships. The system is organized into logical modules that handle specific aspects of the trading pipeline.

## Module Structure

```
src/
├── data/                    # Data loading and preprocessing
│   ├── data_loader.py      # Unified data loader for Polygon/Databento
│   └── ...                 # Additional data utilities
├── features/               # Feature engineering pipeline
│   ├── pipeline.py        # Main feature pipeline
│   ├── technical_indicators.py  # Technical indicators (SMA, EMA, RSI, etc.)
│   ├── microstructure_features.py  # Market microstructure features
│   ├── time_features.py   # Time-based features
│   ├── vpa_features.py    # Volume-Price Analysis features
│   ├── ict_features.py    # ICT (Inner Circle Trader) concepts
│   ├── volatility_features.py  # Volatility modeling
│   ├── smt_features.py    # Smart Money Techniques
│   └── levels_features.py # Support/Resistance levels
├── rl/                     # Reinforcement Learning components
│   ├── train.py           # Main training loop and RL trainer
│   ├── ppo_lstm_policy.py # PPO-LSTM policy implementation
│   ├── callbacks.py       # Training callbacks and utilities
│   └── ...               # Additional RL utilities
├── sim/                    # Simulation and environment
│   ├── env_intraday_rl.py # Main RL trading environment
│   ├── execution.py       # Order execution simulation
│   └── risk.py           # Risk management utilities
├── evaluation/            # Evaluation and backtesting
│   ├── backtest_evaluator.py  # Backtest evaluation framework
│   ├── metrics.py        # Performance metrics calculation
│   └── visualization.py  # Result visualization
├── monitoring/           # System monitoring and logging
│   ├── dashboard.py      # Real-time monitoring dashboard
│   ├── alerts.py         # Alert system
│   └── metrics_logger.py # Metrics logging
├── trading/             # Trading-specific components
│   ├── portfolio.py      # Portfolio management
│   ├── universe.py      # Universe selection
│   └── rules.py         # Trading rules
└── utils/               # Utility functions
    ├── config_loader.py # Configuration management
    ├── logging.py       # Logging utilities
    └── ...             # Additional utilities
```

## Module Roles & Responsibilities

### Data Module (`src/data/`)

**Role**: Data acquisition, loading, and preprocessing

**Key Components**:
- `data_loader.py`: Unified interface for loading market data from multiple sources (Polygon, Databento)
  - Handles timestamp canonicalization
  - Enforces regular trading hours (RTH)
  - Supports data resampling and caching
  - Provides data validation and quality checks

**Relationships**:
- Input: Raw market data files (partitioned by symbol/date)
- Output: Clean, aligned DataFrames for feature engineering
- Dependencies: External data sources, filesystem

### Features Module (`src/features/`)

**Role**: Feature engineering and transformation

**Key Components**:
- `pipeline.py`: Main feature pipeline orchestrator
  - Coordinates feature extraction from multiple categories
  - Handles normalization and feature selection
  - Supports both Polygon and Databento data formats
- `technical_indicators.py`: Technical analysis features
  - Moving averages (SMA, EMA)
  - Momentum indicators (RSI, MACD, Stochastic)
  - Volatility measures (ATR, Bollinger Bands)
- `microstructure_features.py`: Market microstructure features
  - Spread and microprice calculations
  - Order flow imbalance
  - VWAP and TWAP calculations
- `time_features.py`: Time-based features
  - Time of day patterns
  - Day of week effects
  - Session-based features
- `vpa_features.py`: Volume-Price Analysis
- `ict_features.py`: Inner Circle Trader concepts
- `volatility_features.py`: Volatility modeling
- `smt_features.py`: Smart Money Techniques
- `levels_features.py`: Support/Resistance levels

**Relationships**:
- Input: Clean data from data module
- Output: Feature matrices for RL environment
- Dependencies: Technical analysis libraries, statistical functions

### RL Module (`src/rl/`)

**Role**: Reinforcement learning model training and management

**Key Components**:
- `train.py`: Main training orchestrator
  - Implements PPO-LSTM training loop
  - Handles walk-forward optimization
  - Manages model evaluation and saving
- `ppo_lstm_policy.py`: Custom PPO-LSTM policy
  - LSTM-based feature extraction
  - Policy and value networks
  - Multi-ticker support (to be enhanced)
- `callbacks.py`: Training callbacks
  - TensorBoard logging
  - Model checkpointing
  - Early stopping

**Relationships**:
- Input: Feature data, environment instances
- Output: Trained models, training logs
- Dependencies: Stable Baselines3, PyTorch, environment module

### Simulation Module (`src/sim/`)

**Role**: Trading environment simulation and execution

**Key Components**:
- `env_intraday_rl.py`: Main RL trading environment
  - Implements OpenAI Gym interface
  - Handles position management
  - Calculates rewards (multiple types)
  - Manages risk limits
- `execution.py`: Order execution simulation
  - Models market impact
  - Handles slippage and commissions
  - Supports different order types
- `risk.py`: Risk management utilities
  - Position sizing
  - Drawdown control
  - Portfolio-level risk metrics

**Relationships**:
- Input: Feature data, model actions
- Output: Observations, rewards, done signals
- Dependencies: Feature module, risk management

### Evaluation Module (`src/evaluation/`)

**Role**: Model evaluation and performance analysis

**Key Components**:
- `backtest_evaluator.py`: Backtest evaluation framework
  - Runs historical simulations
  - Calculates performance metrics
  - Generates reports
- `metrics.py`: Performance metrics calculation
  - Risk-adjusted returns (Sharpe, Sortino)
  - Drawdown analysis
  - Trade statistics
- `visualization.py`: Result visualization
  - Equity curves
  - Drawdown plots
  - Performance dashboards

**Relationships**:
- Input: Trained models, historical data
- Output: Performance reports, visualizations
- Dependencies: Simulation module, metrics libraries

### Monitoring Module (`src/monitoring/`)

**Role**: Real-time system monitoring and alerting

**Key Components**:
- `dashboard.py`: Real-time monitoring dashboard
  - Live performance metrics
  - System health indicators
  - Interactive visualizations
- `alerts.py`: Alert system
  - Performance degradation detection
  - Risk limit breaches
  - System failure notifications
- `metrics_logger.py`: Metrics logging
  - Time-series metrics storage
  - Performance tracking
  - Audit trail

**Relationships**:
- Input: Live trading data, system metrics
- Output: Dashboard UI, alerts, logs
- Dependencies: Web frameworks, database systems

### Trading Module (`src/trading/`)

**Role**: Trading-specific logic and portfolio management

**Key Components**:
- `portfolio.py`: Portfolio management
  - Multi-ticker position tracking
  - Risk allocation
  - Performance attribution
- `universe.py`: Universe selection
  - Dynamic ticker selection
  - Liquidity filtering
  - Correlation analysis
- `rules.py`: Trading rules
  - Entry/exit conditions
  - Risk management rules
  - Session controls

**Relationships**:
- Input: Market data, model signals
- Output: Portfolio allocations, trading decisions
- Dependencies: Data module, risk management

### Utils Module (`src/utils/`)

**Role**: Utility functions and system configuration

**Key Components**:
- `config_loader.py`: Configuration management
  - YAML configuration loading
  - Environment variable handling
  - Path resolution
- `logging.py`: Logging utilities
  - Structured logging
  - Log rotation
  - Multiple output formats

**Relationships**:
- Input: Configuration files, environment variables
- Output: Configuration objects, loggers
- Dependencies: YAML libraries, logging frameworks

## Data Flow

1. **Data Ingestion**: `data_loader.py` → Raw market data
2. **Feature Engineering**: `pipeline.py` → Feature matrices
3. **Environment**: `env_intraday_rl.py` → Observations/Rewards
4. **Training**: `train.py` + `ppo_lstm_policy.py` → Trained models
5. **Evaluation**: `backtest_evaluator.py` → Performance metrics
6. **Monitoring**: `dashboard.py` → Real-time insights

## Extension Points for Multi-Ticker Support

### Data Module Extensions
- Multi-ticker data synchronization
- Cross-ticker data alignment
- Ticker-specific metadata management

### Features Module Extensions
- Cross-ticker relative features
- Ticker-specific normalization
- Sector-based feature aggregation

### RL Module Extensions
- Multi-ticker PPO-LSTM policy
- Portfolio-level action spaces
- Cross-ticker attention mechanisms

### Simulation Module Extensions
- Multi-ticker environment
- Portfolio-level reward calculation
- Cross-ticker risk management

### Trading Module Extensions
- Dynamic universe selection
- Multi-ticker portfolio optimization
- Correlation-aware position sizing

## Key Design Patterns

1. **Pipeline Pattern**: Feature engineering follows a pipeline pattern with configurable stages
2. **Strategy Pattern**: Multiple reward types and trading rules can be selected at runtime
3. **Observer Pattern**: Callbacks and monitoring use observer pattern for event handling
4. **Factory Pattern**: Environment and model creation use factory patterns for flexibility
5. **Template Method**: Training and evaluation follow template methods with customizable steps

## Configuration Dependencies

Each module relies on configuration from `configs/settings.yaml`:
- Data sources and paths
- Feature engineering parameters
- Environment settings
- Training hyperparameters
- Risk management rules
- Logging and monitoring settings