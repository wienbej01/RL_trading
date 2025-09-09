# Multi-Ticker Configuration Structure

## Overview

This document outlines the comprehensive configuration structure for the multi-ticker RL trading system. The configuration structure is designed to be modular, extensible, and backward-compatible with the existing single-ticker system.

## Configuration Philosophy

### Design Principles
1. **Modularity**: Each major component has its own configuration section
2. **Hierarchy**: Configuration follows a logical hierarchy from general to specific
3. **Extensibility**: Easy to add new parameters and sections
4. **Backward Compatibility**: Existing single-ticker configurations remain valid
5. **Validation**: Built-in validation with sensible defaults
6. **Environment Overrides**: Support for environment-specific overrides

### Configuration Loading
- Configuration is loaded from YAML files
- Environment variables can override specific values
- Configuration can be validated and type-checked
- Default values are provided for all optional parameters

## Top-Level Configuration Structure

```yaml
# Multi-ticker RL trading system configuration
version: "2.0.0"  # Configuration version
description: "Multi-ticker RL trading system with reward overhaul"

# Global settings
global:
  mode: "multi_ticker"  # "single_ticker" or "multi_ticker"
  random_seed: 42
  device: "auto"  # "auto", "cpu", "cuda"
  num_workers: 4
  log_level: "INFO"
  
# Multi-ticker specific settings
multi_ticker:
  enabled: true
  max_tickers: 10
  min_tickers: 3
  portfolio_rebalance_frequency: "1day"
  position_sizing_method: "equal_weight"  # "equal_weight", "risk_parity", "kelly", "custom"
  max_portfolio_exposure: 1.0
  min_position_size: 0.01
  max_position_size: 0.3
  correlation_threshold: 0.7
  diversification_weight: 0.3
  
# Data configuration
data:
  # Data source settings
  source: "polygon"  # "polygon", "databento", "combined"
  tickers:
    - "AAPL"
    - "MSFT"
    - "GOOGL"
    - "AMZN"
    - "META"
    - "TSLA"
    - "NVDA"
    - "JPM"
    - "JNJ"
    - "V"
  
  # Time settings
  timezone: "America/New_York"
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  trading_hours:
    start: "09:30"
    end: "16:00"
    timezone: "America/New_York"
  
  # Data quality settings
  quality_checks:
    enabled: true
    min_data_points: 100
    max_nan_ratio: 0.05
    price_range_check: true
    volume_check: true
    
  # Data caching
  cache:
    enabled: true
    cache_dir: "data/cache"
    max_cache_size: "10GB"
    cache_expiry: "7days"
    
  # Data validation
  validation:
    enabled: true
    check_missing_data: true
    check_outliers: true
    check_anomalies: true
    outlier_method: "zscore"  # "zscore", "iqr", "isolation_forest"
    outlier_threshold: 3.0
    
# Feature engineering configuration
features:
  # Feature groups
  technical:
    enabled: true
    indicators:
      sma:
        windows: [5, 10, 20, 50, 100]
      ema:
        windows: [5, 10, 20, 50, 100]
      rsi:
        windows: [14, 21]
      macd:
        fast_period: 12
        slow_period: 26
        signal_period: 9
      bollinger_bands:
        window: 20
        num_std: 2
      atr:
        window: 14
        
  microstructure:
    enabled: true
    indicators:
      spread:
        enabled: true
      microprice:
        enabled: true
      queue_imbalance:
        enabled: true
      order_flow_imbalance:
        enabled: true
      vwap:
        enabled: true
      twap:
        enabled: true
        window: 5
        
  time_features:
    enabled: true
    features:
      time_of_day:
        enabled: true
        cyclical_encoding: true
      day_of_week:
        enabled: true
        cyclical_encoding: true
      session_features:
        enabled: true
      time_from_open:
        enabled: true
      time_to_close:
        enabled: true
        
  vpa_features:
    enabled: true
    indicators:
      volume_sma:
        windows: [10, 20, 50]
      volume_profile:
        enabled: true
        buckets: 10
      price_volume_trend:
        enabled: true
      on_balance_volume:
        enabled: true
        
  ict_features:
    enabled: true
    concepts:
      fair_value_gap:
        enabled: true
      liquidity_sweep:
        enabled: true
      order_block:
        enabled: true
      market_structure_shift:
        enabled: true
        
  volatility_features:
    enabled: true
    indicators:
      historical_volatility:
        windows: [10, 20, 50]
      garman_klass_volatility:
        enabled: true
        window: 20
      yang_zhang_volatility:
        enabled: true
        window: 20
      garch_volatility:
        enabled: true
        p: 1
        q: 1
        
  smt_features:
    enabled: true
    concepts:
      support_resistance:
        enabled: true
        method: "pivot_points"  # "pivot_points", "fractals", "kmeans"
      market_structure:
        enabled: true
      price_action_patterns:
        enabled: true
        
  levels_features:
    enabled: true
    indicators:
      pivot_points:
        enabled: true
        method: "standard"  # "standard", "fibonacci", "woodie"
      fibonacci_retracement:
        enabled: true
      psychological_levels:
        enabled: true
        round_numbers: true
        whole_numbers: true
        
  # Feature normalization
  normalization:
    method: "rolling"  # "standard", "robust", "rolling", "quantile"
    window: 20
    min_periods: 10
    clip_outliers: true
    clip_threshold: 3.0
    
  # Feature selection
  selection:
    enabled: true
    method: "mutual_info"  # "mutual_info", "f_regression", "rfe", "lasso"
    k_best: 50
    cross_validation: 5
    
  # Feature engineering pipeline
  pipeline:
    parallel_processing: true
    batch_size: 1000
    cache_features: true
    feature_cache_dir: "data/cache/features"
    
# Multi-ticker reward system configuration
multi_ticker_reward:
  enabled: true
  
  # Reward components
  components:
    portfolio_pnl:
      enabled: true
      weight: 1.0
      scaling: linear  # linear, log, sqrt
      
    sharpe_ratio:
      enabled: true
      weight: 0.5
      window: 20
      scaling: linear
      
    sortino_ratio:
      enabled: true
      weight: 0.3
      window: 20
      scaling: linear
      
    diversification:
      enabled: true
      weight: 0.4
      correlation_window: 20
      correlation_method: pearson
      
    concentration_penalty:
      enabled: true
      weight: 0.2
      max_concentration: 0.3
      
    transaction_cost:
      enabled: true
      weight: 0.5
      include_commission: true
      include_slippage: true
      include_market_impact: true
      
    drawdown_penalty:
      enabled: true
      weight: 0.3
      max_drawdown: 0.1
      penalty_factor: 2.0
      
    turnover_penalty:
      enabled: true
      weight: 0.2
      max_turnover: 0.5
      
    risk_adjusted_return:
      enabled: true
      weight: 0.4
      risk_free_rate: 0.02
      risk_aversion: 1.0
      
    regime_aware:
      enabled: true
      weight: 0.3
      regime_weights:
        normal: {pnl: 1.0, sharpe: 0.5, diversification: 0.4}
        volatile: {pnl: 0.5, sharpe: 1.0, diversification: 0.8}
        trending: {pnl: 1.2, sharpe: 0.3, diversification: 0.2}
        ranging: {pnl: 0.3, sharpe: 0.8, diversification: 0.6}
  
  # Cross-ticker correlation settings
  correlation:
    window: 20
    method: pearson  # pearson, spearman, kendall
    min_periods: 10
    update_frequency: 1min
    cache_enabled: true
    diversification_reward:
      enabled: true
      optimal_correlation: -0.2
      reward_scaling: 1.0
    concentration_penalty:
      enabled: true
      max_correlation: 0.7
      penalty_scaling: 1.0
  
  # Risk adjustment settings
  risk_adjustment:
    risk_free_rate: 0.02
    risk_aversion: 1.0
    drawdown_penalty:
      enabled: true
      max_drawdown: 0.1
      penalty_factor: 2.0
      asymmetric_penalty: true
    volatility_penalty:
      enabled: true
      target_volatility: 0.15
      penalty_factor: 1.0
  
  # Regime detection settings
  regime_detection:
    enabled: true
    window: 20
    update_frequency: 5min
    detection_method: volatility_trend  # volatility_trend, markov_switching, ml_based
    thresholds:
      volatility_high: 0.25
      volatility_low: 0.1
      trend_strength_high: 0.7
      trend_strength_low: 0.3
    regime_persistence: 3
  
  # Reward decomposition settings
  decomposition:
    enabled: true
    track_history: true
    history_length: 1000
    analysis_frequency: 1day
    component_analysis:
      enabled: true
      correlation_analysis: true
      contribution_analysis: true
      regime_analysis: true
    reporting:
      enabled: true
      frequency: 1day
      format: json  # json, csv, parquet
  
  # Reward normalization and scaling
  normalization:
    enabled: true
    method: rolling  # standard, robust, rolling, quantile
    window: 20
    min_periods: 10
    clip_rewards: true
    clip_threshold: 3.0
    
  # Reward smoothing
  smoothing:
    enabled: true
    method: exponential  # exponential, moving_average, none
    window: 5
    alpha: 0.2

# Dynamic universe selection configuration
dynamic_universe_selection:
  enabled: true
  
  # Universe parameters
  universe:
    candidate_tickers:
      - AAPL
      - MSFT
      - GOOGL
      - AMZN
      - META
      - TSLA
      - NVDA
      - JPM
      - JNJ
      - V
      - PG
      - UNH
      - HD
      - BAC
      - XOM
      - PFE
      - CSCO
      - ADBE
      - CRM
    min_universe_size: 5
    max_universe_size: 10
    default_universe_size: 8
    rebalance_frequency: 1day
    
  # Selection criteria
  criteria:
    liquidity:
      enabled: true
      weight: 1.0
      threshold: 0.5
      min_volume: 1_000_000
      min_trades: 100
      max_spread: 0.01
      
    volatility:
      enabled: true
      weight: 0.8
      threshold: 0.4
      target_volatility: 0.2
      volatility_window: 20
      volatility_tolerance: 0.1
      
    trend:
      enabled: true
      weight: 0.6
      threshold: 0.3
      trend_window: 20
      min_trend_strength: 0.1
      
    performance:
      enabled: true
      weight: 1.0
      threshold: 0.5
      performance_window: 20
      metrics: [sharpe, sortino]
      metric_weights:
        sharpe: 0.6
        sortino: 0.4
        
    correlation:
      enabled: true
      weight: 0.4
      threshold: 0.3
      max_correlation: 0.7
      correlation_window: 20
      
    sector:
      enabled: true
      weight: 0.3
      threshold: 0.2
      max_sector_concentration: 0.3
      min_sectors: 3
      sector_mapping:
        technology: [AAPL, MSFT, GOOGL, META, NVDA, CSCO, ADBE, CRM]
        healthcare: [JNJ, PFE, UNH]
        financial: [JPM, BAC, V]
        consumer: [PG, HD, AMZN]
        energy: [XOM]
        automotive: [TSLA]
  
  # Regime-aware selection
  regime_aware:
    enabled: true
    regime_criteria_weights:
      normal:
        liquidity: 1.0
        volatility: 0.8
        trend: 0.6
        performance: 1.0
        correlation: 0.4
        sector: 0.3
      volatile:
        liquidity: 1.2
        volatility: 1.0
        trend: 0.3
        performance: 0.8
        correlation: 0.6
        sector: 0.2
      trending:
        liquidity: 0.8
        volatility: 0.6
        trend: 1.2
        performance: 1.0
        correlation: 0.3
        sector: 0.4
      ranging:
        liquidity: 1.0
        volatility: 0.4
        trend: 0.2
        performance: 0.6
        correlation: 0.5
        sector: 0.5
    regime_universe_sizes:
      normal: 8
      volatile: 6
      trending: 10
      ranging: 7
    regime_sector_preferences:
      normal: {technology: 1.0, healthcare: 1.0, financial: 1.0, consumer: 1.0}
      volatile: {technology: 0.8, healthcare: 1.2, financial: 1.0, consumer: 1.0}
      trending: {technology: 1.2, healthcare: 0.8, financial: 1.0, consumer: 1.1}
      ranging: {technology: 0.9, healthcare: 1.1, financial: 1.0, consumer: 1.0}
  
  # Performance tracking
  performance_tracking:
    enabled: true
    lookback_window: 20
    metrics: [sharpe, sortino, calmar, max_drawdown]
    update_frequency: 1day
    performance_decay: 0.95
    
  # Diversity management
  diversity:
    enabled: true
    max_sector_concentration: 0.3
    max_correlation: 0.7
    min_sectors: 3
    correlation_window: 20
    diversity_weight: 0.3
    
  # Stability control
  stability:
    enabled: true
    max_turnover: 0.2
    min_hold_period: 5
    stability_weight: 0.3
    transition_smoothing: true
    
  # Selection process
  selection:
    method: weighted_score  # weighted_score, ranking, threshold, hybrid
    score_aggregation: weighted_sum  # weighted_sum, geometric_mean, harmonic_mean
    normalization_method: min_max  # min_max, z_score, rank
    ranking_method: top_n  # top_n, percentile, threshold
    
  # Monitoring and logging
  monitoring:
    enabled: true
    log_selection_rationale: true
    track_selection_history: true
    history_length: 100
    alert_on_large_changes: true
    change_threshold: 0.3

# RL environment configuration
environment:
  type: "multi_ticker"  # "single_ticker" or "multi_ticker"
  
  # Environment parameters
  parameters:
    initial_cash: 1_000_000
    commission: 0.001
    slippage: 0.0005
    max_position_size: 0.3
    min_position_size: 0.01
    max_portfolio_exposure: 1.0
    
  # Action space
  action_space:
    type: "multi_discrete"  # "multi_discrete", "multi_continuous", "portfolio_weights"
    actions_per_ticker: 3  # buy, hold, sell
    action_scaling: 0.1  # Scale factor for continuous actions
    
  # Observation space
  observation_space:
    include_technical_features: true
    include_microstructure_features: true
    include_time_features: true
    include_portfolio_features: true
    include_correlation_features: true
    include_regime_features: true
    sequence_length: 20  # For LSTM
    flatten_sequence: false
    
  # Reward configuration
  reward:
    type: "multi_ticker"  # "single_ticker" or "multi_ticker"
    scaling: 1.0
    clipping: true
    clip_value: 10.0
    
  # Episode parameters
  episode:
    length: 252  # Trading days
    done_conditions:
      portfolio_drawdown: 0.5
      portfolio_sharpe: -2.0
      time_limit: true
      
  # Risk management
  risk_management:
    max_drawdown: 0.2
    max_position_concentration: 0.3
    max_sector_concentration: 0.5
    stop_loss: 0.05
    take_profit: 0.1
    trailing_stop: true
    
  # Execution simulation
  execution:
    market_impact_model: "linear"  # "linear", "square_root", "none"
    market_impact_factor: 0.1
    execution_delay: 0  # In minutes
    partial_fills: true
    fill_probability: 0.9

# PPO-LSTM configuration
ppo_lstm:
  # Model architecture
  model:
    lstm_hidden_size: 64
    lstm_num_layers: 2
    lstm_dropout: 0.2
    policy_hidden_size: 64
    value_hidden_size: 64
    activation: "tanh"  # "tanh", "relu", "leaky_relu"
    
  # Training parameters
  training:
    learning_rate: 0.0003
    batch_size: 64
    n_steps: 2048
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    clip_range_vf: null
    normalize_advantage: true
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
    
  # Optimization
  optimization:
    optimizer: "adam"  # "adam", "rmsprop", "sgd"
    weight_decay: 0.0
    learning_rate_schedule: "linear"  # "linear", "cosine", "constant"
    entropy_schedule: "linear"  # "linear", "constant", "exponential"
    
  # Multi-ticker specific
  multi_ticker:
    shared_features: true
    ticker_embedding_dim: 16
    attention_mechanism: false
    portfolio_attention: false
    
  # Curriculum learning
  curriculum:
    enabled: true
    phases:
      - name: "single_ticker_warmup"
        duration: 100
        num_tickers: 1
      - name: "small_portfolio"
        duration: 200
        num_tickers: 3
      - name: "medium_portfolio"
        duration: 300
        num_tickers: 5
      - name: "full_portfolio"
        duration: null
        num_tickers: null

# Walk-forward optimization configuration
walkforward:
  enabled: true
  
  # Cross-validation method
  cross_validation:
    method: "leave_one_ticker_out"  # "leave_one_ticker_out", "k_fold", "time_series"
    k_folds: 5
    embargo_period: 5  # Trading days between train and test
    
  # Time windows
  windows:
    train_window: 252  # 1 year
    test_window: 63  # 1 quarter
    step_size: 63  # 1 quarter
    expansion_window: true
    
  # Regime-aware splitting
  regime_aware:
    enabled: true
    ensure_regime_representation: true
    min_regime_samples: 20
    
  # Validation metrics
  metrics:
    primary: "sharpe_ratio"  # Primary metric for model selection
    secondary: ["sortino_ratio", "calmar_ratio", "max_drawdown", "total_return"]
    aggregation_method: "mean"  # "mean", "median", "geometric_mean"
    
  # Model selection
  selection:
    method: "best_primary"  # "best_primary", "multi_objective", "ensemble"
    ensemble_method: "weighted"  # "weighted", "voting", "stacking"
    
  # Results aggregation
  aggregation:
    method: "walk_forward"  # "walk_forward", "ensemble", "best_model"
    confidence_intervals: true
    bootstrap_samples: 1000
    
  # Reporting
  reporting:
    enabled: true
    generate_plots: true
    save_models: true
    save_predictions: true
    output_dir: "results/walkforward"

# Optuna hyperparameter optimization configuration
optuna:
  enabled: true
  
  # Study configuration
  study:
    name: "multi_ticker_rl_optimization"
    direction: "maximize"
    storage: "sqlite:///optuna_studies/multi_ticker_rl.db"
    load_if_exists: true
    
  # Search space
  search_space:
    # Learning rate
    learning_rate:
      type: "float"
      low: 0.0001
      high: 0.001
      log: true
      
    # Batch size
    batch_size:
      type: "categorical"
      choices: [32, 64, 128]
      
    # LSTM parameters
    lstm_hidden_size:
      type: "categorical"
      choices: [32, 64, 128]
      
    lstm_num_layers:
      type: "int"
      low: 1
      high: 3
      
    # PPO parameters
    n_steps:
      type: "categorical"
      choices: [1024, 2048, 4096]
      
    gamma:
      type: "float"
      low: 0.9
      high: 0.999
      
    gae_lambda:
      type: "float"
      low: 0.8
      high: 0.95
      
    clip_range:
      type: "float"
      low: 0.1
      high: 0.3
      
    ent_coef:
      type: "float"
      low: 0.001
      high: 0.1
      
    # Reward parameters
    reward_weights:
      portfolio_pnl:
        type: "float"
        low: 0.5
        high: 2.0
        
      sharpe_ratio:
        type: "float"
        low: 0.1
        high: 1.0
        
      diversification:
        type: "float"
        low: 0.1
        high: 1.0
        
      transaction_cost:
        type: "float"
        low: 0.1
        high: 1.0
        
  # Multi-objective optimization
  multi_objective:
    enabled: true
    objectives:
      - name: "sharpe_ratio"
        direction: "maximize"
      - name: "max_drawdown"
        direction: "minimize"
      - name: "trade_frequency"
        direction: "minimize"
      
  # Optimization settings
  optimization:
    n_trials: 100
    timeout: null  # In seconds
    n_jobs: 4  # Number of parallel jobs
    sampler: "tpe"  # "tpe", "random", "grid"
    pruner: "median"  # "median", "hyperband", "successive_halving"
    
  # Early stopping
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.01
    
  # Parallel execution
  parallel:
    enabled: true
    backend: "ray"  # "ray", "joblib", "multiprocessing"
    num_workers: 4
    
  # Results
  results:
    save_study: true
    save_trials: true
    save_best_model: true
    output_dir: "results/optuna"
    visualization: true

# Population-based training configuration (optional)
pbt:
  enabled: false
  
  # Population settings
  population:
    size: 10
    initial_population: "random"  # "random", "sobol", "halton"
    
  # Mutation settings
  mutation:
    rate: 0.1
    strength: 0.2
    methods:
      - "perturb"
      - "crossover"
      - "resample"
      
  # Training settings
  training:
    steps_per_population: 1000
    eval_episodes: 10
    truncation_selection: true
    truncation_fraction: 0.5
    
  # Hyperparameter space
  hyperparameters:
    learning_rate:
      min: 0.0001
      max: 0.001
      log: true
      
    batch_size:
      values: [32, 64, 128]
      
    clip_range:
      min: 0.1
      max: 0.3
      
    ent_coef:
      min: 0.001
      max: 0.1
      
  # Reward evolution
  reward_evolution:
    enabled: true
    mutation_rate: 0.05
    crossover_rate: 0.1
    
  # Results
  results:
    save_population: true
    save_history: true
    output_dir: "results/pbt"
    visualization: true

# Logging and monitoring configuration
logging:
  # General logging
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/training.log"
  max_file_size: "100MB"
  backup_count: 5
  
  # TensorBoard logging
  tensorboard:
    enabled: true
    log_dir: "logs/tensorboard"
    update_freq: 100
    histogram_freq: 1000
    
  # Metrics logging
  metrics:
    enabled: true
    metrics_dir: "logs/metrics"
    save_frequency: 100
    format: "json"  # "json", "csv", "parquet"
    
  # Performance tracking
  performance:
    enabled: true
    track_portfolio_metrics: true
    track_reward_components: true
    track_universe_changes: true
    save_frequency: 100
    
  # Monitoring
  monitoring:
    enabled: true
    dashboard_port: 8080
    update_frequency: 10  # seconds
    alerts:
      enabled: true
      channels: ["email", "slack"]
      conditions:
        - metric: "portfolio_drawdown"
          threshold: 0.2
          comparison: "greater_than"
        - metric: "sharpe_ratio"
          threshold: -1.0
          comparison: "less_than"

# Paths configuration
paths:
  data_root: "data"
  cache_dir: "data/cache"
  models_dir: "models"
  logs_dir: "logs"
  results_dir: "results"
  configs_dir: "configs"
  scripts_dir: "scripts"

# Secrets configuration
secrets:
  polygon_api_key: ""  # Set via environment variable
  databento_api_key: ""  # Set via environment variable
  email_password: ""  # Set via environment variable
  slack_webhook: ""  # Set via environment variable
```

## Configuration Validation

### Validation Rules
1. **Required Fields**: Ensure all required fields are present
2. **Data Types**: Validate data types for all fields
3. **Value Ranges**: Ensure numeric values are within valid ranges
4. **Dependencies**: Validate dependencies between configuration sections
5. **Consistency**: Ensure consistent settings across sections

### Validation Schema
The configuration can be validated using a JSON schema or Pydantic models. Here's an example of a Pydantic validation model:

```python
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
import yaml

class GlobalConfig(BaseModel):
    mode: str = Field(..., regex=r"^(single_ticker|multi_ticker)$")
    random_seed: int = 42
    device: str = Field("auto", regex=r"^(auto|cpu|cuda)$")
    num_workers: int = Field(4, gt=0)
    log_level: str = Field("INFO", regex=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

class MultiTickerConfig(BaseModel):
    enabled: bool = True
    max_tickers: int = Field(10, gt=0)
    min_tickers: int = Field(3, gt=0)
    portfolio_rebalance_frequency: str = "1day"
    position_sizing_method: str = Field(..., regex=r"^(equal_weight|risk_parity|kelly|custom)$")
    max_portfolio_exposure: float = Field(1.0, gt=0, le=1.0)
    min_position_size: float = Field(0.01, gt=0)
    max_position_size: float = Field(0.3, gt=0)
    correlation_threshold: float = Field(0.7, ge=0, le=1.0)
    diversification_weight: float = Field(0.3, ge=0, le=1.0)
    
    @validator('min_tickers')
    def min_tickers_le_max_tickers(cls, v, values):
        if 'max_tickers' in values and v > values['max_tickers']:
            raise ValueError('min_tickers must be less than or equal to max_tickers')
        return v

class DataConfig(BaseModel):
    source: str = Field(..., regex=r"^(polygon|databento|combined)$")
    tickers: List[str]
    timezone: str = "America/New_York"
    start_date: str
    end_date: str
    trading_hours: Dict[str, str]
    quality_checks: Dict[str, Any]
    cache: Dict[str, Any]
    validation: Dict[str, Any]

class MultiTickerRewardConfig(BaseModel):
    enabled: bool = True
    components: Dict[str, Any]
    correlation: Dict[str, Any]
    risk_adjustment: Dict[str, Any]
    regime_detection: Dict[str, Any]
    decomposition: Dict[str, Any]
    normalization: Dict[str, Any]
    smoothing: Dict[str, Any]

class CompleteConfig(BaseModel):
    version: str
    description: str
    global: GlobalConfig
    multi_ticker: MultiTickerConfig
    data: DataConfig
    multi_ticker_reward: MultiTickerRewardConfig
    # Add other configuration sections...
    
    @validator('version')
    def version_compatibility(cls, v):
        if v < "2.0.0":
            raise ValueError('Configuration version must be at least 2.0.0')
        return v

def load_and_validate_config(config_path: str) -> CompleteConfig:
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return CompleteConfig(**config_dict)
```

## Environment Overrides

### Environment Variable Mapping
Configuration values can be overridden using environment variables. The mapping follows a hierarchical structure:

```bash
# Global settings
export RL_MODE=multi_ticker
export RL_RANDOM_SEED=42
export RL_DEVICE=cuda

# Multi-ticker settings
export RL_MULTITICKER_MAX_TICKERS=10
export RL_MULTITICKER_MIN_TICKERS=3

# Data settings
export RL_DATA_SOURCE=polygon
export RL_DATA_START_DATE=2020-01-01
export RL_DATA_END_DATE=2023-12-31

# Reward settings
export RL_REWARD_COMPONENTS_PORTFOLIO_PNL_WEIGHT=1.0
export RL_REWARD_COMPONENTS_SHARPE_RATIO_WEIGHT=0.5

# Secrets
export POLYGON_API_KEY=your_api_key_here
export DATABENTO_API_KEY=your_api_key_here
```

### Override Logic
1. Environment variables take precedence over YAML configuration
2. Nested configuration is accessed using underscores
3. Arrays and dictionaries can be partially overridden
4. Type conversion is applied based on the configuration schema

## Configuration Examples

### Single-Ticker Configuration (Backward Compatibility)
```yaml
version: "2.0.0"
description: "Single-ticker RL trading system"

global:
  mode: "single_ticker"
  random_seed: 42
  device: "auto"
  num_workers: 4
  log_level: "INFO"

data:
  source: "polygon"
  tickers: ["AAPL"]
  timezone: "America/New_York"
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  trading_hours:
    start: "09:30"
    end: "16:00"
    timezone: "America/New_York"

# Single-ticker reward configuration
reward:
  type: "hybrid2"
  scaling: 1.0
  clipping: true
  clip_value: 10.0
  components:
    pnl:
      weight: 1.0
    dsr:
      weight: 0.5
    sharpe:
      weight: 0.3
    directional:
      weight: 0.2

# Environment configuration
environment:
  type: "single_ticker"
  parameters:
    initial_cash: 100000
    commission: 0.001
    slippage: 0.0005
    max_position_size: 1.0
    min_position_size: 0.01
```

### Multi-Ticker Configuration
```yaml
version: "2.0.0"
description: "Multi-ticker RL trading system with reward overhaul"

global:
  mode: "multi_ticker"
  random_seed: 42
  device: "cuda"
  num_workers: 8
  log_level: "INFO"

multi_ticker:
  enabled: true
  max_tickers: 10
  min_tickers: 3
  portfolio_rebalance_frequency: "1day"
  position_sizing_method: "equal_weight"
  max_portfolio_exposure: 1.0
  min_position_size: 0.01
  max_position_size: 0.3

data:
  source: "polygon"
  tickers:
    - "AAPL"
    - "MSFT"
    - "GOOGL"
    - "AMZN"
    - "META"
    - "TSLA"
    - "NVDA"
    - "JPM"
    - "JNJ"
    - "V"
  timezone: "America/New_York"
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  trading_hours:
    start: "09:30"
    end: "16:00"
    timezone: "America/New_York"

multi_ticker_reward:
  enabled: true
  components:
    portfolio_pnl:
      enabled: true
      weight: 1.0
    sharpe_ratio:
      enabled: true
      weight: 0.5
    diversification:
      enabled: true
      weight: 0.4
    transaction_cost:
      enabled: true
      weight: 0.5

dynamic_universe_selection:
  enabled: true
  universe:
    candidate_tickers:
      - "AAPL"
      - "MSFT"
      - "GOOGL"
      - "AMZN"
      - "META"
      - "TSLA"
      - "NVDA"
      - "JPM"
      - "JNJ"
      - "V"
      - "PG"
      - "UNH"
      - "HD"
      - "BAC"
      - "XOM"
      - "PFE"
      - "CSCO"
      - "ADBE"
      - "CRM"
    min_universe_size: 5
    max_universe_size: 10
    default_universe_size: 8
    rebalance_frequency: "1day"
```

## Migration Guide

### From Single-Ticker to Multi-Ticker
1. **Update Configuration Version**: Change version to "2.0.0"
2. **Set Global Mode**: Set `global.mode` to "multi_ticker"
3. **Add Multi-Ticker Section**: Add the `multi_ticker` section with appropriate settings
4. **Update Data Section**: Add multiple tickers to the `data.tickers` list
5. **Update Reward Section**: Replace single-ticker reward config with `multi_ticker_reward` section
6. **Add Dynamic Universe**: Add `dynamic_universe_selection` section if needed
7. **Update Environment**: Set `environment.type` to "multi_ticker"
8. **Update PPO-LSTM**: Add multi-ticker specific settings to `ppo_lstm.multi_ticker`

### Configuration Validation
1. **Run Validation**: Use the provided validation script to check configuration
2. **Check Dependencies**: Ensure all required dependencies are satisfied
3. **Test Loading**: Test configuration loading in your application
4. **Monitor Runtime**: Monitor for any configuration-related issues at runtime

## Best Practices

### Configuration Management
1. **Version Control**: Store configuration files in version control
2. **Environment Separation**: Use different configurations for development, testing, and production
3. **Documentation**: Document all configuration options and their effects
4. **Validation**: Always validate configuration before use
5. **Monitoring**: Monitor configuration changes and their effects

### Performance Optimization
1. **Caching**: Cache configuration to avoid repeated loading
2. **Lazy Loading**: Load configuration sections only when needed
3. **Environment Overrides**: Use environment overrides for secrets and environment-specific settings
4. **Hot Reload**: Support for hot reloading configuration changes (if needed)

### Security
1. **Secrets Management**: Never store secrets in configuration files
2. **Access Control**: Restrict access to configuration files
3. **Encryption**: Encrypt sensitive configuration values
4. **Audit Trail**: Log configuration changes for audit purposes