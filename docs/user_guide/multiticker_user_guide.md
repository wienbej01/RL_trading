# Multi-Ticker RL Trading System User Guide

## Overview

This guide provides comprehensive instructions for using the Multi-Ticker RL Trading System, an advanced reinforcement learning framework designed for intraday trading across multiple financial instruments. The system combines sophisticated reward shaping, portfolio management, and robust validation methodologies to deliver high-performance trading strategies.

## Key Features

- **Multi-Ticker Support**: Trade multiple instruments simultaneously with intelligent portfolio allocation
- **Advanced Reward System**: Hybrid reward function with regime weighting and asymmetric penalties
- **Dynamic Universe Selection**: Automatically adapt to changing market conditions
- **Walk-Forward Optimization**: Robust validation with Leave-One-Ticker-Out cross-validation
- **Hyperparameter Optimization**: Automated tuning with Optuna for optimal performance
- **Rich Monitoring**: Comprehensive performance tracking and alerting

## System Architecture

The Multi-Ticker RL Trading System consists of several key components:

1. **Data Loading**: Efficient loading and preprocessing of multi-ticker market data
2. **Feature Engineering**: Advanced feature extraction with ticker-specific normalization
3. **RL Environment**: Multi-ticker trading environment with portfolio management
4. **Training**: PPO-LSTM training with curriculum learning and reward shaping
5. **Evaluation**: Comprehensive performance evaluation and benchmarking
6. **Monitoring**: Real-time performance tracking and alerting

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Sufficient disk space for market data (varies by universe size)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/rl-intraday.git
cd rl-intraday
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
export POLYGON_API_KEY="your_polygon_api_key"
export RL_DATA_ROOT="/path/to/data"
export RL_CACHE_DIR="/path/to/cache"
```

## Configuration

### Configuration File Structure

The system uses a YAML configuration file located at `configs/settings.yaml`. The configuration is organized into several sections:

```yaml
# Data configuration
data:
  tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  data_source: "polygon"  # or "databento"
  resample_freq: "5min"
  
# Feature engineering
features:
  technical:
    calculate_returns: true
    sma_windows: [10, 20, 50]
    calculate_rsi: true
    calculate_macd: true
    calculate_bollinger_bands: true
  microstructure:
    calculate_spread: true
    calculate_vwap: true
  time:
    extract_time_of_day: true
    extract_day_of_week: true
    
# Environment configuration
environment:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
  max_positions: 5
  position_size: 0.2
  reward_type: "hybrid2"
  
# Training configuration
training:
  total_timesteps: 1000000
  learning_rate: 0.0003
  batch_size: 64
  n_steps: 2048
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  
# Walk-forward optimization
walkforward:
  enabled: true
  n_folds: 5
  embargo_days: 10
  test_size: 0.2
  regime_aware: true
  
# Hyperparameter optimization
hpo:
  enabled: true
  n_trials: 100
  direction: "maximize"
  metric: "sharpe_ratio"
  pruner: "median"
  sampler: "tpe"
  
# Logging and monitoring
logging:
  level: "INFO"
  tensorboard: true
  checkpoint_freq: 10000
  eval_freq: 5000
  
monitoring:
  enabled: true
  alert_thresholds:
    drawdown: 0.15
    sharpe_ratio: 0.5
    win_rate: 0.45
```

### Multi-Ticker Configuration

For multi-ticker trading, the configuration includes several important parameters:

```yaml
# Multi-ticker specific settings
multiticker:
  # Universe selection
  universe:
    selection_method: "dynamic"  # "fixed" or "dynamic"
    max_tickers: 10
    min_tickers: 3
    rebalance_freq: "1M"
    selection_metrics: ["liquidity", "volatility", "trend_strength"]
    
  # Portfolio management
  portfolio:
    allocation_method: "equal"  # "equal", "risk_parity", "kelly", "adaptive"
    max_correlation: 0.7
    rebalance_threshold: 0.1
    position_sizing: "fixed_fraction"  # "fixed_fraction", "kelly", "volatility_target"
    
  # Cross-ticker correlations
  correlations:
    use_correlation_matrix: true
    correlation_window: 20
    correlation_threshold: 0.5
    diversification_bonus: 0.1
    
  # Reward function
  reward:
    cross_ticker_weight: 0.3
    portfolio_weight: 0.4
    individual_weight: 0.3
    regime_weighting: true
    asymmetric_drawdown: true
```

## Usage

### Basic Usage

#### 1. Training a Model

```python
from src.rl.multiticker_trainer import MultiTickerRLTrainer
from src.utils.config_loader import load_config

# Load configuration
config = load_config()

# Initialize trainer
trainer = MultiTickerRLTrainer(config)

# Train model
model = trainer.train()

# Save model
trainer.save_model("models/multiticker_model")
```

#### 2. Running Backtests

```python
from src.evaluation.multiticker_evaluator import MultiTickerEvaluator
from src.rl.multiticker_policy import MultiTickerPPOLSTMPolicy

# Load model
model = MultiTickerPPOLSTMPolicy.load("models/multiticker_model")

# Initialize evaluator
evaluator = MultiTickerEvaluator(config)

# Run backtest
results = evaluator.evaluate_model(model, test_data)

# Print results
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

#### 3. Walk-Forward Optimization

```python
from src.rl.multiticker_trainer import MultiTickerRLTrainer

# Initialize trainer with WFO enabled
trainer = MultiTickerRLTrainer(config)

# Run walk-forward optimization
wfo_results = trainer.walk_forward_training()

# Print aggregated results
print(f"Average Sharpe Ratio: {wfo_results['avg_sharpe_ratio']:.2f}")
print(f"Average Total Return: {wfo_results['avg_total_return']:.2%}")
print(f"Average Max Drawdown: {wfo_results['avg_max_drawdown']:.2%}")
```

#### 4. Hyperparameter Optimization

```python
from src.rl.multiticker_trainer import MultiTickerRLTrainer

# Initialize trainer with HPO enabled
trainer = MultiTickerRLTrainer(config)

# Run hyperparameter optimization
hpo_results = trainer.optimize_hyperparameters()

# Print best parameters
print("Best parameters:")
for param, value in hpo_results['best_params'].items():
    print(f"  {param}: {value}")

print(f"Best Sharpe Ratio: {hpo_results['best_value']:.2f}")
```

### Advanced Usage

#### 1. Custom Reward Functions

You can define custom reward functions by extending the `MultiTickerRewardCalculator` class:

```python
from src.sim.multiticker_env import MultiTickerRewardCalculator

class CustomRewardCalculator(MultiTickerRewardCalculator):
    def __init__(self, config):
        super().__init__(config)
        # Initialize custom parameters
        
    def calculate_reward(self, portfolio_state, action, next_portfolio_state):
        # Calculate base reward
        base_reward = super().calculate_reward(portfolio_state, action, next_portfolio_state)
        
        # Add custom reward components
        custom_reward = self._calculate_custom_reward(portfolio_state, action, next_portfolio_state)
        
        # Combine rewards
        total_reward = base_reward + custom_reward
        
        return total_reward
    
    def _calculate_custom_reward(self, portfolio_state, action, next_portfolio_state):
        # Implement custom reward logic
        # For example, reward for diversification
        diversification_reward = self._calculate_diversification_reward(portfolio_state)
        
        return diversification_reward
    
    def _calculate_diversification_reward(self, portfolio_state):
        # Calculate diversification metric
        positions = portfolio_state['positions']
        if len(positions) <= 1:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        total_value = sum(abs(pos) for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        hhi = sum((abs(pos) / total_value) ** 2 for pos in positions.values())
        
        # Reward for lower HHI (more diversified)
        diversification_reward = (1 - hhi) * 0.1
        
        return diversification_reward

# Use custom reward calculator
config['environment']['reward_calculator'] = CustomRewardCalculator
```

#### 2. Custom Feature Engineering

You can extend the feature pipeline to include custom features:

```python
from src.features.multiticker_pipeline import MultiTickerFeaturePipeline

class CustomFeaturePipeline(MultiTickerFeaturePipeline):
    def __init__(self, config):
        super().__init__(config)
        # Initialize custom feature parameters
        
    def _extract_features(self, data):
        # Extract base features
        features = super()._extract_features(data)
        
        # Add custom features
        custom_features = self._extract_custom_features(data)
        
        # Combine features
        combined_features = pd.concat([features, custom_features], axis=1)
        
        return combined_features
    
    def _extract_custom_features(self, data):
        custom_features = pd.DataFrame(index=data.index)
        
        # Example: Custom momentum feature
        for ticker in data.columns.get_level_values('ticker').unique():
            if (ticker, 'close') in data.columns:
                prices = data[(ticker, 'close')]
                # Calculate custom momentum indicator
                momentum = self._calculate_custom_momentum(prices)
                custom_features[(ticker, 'custom_momentum')] = momentum
        
        return custom_features
    
    def _calculate_custom_momentum(self, prices):
        # Implement custom momentum calculation
        # For example, weighted momentum with more weight on recent prices
        returns = prices.pct_change()
        
        # Calculate weighted returns
        weights = np.exp(np.linspace(-1, 0, 21))  # Exponential weights
        weights = weights / weights.sum()
        
        weighted_momentum = returns.rolling(21).apply(
            lambda x: np.sum(x * weights) if not x.isna().any() else np.nan
        )
        
        return weighted_momentum

# Use custom feature pipeline
config['features']['pipeline_class'] = CustomFeaturePipeline
```

#### 3. Custom Universe Selection

You can implement custom universe selection strategies:

```python
from src.data.multiticker_data_loader import MultiTickerDataLoader

class CustomUniverseSelector:
    def __init__(self, config):
        self.config = config
        # Initialize custom parameters
        
    def select_universe(self, data, current_universe=None):
        """
        Select universe of tickers based on custom criteria.
        
        Args:
            data: Market data for all potential tickers
            current_universe: Current universe of tickers
            
        Returns:
            Selected universe of tickers
        """
        # Get selection parameters
        max_tickers = self.config.get('max_tickers', 10)
        min_tickers = self.config.get('min_tickers', 3)
        
        # Calculate selection metrics for each ticker
        ticker_metrics = {}
        
        for ticker in data.columns.get_level_values('ticker').unique():
            if (ticker, 'close') in data.columns:
                prices = data[(ticker, 'close')]
                volume = data[(ticker, 'volume')] if (ticker, 'volume') in data.columns else None
                
                # Calculate custom selection metrics
                metrics = self._calculate_selection_metrics(prices, volume)
                ticker_metrics[ticker] = metrics
        
        # Score and rank tickers
        ticker_scores = {}
        for ticker, metrics in ticker_metrics.items():
            score = self._calculate_ticker_score(metrics)
            ticker_scores[ticker] = score
        
        # Sort tickers by score
        sorted_tickers = sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top tickers
        selected_tickers = [ticker for ticker, _ in sorted_tickers[:max_tickers]]
        
        # Ensure minimum number of tickers
        if len(selected_tickers) < min_tickers:
            selected_tickers = [ticker for ticker, _ in sorted_tickers[:min_tickers]]
        
        return selected_tickers
    
    def _calculate_selection_metrics(self, prices, volume=None):
        # Calculate custom selection metrics
        metrics = {}
        
        # Liquidity metric (if volume data is available)
        if volume is not None:
            metrics['liquidity'] = volume.mean()
        else:
            metrics['liquidity'] = 1.0  # Default value
        
        # Volatility metric
        returns = prices.pct_change().dropna()
        metrics['volatility'] = returns.std()
        
        # Trend strength metric
        metrics['trend_strength'] = self._calculate_trend_strength(prices)
        
        # Price momentum
        metrics['momentum'] = (prices.iloc[-1] / prices.iloc[-21]) - 1
        
        return metrics
    
    def _calculate_trend_strength(self, prices):
        # Calculate trend strength using linear regression
        x = np.arange(len(prices))
        y = prices.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 2:
            return 0.0
        
        # Calculate linear regression
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize slope by price level
        trend_strength = slope / np.mean(y)
        
        return trend_strength
    
    def _calculate_ticker_score(self, metrics):
        # Calculate composite score from metrics
        # This is a simple example - you can implement more complex scoring
        
        # Normalize metrics
        liquidity_score = min(metrics['liquidity'] / 1e6, 1.0)  # Cap at 1M volume
        volatility_score = min(metrics['volatility'] / 0.02, 1.0)  # Cap at 2% daily vol
        trend_score = min(abs(metrics['trend_strength']) * 100, 1.0)  # Cap and scale
        momentum_score = min(abs(metrics['momentum']) * 10, 1.0)  # Cap and scale
        
        # Calculate weighted score
        score = (
            0.2 * liquidity_score +
            0.3 * volatility_score +
            0.25 * trend_score +
            0.25 * momentum_score
        )
        
        return score

# Use custom universe selector
config['multiticker']['universe']['selector_class'] = CustomUniverseSelector
```

## Monitoring and Evaluation

### Performance Metrics

The system tracks a comprehensive set of performance metrics:

#### Portfolio-Level Metrics
- **Total Return**: Overall portfolio return
- **Annualized Return**: Return annualized for comparison
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Maximum peak-to-trough decline
- **Calmar Ratio**: Return relative to max drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses

#### Ticker-Level Metrics
- **Individual Returns**: Return for each ticker
- **Contribution to Portfolio**: Contribution of each ticker to overall performance
- **Correlation with Portfolio**: Correlation of each ticker with portfolio returns
- **Turnover**: Trading frequency for each ticker

#### Risk Metrics
- **Value at Risk (VaR)**: Maximum expected loss at a given confidence level
- **Conditional VaR (CVaR)**: Expected loss given that VaR is breached
- **Beta**: Market sensitivity
- **Alpha**: Excess return relative to market

### Monitoring Dashboard

The system includes a real-time monitoring dashboard that provides:

1. **Performance Overview**: Key performance metrics at a glance
2. **Equity Curve**: Visualization of portfolio value over time
3. **Drawdown Analysis**: Visualization of drawdown periods
4. **Ticker Contribution**: Breakdown of performance by ticker
5. **Risk Metrics**: Current risk exposure metrics
6. **Alerts**: Notifications for performance degradation or risk breaches

### Alerts

The system can be configured to send alerts when certain thresholds are breached:

```yaml
monitoring:
  alerts:
    enabled: true
    channels: ["email", "slack"]  # Notification channels
    
    # Performance alerts
    drawdown_threshold: 0.15  # Alert if drawdown exceeds 15%
    sharpe_threshold: 0.5    # Alert if Sharpe ratio drops below 0.5
    win_rate_threshold: 0.45  # Alert if win rate drops below 45%
    
    # Risk alerts
    var_threshold: 0.05      # Alert if daily VaR exceeds 5%
    concentration_threshold: 0.3  # Alert if single position exceeds 30%
    
    # System alerts
    data_freshness: 3600     # Alert if data is older than 1 hour (in seconds)
    system_health: true      # Alert on system errors
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors During Training

**Problem**: Training fails with out-of-memory errors.

**Solution**:
- Reduce `batch_size` in configuration
- Reduce `n_steps` in configuration
- Enable gradient checkpointing
- Use mixed precision training

```yaml
training:
  batch_size: 32  # Reduced from 64
  n_steps: 1024   # Reduced from 2048
  use_gradient_checkpointing: true
  mixed_precision: true
```

#### 2. Slow Training Performance

**Problem**: Training is slower than expected.

**Solution**:
- Ensure GPU is being used
- Increase `num_workers` for data loading
- Optimize feature pipeline
- Use smaller universe of tickers

```yaml
training:
  num_workers: 8  # Increase based on CPU cores
  device: "cuda"  # Ensure GPU is used

data:
  num_workers: 8  # Increase for faster data loading

multiticker:
  universe:
    max_tickers: 5  # Reduce for faster training
```

#### 3. Poor Model Performance

**Problem**: Model performance is worse than expected.

**Solution**:
- Check data quality and preprocessing
- Adjust reward function parameters
- Tune hyperparameters using HPO
- Increase training timesteps
- Try different feature sets

```yaml
environment:
  reward_type: "hybrid2"  # Try different reward types
  reward_params:
    pnl_weight: 0.5
    dsr_weight: 0.3
    sharpe_weight: 0.2

training:
  total_timesteps: 2000000  # Increase training time

hpo:
  enabled: true  # Enable hyperparameter optimization
  n_trials: 200  # Increase number of trials
```

#### 4. Data Loading Issues

**Problem**: Errors when loading market data.

**Solution**:
- Check data file paths and permissions
- Verify data format and structure
- Ensure date ranges are valid
- Check for missing or corrupted data

```python
# Debug data loading
from src.data.multiticker_data_loader import MultiTickerDataLoader
from src.utils.config_loader import load_config

config = load_config()
loader = MultiTickerDataLoader(config)

# Try loading data for a single ticker first
single_ticker_data = loader.load_data(["AAPL"])
print(f"Single ticker data shape: {single_ticker_data.shape}")

# Check for missing values
print(f"Missing values: {single_ticker_data.isnull().sum().sum()}")

# Check date range
print(f"Date range: {single_ticker_data.index.min()} to {single_ticker_data.index.max()}")
```

### Debug Mode

The system includes a debug mode that provides additional logging and validation:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode in configuration
config['debug'] = True
config['debug']['validate_data'] = True
config['debug']['log_gradients'] = True
config['debug']['log_rewards'] = True
```

## Best Practices

### 1. Data Management

- **Data Quality**: Ensure high-quality, clean market data
- **Data Freshness**: Regularly update market data
- **Data Validation**: Implement data validation checks
- **Data Backup**: Maintain backups of historical data

### 2. Model Training

- **Start Simple**: Begin with simple configurations and gradually increase complexity
- **Hyperparameter Tuning**: Use systematic hyperparameter optimization
- **Cross-Validation**: Implement robust validation methodologies
- **Regular Evaluation**: Regularly evaluate model performance

### 3. Risk Management

- **Position Sizing**: Use appropriate position sizing strategies
- **Diversification**: Maintain diversified portfolios
- **Risk Limits**: Implement and respect risk limits
- **Regular Monitoring**: Continuously monitor risk metrics

### 4. Deployment

- **Staging Environment**: Test in staging before production
- **Gradual Rollout**: Gradually increase trading capital
- **Performance Monitoring**: Continuously monitor performance
- **Fallback Mechanisms**: Implement fallback mechanisms

## Examples and Tutorials

### Example 1: Basic Multi-Ticker Training

```python
"""
Basic example of training a multi-ticker RL model.
"""

from src.rl.multiticker_trainer import MultiTickerRLTrainer
from src.utils.config_loader import load_config

# Load configuration
config = load_config("configs/basic_multiticker.yaml")

# Initialize trainer
trainer = MultiTickerRLTrainer(config)

# Train model
print("Starting training...")
model = trainer.train()

# Save model
trainer.save_model("models/basic_multiticker_model")

print("Training completed and model saved.")
```

### Example 2: Walk-Forward Optimization

```python
"""
Example of running walk-forward optimization.
"""

from src.rl.multiticker_trainer import MultiTickerRLTrainer
from src.utils.config_loader import load_config

# Load configuration with WFO enabled
config = load_config("configs/wfo_multiticker.yaml")

# Initialize trainer
trainer = MultiTickerRLTrainer(config)

# Run walk-forward optimization
print("Starting walk-forward optimization...")
wfo_results = trainer.walk_forward_training()

# Print results
print("\nWalk-Forward Optimization Results:")
print(f"Average Sharpe Ratio: {wfo_results['avg_sharpe_ratio']:.2f}")
print(f"Average Total Return: {wfo_results['avg_total_return']:.2%}")
print(f"Average Max Drawdown: {wfo_results['avg_max_drawdown']:.2%}")
print(f"Fold Results: {wfo_results['fold_results']}")

# Save results
trainer.save_wfo_results("results/wfo_results.pkl")

print("Walk-forward optimization completed.")
```

### Example 3: Hyperparameter Optimization

```python
"""
Example of running hyperparameter optimization.
"""

from src.rl.multiticker_trainer import MultiTickerRLTrainer
from src.utils.config_loader import load_config

# Load configuration with HPO enabled
config = load_config("configs/hpo_multiticker.yaml")

# Initialize trainer
trainer = MultiTickerRLTrainer(config)

# Run hyperparameter optimization
print("Starting hyperparameter optimization...")
hpo_results = trainer.optimize_hyperparameters()

# Print results
print("\nHyperparameter Optimization Results:")
print("Best parameters:")
for param, value in hpo_results['best_params'].items():
    print(f"  {param}: {value}")

print(f"\nBest Sharpe Ratio: {hpo_results['best_value']:.2f}")
print(f"Optimization History: {len(hpo_results['history'])} trials")

# Save results
trainer.save_hpo_results("results/hpo_results.pkl")

print("Hyperparameter optimization completed.")
```

### Example 4: Custom Reward Function

```python
"""
Example of implementing a custom reward function.
"""

from src.sim.multiticker_env import MultiTickerRewardCalculator
from src.rl.multiticker_trainer import MultiTickerRLTrainer
from src.utils.config_loader import load_config

class CustomRewardCalculator(MultiTickerRewardCalculator):
    def __init__(self, config):
        super().__init__(config)
        
    def calculate_reward(self, portfolio_state, action, next_portfolio_state):
        # Calculate base reward
        base_reward = super().calculate_reward(portfolio_state, action, next_portfolio_state)
        
        # Add custom reward for diversification
        diversification_reward = self._calculate_diversification_reward(portfolio_state)
        
        # Add custom penalty for concentration
        concentration_penalty = self._calculate_concentration_penalty(portfolio_state)
        
        # Combine rewards
        total_reward = base_reward + diversification_reward - concentration_penalty
        
        return total_reward
    
    def _calculate_diversification_reward(self, portfolio_state):
        positions = portfolio_state['positions']
        if len(positions) <= 1:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        total_value = sum(abs(pos) for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        hhi = sum((abs(pos) / total_value) ** 2 for pos in positions.values())
        
        # Reward for lower HHI (more diversified)
        diversification_reward = (1 - hhi) * 0.1
        
        return diversification_reward
    
    def _calculate_concentration_penalty(self, portfolio_state):
        positions = portfolio_state['positions']
        if len(positions) == 0:
            return 0.0
        
        # Calculate concentration
        total_value = sum(abs(pos) for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        max_concentration = max(abs(pos) / total_value for pos in positions.values())
        
        # Penalty for high concentration
        if max_concentration > 0.4:
            concentration_penalty = (max_concentration - 0.4) * 0.2
        else:
            concentration_penalty = 0.0
        
        return concentration_penalty

# Load configuration
config = load_config("configs/custom_reward.yaml")

# Set custom reward calculator
config['environment']['reward_calculator'] = CustomRewardCalculator

# Initialize trainer
trainer = MultiTickerRLTrainer(config)

# Train model with custom reward
print("Starting training with custom reward function...")
model = trainer.train()

# Save model
trainer.save_model("models/custom_reward_model")

print("Training with custom reward function completed.")
```

## API Reference

### MultiTickerRLTrainer

#### `__init__(self, config)`

Initialize the trainer with configuration.

**Parameters**:
- `config` (dict): Configuration dictionary

**Returns**:
- `MultiTickerRLTrainer`: Initialized trainer

#### `train(self)`

Train the RL model.

**Returns**:
- `MultiTickerPPOLSTMPolicy`: Trained model

#### `walk_forward_training(self)`

Run walk-forward optimization.

**Returns**:
- `dict`: Walk-forward optimization results

#### `optimize_hyperparameters(self)`

Run hyperparameter optimization.

**Returns**:
- `dict`: Hyperparameter optimization results

#### `save_model(self, path)`

Save trained model to disk.

**Parameters**:
- `path` (str): Path to save model

### MultiTickerEvaluator

#### `__init__(self, config)`

Initialize the evaluator with configuration.

**Parameters**:
- `config` (dict): Configuration dictionary

**Returns**:
- `MultiTickerEvaluator`: Initialized evaluator

#### `evaluate_model(self, model, data)`

Evaluate model on test data.

**Parameters**:
- `model` (MultiTickerPPOLSTMPolicy): Trained model
- `data` (dict): Test data

**Returns**:
- `dict`: Evaluation results

#### `generate_report(self, results, output_path)`

Generate evaluation report.

**Parameters**:
- `results` (dict): Evaluation results
- `output_path` (str): Path to save report

### MultiTickerDataLoader

#### `__init__(self, config)`

Initialize the data loader with configuration.

**Parameters**:
- `config` (dict): Configuration dictionary

**Returns**:
- `MultiTickerDataLoader`: Initialized data loader

#### `load_data(self, tickers=None, start_date=None, end_date=None)`

Load market data.

**Parameters**:
- `tickers` (list, optional): List of tickers to load
- `start_date` (str, optional): Start date for data
- `end_date` (str, optional): End date for data

**Returns**:
- `pd.DataFrame`: Loaded market data

#### `get_universe(self, date=None)`

Get current universe of tickers.

**Parameters**:
- `date` (str, optional): Date for universe selection

**Returns**:
- `list`: Universe of tickers

## Conclusion

The Multi-Ticker RL Trading System provides a comprehensive framework for developing and deploying reinforcement learning-based trading strategies across multiple financial instruments. By following this guide, you should be able to effectively use the system to train, evaluate, and deploy high-performance trading strategies.

For additional support or questions, please refer to the project documentation or contact the development team.