# Reward Mechanisms in RL Trading System

## Overview

This document provides a comprehensive guide to the single-ticker and multi-ticker reward mechanisms implemented in the RL trading system. The reward system is a critical component that shapes the agent's learning process and trading behavior.

## Single-Ticker Reward Mechanisms

### Available Reward Types

The system supports several reward types for single-ticker trading:

1. **PNL (Profit & Loss)**
   - Direct reward based on realized and unrealized profits/losses
   - Formula: `reward = (current_equity - initial_equity) / initial_equity`
   - Simple and intuitive but may lead to excessive risk-taking

2. **DSR (Downside Risk-Adjusted)**
   - Focuses on minimizing downside risk while capturing upside
   - Formula: `reward = return - risk_penalty * downside_volatility`
   - More conservative approach that penalizes volatility

3. **Sharpe Ratio**
   - Risk-adjusted return metric
   - Formula: `reward = mean_return / std_return`
   - Encourages consistent returns with lower volatility

4. **Blend**
   - Combination of multiple reward components
   - Formula: `reward = w1 * pnl + w2 * sharpe + w3 * dsr`
   - Balanced approach with configurable weights

5. **Directional**
   - Rewards correct directional predictions
   - Formula: `reward = sign(prediction) * sign(actual_return)`
   - Focuses on prediction accuracy rather than magnitude

6. **Hybrid**
   - Advanced combination with regime awareness
   - Formula: `reward = base_reward * regime_weight + activity_penalty`
   - Adapts to different market regimes

7. **Hybrid2 (Recommended)**
   - Most sophisticated single-ticker reward mechanism
   - Combines multiple components with advanced shaping

### Hybrid2 Reward Components

The Hybrid2 reward mechanism includes:

1. **Base PNL Component**
   ```python
   base_pnl = (current_equity - previous_equity) / previous_equity
   ```

2. **Risk Adjustment**
   ```python
   risk_penalty = drawdown_penalty * max_drawdown + volatility_penalty * return_volatility
   ```

3. **Regime Weighting**
   ```python
   regime_weight = get_regime_weight(current_market_regime)
   adjusted_reward = base_pnl * regime_weight
   ```

4. **Activity Shaping**
   ```python
   activity_penalty = lagrangian_activity_penalty(trade_frequency, target_frequency)
   ```

5. **Microstructure Features**
   ```python
   microstructure_bonus = pca_microstructure_features(market_microstructure_data)
   ```

6. **Opportunity/Capture Shaping**
   ```python
   opportunity_bonus = opportunity_cost_analysis(missed_opportunities, captured_opportunities)
   ```

7. **Potential-Based Reward Shaping**
   ```python
   potential_bonus = potential_based_shaping(current_state, next_state)
   ```

### Configuration Example

```yaml
environment:
  reward:
    type: "hybrid2"
    hybrid2:
      base_pnl_weight: 1.0
      risk_penalty_weight: 0.5
      regime_weighting: true
      activity_shaping: true
      microstructure_features: true
      opportunity_shaping: true
      potential_shaping: true
      drawdown_penalty: 2.0
      volatility_penalty: 0.3
      target_trade_frequency: 5.0
      lagrangian_lambda: 0.1
```

## Multi-Ticker Reward Mechanisms

### Overview

The multi-ticker reward system extends the single-ticker concepts to handle multiple assets simultaneously, with additional considerations for:

- Portfolio-level optimization
- Cross-ticker correlations
- Position sizing across assets
- Risk management at the portfolio level

### Multi-Ticker Hybrid2 Reward Components

1. **Portfolio PNL Component**
   ```python
   portfolio_pnl = (total_portfolio_value - previous_portfolio_value) / previous_portfolio_value
   ```

2. **Risk-Adjusted Returns**
   ```python
   portfolio_risk = calculate_portfolio_risk(positions, covariance_matrix)
   risk_adjusted_return = portfolio_pnl / (portfolio_risk + epsilon)
   ```

3. **Diversification Bonus**
   ```python
   diversification_bonus = calculate_diversification_score(positions, correlation_matrix)
   ```

4. **Cross-Ticker Correlation Penalty**
   ```python
   correlation_penalty = calculate_correlation_penalty(positions, correlation_matrix)
   ```

5. **Concentration Risk Penalty**
   ```python
   concentration_penalty = calculate_concentration_risk(positions)
   ```

6. **Regime-Aware Portfolio Adjustment**
   ```python
   regime_adjustment = get_portfolio_regime_adjustment(market_regime, sector_exposures)
   ```

7. **Activity Shaping (Portfolio Level)**
   ```python
   portfolio_activity_penalty = lagrangian_portfolio_activity_penalty(
       total_trades, target_portfolio_turnover
   )
   ```

### Configuration Example

```yaml
environment:
  reward:
    type: "multiticker_hybrid2"
    multiticker_hybrid2:
      portfolio_pnl_weight: 1.0
      risk_adjustment_weight: 0.8
      diversification_bonus_weight: 0.3
      correlation_penalty_weight: 0.4
      concentration_penalty_weight: 0.5
      regime_weighting: true
      activity_shaping: true
      drawdown_penalty: 2.5
      volatility_penalty: 0.4
      target_portfolio_turnover: 0.1  # Daily turnover target
      max_position_concentration: 0.2  # Max 20% in single position
      correlation_threshold: 0.7  # Penalty threshold for correlations
```

## Data Sources

### Single-Ticker Data Sources

1. **Primary Data Source**
   - Polygon.io for intraday OHLCV data
   - 1-minute resolution for intraday trading
   - Historical data from 2020 onwards

2. **Data Structure**
   ```
   data/polygon/historical/symbol={TICKER}/year={YEAR}/month={MONTH}/day={DAY}/data.parquet
   ```

3. **Data Columns**
   - timestamp: Unix timestamp in milliseconds
   - open: Opening price
   - high: Highest price
   - low: Lowest price
   - close: Closing price
   - volume: Trading volume
   - vwap: Volume-weighted average price
   - transactions: Number of transactions

### Multi-Ticker Data Sources

1. **Primary Data Source**
   - Same as single-ticker but for multiple symbols
   - Data loaded and aligned across tickers

2. **Data Structure**
   ```
   data/polygon/historical/symbol={TICKER}/year={YEAR}/month={MONTH}/day={DAY}/data.parquet
   ```

3. **Additional Data Requirements**
   - Correlation matrices between assets
   - Sector classifications
   - Market regime indicators
   - VIX data for market volatility

### Data Loading Process

1. **Single-Ticker Loading**
   ```python
   from src.data.data_loader import UnifiedDataLoader
   
   data_loader = UnifiedDataLoader(config_path="configs/settings.yaml")
   data = data_loader.load_ohlcv(
       symbol="AAPL",
       start=pd.Timestamp("2020-01-01"),
       end=pd.Timestamp("2024-12-31")
   )
   ```

2. **Multi-Ticker Loading**
   ```python
   tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
   all_data = []
   
   for ticker in tickers:
       ticker_data = data_loader.load_ohlcv(
           symbol=ticker,
           start=pd.Timestamp("2020-01-01"),
           end=pd.Timestamp("2024-12-31")
       )
       ticker_data['ticker'] = ticker
       all_data.append(ticker_data)
   
   data = pd.concat(all_data, axis=0)
   ```

## Implementation Details

### Reward Calculation Pipeline

1. **Single-Ticker Pipeline**
   ```
   Raw Data → Feature Engineering → Reward Calculation → Reward Shaping → Final Reward
   ```

2. **Multi-Ticker Pipeline**
   ```
   Raw Data → Feature Engineering → Portfolio Construction → 
   Portfolio Metrics → Reward Calculation → Reward Shaping → Final Reward
   ```

### Key Classes and Functions

1. **Single-Ticker Reward**
   - `src/sim/env_intraday_rl.py`: Contains reward calculation logic
   - `calculate_hybrid2_reward()`: Main reward calculation function
   - `apply_reward_shaping()`: Applies various shaping techniques

2. **Multi-Ticker Reward**
   - `src/sim/multiticker_env.py`: Multi-ticker environment
   - `calculate_multiticker_hybrid2_reward()`: Portfolio-level reward calculation
   - `calculate_portfolio_metrics()`: Portfolio risk and performance metrics

### Reward Decomposition and Logging

The system provides detailed reward decomposition for analysis:

```python
reward_components = {
    'base_pnl': 0.012,
    'risk_penalty': -0.003,
    'regime_adjustment': 1.2,
    'activity_penalty': -0.001,
    'microstructure_bonus': 0.002,
    'opportunity_bonus': 0.001,
    'potential_bonus': 0.0005,
    'final_reward': 0.0115
}
```

## Outstanding Issues and Limitations

### Known Issues

1. **Feature Selection Bug**
   - Issue: TypeError in feature selection when selected_features is None
   - Status: Identified, needs fix in `src/features/pipeline.py` line 1080
   - Impact: Prevents proper feature selection in multi-ticker pipeline

2. **Data Loading Limitations**
   - Issue: Synthetic data generation for missing tickers
   - Status: Temporary workaround implemented
   - Impact: Results may not reflect real market conditions

3. **Memory Usage**
   - Issue: High memory usage with multiple tickers and long time periods
   - Status: Optimization in progress
   - Impact: Limits scalability to large ticker universes

### Current Limitations

1. **Correlation Estimation**
   - Limited historical data for correlation estimation
   - Static correlation matrices (not time-varying)
   - No regime-specific correlation modeling

2. **Transaction Costs**
   - Simplified transaction cost model
   - No market impact modeling
   - Fixed commission structure

3. **Market Regime Detection**
   - Basic regime detection based on volatility
   - Limited regime types (high/low volatility)
   - No fundamental regime indicators

### Future Enhancements

1. **Advanced Reward Shaping**
   - Time-decay rewards for longer-term positions
   - Sector-specific reward adjustments
   - News sentiment integration

2. **Risk Management**
   - Dynamic position sizing based on volatility
   - Portfolio-level stop-loss mechanisms
   - Beta-neutral strategies

3. **Data Sources**
   - Alternative data integration
   - Real-time data feeds
   - Options market data for volatility surface

## Best Practices

### Single-Ticker Configuration

1. **Start Simple**
   - Begin with PNL or Sharpe ratio rewards
   - Gradually add complexity with Hybrid2
   - Monitor reward decomposition for insights

2. **Parameter Tuning**
   - Adjust reward weights based on backtesting
   - Balance risk and return components
   - Consider market conditions when setting parameters

### Multi-Ticker Configuration

1. **Diversification Focus**
   - Emphasize diversification bonus for correlated assets
   - Set appropriate concentration limits
   - Monitor portfolio-level metrics

2. **Risk Management**
   - Use correlation penalties to avoid overexposure
   - Implement position sizing rules
   - Regularly rebalance portfolio weights

### Monitoring and Analysis

1. **Reward Decomposition**
   - Regularly analyze reward components
   - Identify dominant reward drivers
   - Adjust weights based on performance

2. **Performance Metrics**
   - Track both single-ticker and portfolio metrics
   - Monitor risk-adjusted returns
   - Compare against benchmarks

## Troubleshooting

### Common Issues

1. **Reward Instability**
   - Check for data quality issues
   - Verify reward component calculations
   - Adjust reward smoothing parameters

2. **Poor Learning Performance**
   - Review reward signal strength
   - Check for reward sparsity
   - Consider reward normalization

3. **Memory Issues**
   - Reduce data frequency if needed
   - Implement data chunking
   - Optimize feature calculations

### Debug Steps

1. **Enable Detailed Logging**
   ```python
   logging.getLogger('src.sim.env_intraday_rl').setLevel(logging.DEBUG)
   ```

2. **Check Reward Components**
   ```python
   # In the environment
   reward_components = env.get_last_reward_components()
   print(reward_components)
   ```

3. **Validate Data Quality**
   ```python
   # Check for missing values
   print(data.isnull().sum())
   
   # Check for outliers
   print(data.describe())
   ```

## Conclusion

The reward mechanisms in this RL trading system provide a flexible framework for both single-ticker and multi-ticker trading strategies. The Hybrid2 reward system offers sophisticated reward shaping with multiple components that can be customized based on trading objectives and market conditions.

For optimal results, carefully tune the reward parameters based on backtesting and continuously monitor the reward decomposition to understand the drivers of trading performance.