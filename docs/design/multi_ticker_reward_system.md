# Multi-Ticker Reward System with Cross-Ticker Correlations Design

## Overview

This document outlines the design for an advanced multi-ticker reward system that incorporates cross-ticker correlations and relationships. The reward system will provide sophisticated feedback to the RL agent that considers both individual ticker performance and portfolio-level dynamics.

## Requirements

### Functional Requirements
1. **Multi-Ticker Rewards**: Calculate rewards that consider all tickers in the portfolio
2. **Cross-Ticker Correlations**: Incorporate correlation relationships between tickers
3. **Portfolio-Level Metrics**: Use portfolio-level performance metrics for rewards
4. **Risk-Adjusted Returns**: Provide risk-adjusted reward calculations
5. **Regime Awareness**: Adapt reward calculations based on market regimes
6. **Reward Decomposition**: Allow decomposition of rewards into components
7. **Customizable Reward Functions**: Support configurable reward components

### Non-Functional Requirements
1. **Performance**: Efficient reward calculation for multiple tickers
2. **Stability**: Stable reward signals to aid learning
3. **Interpretability**: Clear, interpretable reward components
4. **Flexibility**: Configurable reward functions and weights
5. **Backward Compatibility**: Maintain compatibility with existing single-ticker reward systems

## Architecture Overview

### Core Components

#### 1. MultiTickerRewardSystem
The main reward system class that coordinates reward calculation across multiple tickers.

```python
class MultiTickerRewardSystem:
    """
    Multi-ticker reward system with cross-ticker correlations.
    
    Calculates rewards that consider both individual ticker performance
    and portfolio-level dynamics including cross-ticker correlations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-ticker reward system.
        
        Args:
            config: Configuration dictionary with reward settings
        """
        self.config = config
        self.reward_components = self._initialize_reward_components()
        self.correlation_manager = CorrelationManager(config.get('correlation', {}))
        self.risk_adjuster = RiskAdjuster(config.get('risk_adjustment', {}))
        self.regime_detector = RegimeDetector(config.get('regime_detection', {}))
        self.reward_decomposer = RewardDecomposer(config.get('decomposition', {}))
        
    def calculate_reward(self, portfolio_state: PortfolioState,
                        market_data: Dict[str, MarketData],
                        action: np.ndarray,
                        prev_portfolio_state: PortfolioState) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for the current state transition.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            action: Action taken by the agent
            prev_portfolio_state: Previous portfolio state
            
        Returns:
            Total reward and dictionary of component rewards
        """
        pass
        
    def get_reward_components(self) -> Dict[str, RewardComponent]:
        """
        Get all reward components.
        
        Returns:
            Dictionary of reward component name -> RewardComponent
        """
        pass
```

#### 2. RewardComponent
Base class for individual reward components.

```python
class RewardComponent(ABC):
    """
    Base class for reward components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward component.
        
        Args:
            config: Component-specific configuration
        """
        self.config = config
        self.weight = config.get('weight', 1.0)
        self.enabled = config.get('enabled', True)
        
    @abstractmethod
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 action: np.ndarray,
                 prev_portfolio_state: PortfolioState) -> float:
        """
        Calculate component reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data
            action: Action taken
            prev_portfolio_state: Previous portfolio state
            
        Returns:
            Component reward value
        """
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """
        Get component name.
        
        Returns:
            Component name
        """
        pass
```

#### 3. CorrelationManager
Component for managing cross-ticker correlations in reward calculations.

```python
class CorrelationManager:
    """
    Cross-ticker correlation manager for reward calculations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize correlation manager.
        
        Args:
            config: Correlation configuration
        """
        self.config = config
        self.correlation_window = config.get('window', 20)
        self.correlation_method = config.get('method', 'pearson')  # pearson, spearman, kendall
        self.correlation_cache = {}
        
    def calculate_correlation_matrix(self, returns: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate correlation matrix for ticker returns.
        
        Args:
            returns: Dictionary of ticker -> return series
            
        Returns:
            Correlation matrix
        """
        pass
        
    def calculate_diversification_reward(self, portfolio_state: PortfolioState,
                                        correlation_matrix: np.ndarray) -> float:
        """
        Calculate reward based on portfolio diversification.
        
        Args:
            portfolio_state: Current portfolio state
            correlation_matrix: Correlation matrix
            
        Returns:
            Diversification reward
        """
        pass
        
    def calculate_concentration_penalty(self, portfolio_state: PortfolioState,
                                       correlation_matrix: np.ndarray) -> float:
        """
        Calculate penalty for portfolio concentration.
        
        Args:
            portfolio_state: Current portfolio state
            correlation_matrix: Correlation matrix
            
        Returns:
            Concentration penalty
        """
        pass
```

#### 4. RiskAdjuster
Component for risk-adjusting reward calculations.

```python
class RiskAdjuster:
    """
    Risk adjustment component for reward calculations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk adjuster.
        
        Args:
            config: Risk adjustment configuration
        """
        self.config = config
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.risk_aversion = config.get('risk_aversion', 1.0)
        self.drawdown_penalty = config.get('drawdown_penalty', 1.0)
        
    def calculate_sharpe_ratio_reward(self, portfolio_state: PortfolioState,
                                    prev_portfolio_state: PortfolioState) -> float:
        """
        Calculate reward based on Sharpe ratio improvement.
        
        Args:
            portfolio_state: Current portfolio state
            prev_portfolio_state: Previous portfolio state
            
        Returns:
            Sharpe ratio reward
        """
        pass
        
    def calculate_sortino_ratio_reward(self, portfolio_state: PortfolioState,
                                      prev_portfolio_state: PortfolioState) -> float:
        """
        Calculate reward based on Sortino ratio improvement.
        
        Args:
            portfolio_state: Current portfolio state
            prev_portfolio_state: Previous portfolio state
            
        Returns:
            Sortino ratio reward
        """
        pass
        
    def calculate_drawdown_penalty(self, portfolio_state: PortfolioState,
                                 prev_portfolio_state: PortfolioState) -> float:
        """
        Calculate penalty for drawdown.
        
        Args:
            portfolio_state: Current portfolio state
            prev_portfolio_state: Previous portfolio state
            
        Returns:
            Drawdown penalty
        """
        pass
```

#### 5. RegimeDetector
Component for detecting market regimes and adjusting rewards accordingly.

```python
class RegimeDetector:
    """
    Market regime detector for reward calculations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize regime detector.
        
        Args:
            config: Regime detection configuration
        """
        self.config = config
        self.regime_window = config.get('window', 20)
        self.regime_thresholds = config.get('thresholds', {})
        self.current_regime = 'normal'
        
    def detect_regime(self, market_data: Dict[str, MarketData]) -> str:
        """
        Detect current market regime.
        
        Args:
            market_data: Current market data
            
        Returns:
            Current regime ('normal', 'volatile', 'trending', 'ranging')
        """
        pass
        
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Get reward component weights for the current regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary of component name -> weight
        """
        pass
```

#### 6. RewardDecomposer
Component for decomposing rewards into interpretable components.

```python
class RewardDecomposer:
    """
    Reward decomposition component for interpretability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward decomposer.
        
        Args:
            config: Decomposition configuration
        """
        self.config = config
        self.decomposition_history = []
        self.component_contributions = {}
        
    def decompose_reward(self, total_reward: float,
                        component_rewards: Dict[str, float]) -> Dict[str, Any]:
        """
        Decompose total reward into components.
        
        Args:
            total_reward: Total reward
            component_rewards: Dictionary of component rewards
            
        Returns:
            Decomposition dictionary
        """
        pass
        
    def get_contribution_analysis(self) -> Dict[str, Any]:
        """
        Get analysis of component contributions over time.
        
        Returns:
            Contribution analysis dictionary
        """
        pass
```

### Reward Components

#### 1. PortfolioPNLComponent
Reward component based on portfolio P&L.

```python
class PortfolioPNLComponent(RewardComponent):
    """
    Portfolio P&L reward component.
    """
    
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 action: np.ndarray,
                 prev_portfolio_state: PortfolioState) -> float:
        """
        Calculate P&L reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data
            action: Action taken
            prev_portfolio_state: Previous portfolio state
            
        Returns:
            P&L reward
        """
        pnl_change = portfolio_state.unrealized_pnl - prev_portfolio_state.unrealized_pnl
        return pnl_change * self.weight
        
    def get_name(self) -> str:
        return "portfolio_pnl"
```

#### 2. SharpeRatioComponent
Reward component based on Sharpe ratio.

```python
class SharpeRatioComponent(RewardComponent):
    """
    Sharpe ratio reward component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.return_history = []
        self.window = config.get('window', 20)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 action: np.ndarray,
                 prev_portfolio_state: PortfolioState) -> float:
        """
        Calculate Sharpe ratio reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data
            action: Action taken
            prev_portfolio_state: Previous portfolio state
            
        Returns:
            Sharpe ratio reward
        """
        # Calculate current return
        current_return = (portfolio_state.total_value - prev_portfolio_state.total_value) / prev_portfolio_state.total_value
        self.return_history.append(current_return)
        
        # Keep only recent history
        if len(self.return_history) > self.window:
            self.return_history.pop(0)
            
        # Calculate Sharpe ratio
        if len(self.return_history) > 1:
            returns = np.array(self.return_history)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            return sharpe_ratio * self.weight
        else:
            return 0.0
            
    def get_name(self) -> str:
        return "sharpe_ratio"
```

#### 3. DiversificationComponent
Reward component based on portfolio diversification.

```python
class DiversificationComponent(RewardComponent):
    """
    Diversification reward component.
    """
    
    def __init__(self, config: Dict[str, Any], correlation_manager: CorrelationManager):
        super().__init__(config)
        self.correlation_manager = correlation_manager
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 action: np.ndarray,
                 prev_portfolio_state: PortfolioState) -> float:
        """
        Calculate diversification reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data
            action: Action taken
            prev_portfolio_state: Previous portfolio state
            
        Returns:
            Diversification reward
        """
        # Calculate returns for correlation
        returns = {}
        for ticker, position in portfolio_state.positions.items():
            if ticker in market_data:
                current_price = market_data[ticker].close
                prev_price = position.average_price
                if prev_price > 0:
                    returns[ticker] = (current_price - prev_price) / prev_price
                    
        # Calculate correlation matrix
        if len(returns) > 1:
            correlation_matrix = self.correlation_manager.calculate_correlation_matrix(returns)
            diversification_reward = self.correlation_manager.calculate_diversification_reward(
                portfolio_state, correlation_matrix
            )
            return diversification_reward * self.weight
        else:
            return 0.0
            
    def get_name(self) -> str:
        return "diversification"
```

#### 4. TransactionCostComponent
Penalty component for transaction costs.

```python
class TransactionCostComponent(RewardComponent):
    """
    Transaction cost penalty component.
    """
    
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 action: np.ndarray,
                 prev_portfolio_state: PortfolioState) -> float:
        """
        Calculate transaction cost penalty.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data
            action: Action taken
            prev_portfolio_state: Previous portfolio state
            
        Returns:
            Transaction cost penalty
        """
        # Calculate transaction costs since last state
        total_costs = 0.0
        for ticker, position in portfolio_state.positions.items():
            if ticker in prev_portfolio_state.positions:
                prev_position = prev_portfolio_state.positions[ticker]
                # Calculate costs for trades
                for trade in position.trades:
                    if trade.timestamp > prev_position.trades[-1].timestamp:
                        total_costs += trade.commission + trade.slippage
                        
        # Return negative reward (penalty)
        return -total_costs * self.weight
        
    def get_name(self) -> str:
        return "transaction_cost"
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MultiTicker     │    │ Reward          │    │ Portfolio       │
│ RewardSystem    │───▶│ Components      │───▶│ State          │
│ (Coordinator)   │    │ (Calculation)   │    │ (Input)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      │                      │
         │                      ▼                      ▼
         │                ┌─────────────────┐    ┌─────────────────┐
         │                │ Correlation     │    │ Market Data     │
         │                │ Manager         │    │ (Input)         │
         │                └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      │                      │
         │                      ▼                      ▼
         │                ┌─────────────────┐    ┌─────────────────┐
         │                │ Risk Adjuster   │    │ Action          │
         │                │ (Risk Adj)      │    │ (Input)         │
         │                └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      │                      │
         │                      ▼                      ▼
         │                ┌─────────────────┐    ┌─────────────────┐
         └────────────────▶│ Regime Detector │    │ Reward History  │
                          │ (Regime Adj)    │    │ (Output)        │
                          └─────────────────┘    └─────────────────┘
                                 │
                                 ▼
                          ┌─────────────────┐
                          │ Reward          │
                          │ Decomposer      │
                          │ (Analysis)      │
                          └─────────────────┘
```

## Configuration Structure

### Multi-Ticker Reward Configuration

```yaml
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
      optimal_correlation: -0.2  # Slightly negative correlation is optimal
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
      asymmetric_penalty: true  # Higher penalty for drawdowns
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
    regime_persistence: 3  # Number of periods to confirm regime change
  
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
    alpha: 0.2  # For exponential smoothing
```

## Implementation Details

### Cross-Ticker Correlation Integration

#### 1. Correlation Calculation
- Calculate rolling correlations between ticker returns
- Support multiple correlation methods (Pearson, Spearman, Kendall)
- Cache correlation matrices for efficiency
- Update correlations at configurable intervals

#### 2. Diversification Reward
- Reward portfolios with low correlation between positions
- Optimal correlation slightly negative for diversification
- Scale reward based on distance from optimal correlation
- Consider position weights in correlation calculation

#### 3. Concentration Penalty
- Penalize portfolios with high correlation between positions
- Apply higher penalties for correlations above threshold
- Consider both pairwise and portfolio-level concentration
- Scale penalty based on position sizes

### Risk-Adjusted Rewards

#### 1. Sharpe Ratio Integration
- Calculate rolling Sharpe ratio for portfolio
- Reward improvements in Sharpe ratio
- Consider both absolute and relative Sharpe ratio
- Adjust reward based on risk-free rate

#### 2. Drawdown Penalty
- Apply asymmetric penalty for drawdowns
- Higher penalty for larger drawdowns
- Consider both current and maximum drawdown
- Scale penalty based on risk aversion

#### 3. Volatility Adjustment
- Adjust rewards based on portfolio volatility
- Penalize excessive volatility
- Reward consistent returns
- Consider target volatility levels

### Regime-Aware Rewards

#### 1. Regime Detection
- Detect market regimes based on volatility and trend
- Use multiple detection methods for robustness
- Implement regime persistence to avoid whipsaws
- Update regime weights smoothly

#### 2. Regime-Specific Weights
- Adjust component weights based on current regime
- Emphasize different metrics in different regimes
- Smooth transitions between regime weights
- Allow custom weight configurations

#### 3. Regime Adaptation
- Adapt reward calculations to market conditions
- Consider regime-specific risk preferences
- Adjust time horizons based on regime
- Implement regime-specific constraints

### Reward Decomposition

#### 1. Component Tracking
- Track individual component contributions over time
- Maintain history of component rewards
- Calculate component statistics
- Identify dominant reward components

#### 2. Contribution Analysis
- Analyze correlation between components
- Identify redundant components
- Detect component interactions
- Provide interpretability insights

#### 3. Regime Analysis
- Analyze component behavior across regimes
- Identify regime-specific patterns
- Provide regime-specific insights
- Support regime-based optimization

## Performance Considerations

### Computational Efficiency
1. **Vectorization**: Use vectorized operations for correlation calculations
2. **Caching**: Cache correlation matrices and other expensive calculations
3. **Incremental Updates**: Update rewards incrementally where possible
4. **Parallel Processing**: Process multiple components in parallel

### Memory Optimization
1. **History Management**: Limit history length for rolling calculations
2. **Data Structures**: Use efficient data structures for correlation matrices
3. **Memory Pooling**: Reuse memory objects for reward calculations
4. **Garbage Collection**: Implement proper memory management

### Numerical Stability
1. **Numerical Methods**: Use numerically stable algorithms
2. **Regularization**: Apply regularization where needed
3. **Clipping**: Clip extreme reward values
4. **Smoothing**: Apply smoothing to reduce noise

## Testing Strategy

### Unit Tests
1. **Individual Components**: Test each reward component in isolation
2. **Correlation Calculations**: Test correlation matrix calculations
3. **Risk Adjustments**: Test risk adjustment methods
4. **Regime Detection**: Test regime detection logic

### Integration Tests
1. **Reward System**: Test complete reward system
2. **Multi-Ticker**: Test with multiple tickers
3. **Regime Transitions**: Test regime weight transitions
4. **Decomposition**: Test reward decomposition

### Performance Tests
1. **Scalability**: Test with increasing numbers of tickers
2. **Computation Time**: Measure reward calculation performance
3. **Memory Usage**: Monitor memory consumption
4. **Numerical Stability**: Test numerical stability

## Migration Path

### Phase 1: Backward Compatibility
1. Maintain existing single-ticker reward system
2. Add multi-ticker functionality as an option
3. Ensure existing tests continue to pass

### Phase 2: Coexistence
1. Support both single and multi-ticker reward systems
2. Provide migration utilities
3. Update documentation and examples

### Phase 3: Multi-Ticker First
1. Make multi-ticker the primary reward system
2. Maintain single-ticker compatibility layer
3. Deprecate single-ticker specific features

## Future Enhancements

### Advanced Correlation Models
1. **Dynamic Correlation**: Time-varying correlation models
2. **Non-linear Correlation**: Non-linear dependence measures
3. **Lead-Lag Relationships**: Lead-lag correlation analysis
4. **Regime-Specific Correlation**: Correlation patterns by regime

### Advanced Risk Models
1. **Conditional Value at Risk (CVaR)**: CVaR-based risk adjustment
2. **Expected Shortfall**: Expected shortfall calculations
3. **Tail Risk Measures**: Tail risk reward adjustments
4. **Liquidity Risk**: Liquidity risk considerations

### Machine Learning Integration
1. **Learned Reward Functions**: ML-based reward function learning
2. **Adaptive Weights**: Adaptive component weight optimization
3. **Meta-Learning**: Meta-learning for reward function adaptation
4. **Reinforcement Learning**: RL for reward function optimization

## Conclusion

The multi-ticker reward system with cross-ticker correlations provides a sophisticated framework for calculating rewards that consider both individual ticker performance and portfolio-level dynamics. By incorporating cross-ticker correlations, risk adjustments, regime awareness, and reward decomposition, we can provide more informative and stable reward signals to the RL agent.

The design emphasizes modularity, with clear separation of concerns between reward components, correlation management, risk adjustment, and regime detection. This approach allows for independent development and testing of each component while ensuring they work together seamlessly.

The architecture is designed to scale to support 10+ tickers with complex correlation relationships, sophisticated risk management, and regime-aware reward calculations. This provides a solid foundation for developing advanced multi-ticker trading strategies.