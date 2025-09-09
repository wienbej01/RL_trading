# Advanced Hybrid2 Reward with Regime Weighting

## Overview

This document outlines the design for an advanced hybrid2 reward system with regime weighting for the multi-ticker RL trading system. The hybrid2 reward system combines multiple reward components to create a comprehensive reward signal that encourages desirable trading behaviors while managing risk.

## Requirements

### Functional Requirements
1. **Multi-Component Reward**: Combine multiple reward components into a single reward signal
2. **Regime Awareness**: Adjust reward weights based on market regimes
3. **Portfolio-Level Optimization**: Optimize for portfolio-level metrics rather than individual tickers
4. **Risk-Adjusted Returns**: Incorporate risk adjustments into the reward calculation
5. **Transaction Cost Awareness**: Account for transaction costs in the reward signal
6. **Diversification Incentives**: Encourage portfolio diversification through reward shaping
7. **Drawdown Control**: Penalize excessive drawdowns with asymmetric penalties

### Non-Functional Requirements
1. **Performance**: Efficient reward calculation with minimal computational overhead
2. **Stability**: Stable reward signals that don't introduce excessive noise
3. **Interpretability**: Clear, interpretable reward components
4. **Configurability**: Flexible configuration of reward weights and parameters
5. **Adaptability**: Adaptive to changing market conditions

## Architecture Overview

### Core Components

#### 1. MultiTickerRewardCalculator
The main class responsible for calculating the hybrid2 reward for multi-ticker trading.

```python
class MultiTickerRewardCalculator:
    """
    Advanced hybrid2 reward calculator for multi-ticker trading.
    
    Combines multiple reward components with regime-aware weighting
    to create a comprehensive reward signal.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-ticker reward calculator.
        
        Args:
            config: Configuration dictionary with reward settings
        """
        self.config = config
        self.reward_components = self._initialize_reward_components()
        self.regime_detector = RegimeDetector(config.get('regime_detection', {}))
        self.reward_normalizer = RewardNormalizer(config.get('normalization', {}))
        self.reward_smoother = RewardSmoother(config.get('smoothing', {}))
        self.reward_history = []
        self.component_history = {}
        
    def calculate_reward(self, portfolio_state: PortfolioState,
                        market_data: Dict[str, MarketData],
                        actions: Dict[str, float],
                        previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate the hybrid2 reward for the current step.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Hybrid2 reward value
        """
        pass
        
    def get_reward_components(self) -> Dict[str, float]:
        """
        Get the individual reward components.
        
        Returns:
            Dictionary of component name -> value
        """
        pass
        
    def get_regime_weights(self) -> Dict[str, float]:
        """
        Get the current regime-based weights.
        
        Returns:
            Dictionary of component name -> weight
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
        self.name = self.get_name()
        self.normalization = config.get('normalization', {})
        
    @abstractmethod
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate the component reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
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
        
    def normalize(self, value: float) -> float:
        """
        Normalize the component value.
        
        Args:
            value: Raw component value
            
        Returns:
            Normalized value
        """
        if not self.normalization.get('enabled', False):
            return value
            
        method = self.normalization.get('method', 'standard')
        
        if method == 'standard':
            # Standard normalization (z-score)
            mean = self.normalization.get('mean', 0.0)
            std = self.normalization.get('std', 1.0)
            return (value - mean) / std if std > 0 else value
        elif method == 'min_max':
            # Min-max normalization
            min_val = self.normalization.get('min', 0.0)
            max_val = self.normalization.get('max', 1.0)
            return (value - min_val) / (max_val - min_val) if max_val > min_val else value
        elif method == 'clip':
            # Clipping
            min_val = self.normalization.get('min', -10.0)
            max_val = self.normalization.get('max', 10.0)
            return np.clip(value, min_val, max_val)
        else:
            return value
```

#### 3. RegimeDetector
Component for detecting market regimes and adjusting reward weights.

```python
class RegimeDetector:
    """
    Market regime detector for reward weight adjustment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize regime detector.
        
        Args:
            config: Regime detection configuration
        """
        self.config = config
        self.detection_method = config.get('detection_method', 'volatility_trend')
        self.window = config.get('window', 20)
        self.update_frequency = config.get('update_frequency', 5)
        self.thresholds = config.get('thresholds', {})
        self.current_regime = 'normal'
        self.regime_history = []
        self.regime_persistence = config.get('regime_persistence', 3)
        self.regime_counter = 0
        
    def detect_regime(self, market_data: Dict[str, MarketData],
                     portfolio_state: PortfolioState) -> str:
        """
        Detect the current market regime.
        
        Args:
            market_data: Current market data for all tickers
            portfolio_state: Current portfolio state
            
        Returns:
            Current regime name
        """
        pass
        
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Get reward component weights for the current regime.
        
        Args:
            regime: Current regime name
            
        Returns:
            Dictionary of component name -> weight
        """
        pass
```

#### 4. RewardNormalizer
Component for normalizing reward values.

```python
class RewardNormalizer:
    """
    Reward normalizer for stable reward signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward normalizer.
        
        Args:
            config: Normalization configuration
        """
        self.config = config
        self.method = config.get('method', 'rolling')
        self.window = config.get('window', 20)
        self.min_periods = config.get('min_periods', 10)
        self.clip_outliers = config.get('clip_outliers', True)
        self.clip_threshold = config.get('clip_threshold', 3.0)
        self.reward_history = []
        
    def normalize(self, reward: float) -> float:
        """
        Normalize the reward value.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Normalized reward value
        """
        pass
```

#### 5. RewardSmoother
Component for smoothing reward signals.

```python
class RewardSmoother:
    """
    Reward smoother for reducing noise in reward signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward smoother.
        
        Args:
            config: Smoothing configuration
        """
        self.config = config
        self.method = config.get('method', 'exponential')
        self.window = config.get('window', 5)
        self.alpha = config.get('alpha', 0.2)
        self.reward_history = []
        
    def smooth(self, reward: float) -> float:
        """
        Smooth the reward value.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Smoothed reward value
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
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaling = config.get('scaling', 'linear')
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate portfolio P&L reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Portfolio P&L reward
        """
        # Calculate portfolio P&L
        portfolio_pnl = portfolio_state.total_value - previous_portfolio_state.total_value
        
        # Normalize by portfolio value
        normalized_pnl = portfolio_pnl / previous_portfolio_state.total_value
        
        # Apply scaling
        if self.scaling == 'linear':
            reward = normalized_pnl
        elif self.scaling == 'log':
            reward = np.sign(normalized_pnl) * np.log(1 + abs(normalized_pnl))
        elif self.scaling == 'sqrt':
            reward = np.sign(normalized_pnl) * np.sqrt(abs(normalized_pnl))
        else:
            reward = normalized_pnl
            
        return self.normalize(reward)
        
    def get_name(self) -> str:
        return "portfolio_pnl"
```

#### 2. SharpeRatioComponent
Reward component based on portfolio Sharpe ratio.

```python
class SharpeRatioComponent(RewardComponent):
    """
    Sharpe ratio reward component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window = config.get('window', 20)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.annualization_factor = config.get('annualization_factor', 252)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate Sharpe ratio reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Sharpe ratio reward
        """
        # Get portfolio returns history
        returns_history = portfolio_state.returns_history
        
        if len(returns_history) < self.window:
            return 0.0  # Not enough data
            
        # Calculate recent returns
        recent_returns = returns_history[-self.window:]
        
        # Calculate Sharpe ratio
        excess_returns = np.array(recent_returns) - self.risk_free_rate / self.annualization_factor
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.annualization_factor)
        
        return self.normalize(sharpe_ratio)
        
    def get_name(self) -> str:
        return "sharpe_ratio"
```

#### 3. SortinoRatioComponent
Reward component based on portfolio Sortino ratio.

```python
class SortinoRatioComponent(RewardComponent):
    """
    Sortino ratio reward component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window = config.get('window', 20)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.annualization_factor = config.get('annualization_factor', 252)
        self.mar = config.get('minimum_acceptable_return', 0.0)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate Sortino ratio reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Sortino ratio reward
        """
        # Get portfolio returns history
        returns_history = portfolio_state.returns_history
        
        if len(returns_history) < self.window:
            return 0.0  # Not enough data
            
        # Calculate recent returns
        recent_returns = returns_history[-self.window:]
        
        # Calculate downside returns
        excess_returns = np.array(recent_returns) - self.risk_free_rate / self.annualization_factor
        downside_returns = excess_returns[excess_returns < self.mar]
        
        if len(downside_returns) == 0:
            return 10.0  # Perfect Sortino ratio if no downside
            
        # Calculate Sortino ratio
        downside_deviation = np.std(downside_returns)
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(self.annualization_factor)
        
        return self.normalize(sortino_ratio)
        
    def get_name(self) -> str:
        return "sortino_ratio"
```

#### 4. DiversificationComponent
Reward component based on portfolio diversification.

```python
class DiversificationComponent(RewardComponent):
    """
    Diversification reward component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.correlation_window = config.get('correlation_window', 20)
        self.optimal_correlation = config.get('optimal_correlation', -0.2)
        self.correlation_method = config.get('correlation_method', 'pearson')
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate diversification reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Diversification reward
        """
        # Get portfolio positions
        positions = portfolio_state.positions
        
        if len(positions) < 2:
            return 0.0  # No diversification with single position
            
        # Get returns for each position
        returns_matrix = []
        for ticker in positions:
            if ticker in market_data and hasattr(market_data[ticker], 'returns_history'):
                returns = market_data[ticker].returns_history[-self.correlation_window:]
                if len(returns) == self.correlation_window:
                    returns_matrix.append(returns)
                    
        if len(returns_matrix) < 2:
            return 0.0  # Not enough data
            
        # Calculate correlation matrix
        returns_matrix = np.array(returns_matrix).T
        if self.correlation_method == 'pearson':
            corr_matrix = np.corrcoef(returns_matrix)
        elif self.correlation_method == 'spearman':
            corr_matrix = np.corrcoef(rankdata(returns_matrix, axis=0))
        else:
            corr_matrix = np.corrcoef(returns_matrix)
            
        # Calculate average correlation (excluding diagonal)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        avg_correlation = np.mean(corr_matrix[mask])
        
        # Calculate diversification reward (lower correlation = higher reward)
        correlation_diff = self.optimal_correlation - avg_correlation
        reward = -correlation_diff  # Negative because we want low correlation
        
        return self.normalize(reward)
        
    def get_name(self) -> str:
        return "diversification"
```

#### 5. TransactionCostComponent
Reward component based on transaction costs.

```python
class TransactionCostComponent(RewardComponent):
    """
    Transaction cost reward component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.include_commission = config.get('include_commission', True)
        self.include_slippage = config.get('include_slippage', True)
        self.include_market_impact = config.get('include_market_impact', True)
        self.commission_rate = config.get('commission_rate', 0.001)
        self.slippage_rate = config.get('slippage_rate', 0.0005)
        self.market_impact_factor = config.get('market_impact_factor', 0.1)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate transaction cost reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Transaction cost reward (negative for costs)
        """
        total_cost = 0.0
        
        for ticker, action in actions.items():
            if ticker not in market_data:
                continue
                
            # Calculate trade size
            previous_position = previous_portfolio_state.positions.get(ticker, 0.0)
            current_position = portfolio_state.positions.get(ticker, 0.0)
            trade_size = abs(current_position - previous_position)
            
            if trade_size == 0:
                continue
                
            # Get current price
            current_price = market_data[ticker].close
            
            # Calculate commission
            if self.include_commission:
                commission = trade_size * current_price * self.commission_rate
                total_cost += commission
                
            # Calculate slippage
            if self.include_slippage:
                slippage = trade_size * current_price * self.slippage_rate
                total_cost += slippage
                
            # Calculate market impact
            if self.include_market_impact:
                # Simple market impact model: impact = factor * sqrt(trade_size)
                market_impact = self.market_impact_factor * np.sqrt(trade_size) * current_price
                total_cost += market_impact
                
        # Normalize by portfolio value
        normalized_cost = total_cost / previous_portfolio_state.total_value
        
        # Return negative cost (penalty)
        return self.normalize(-normalized_cost)
        
    def get_name(self) -> str:
        return "transaction_cost"
```

#### 6. DrawdownPenaltyComponent
Reward component for drawdown penalty.

```python
class DrawdownPenaltyComponent(RewardComponent):
    """
    Drawdown penalty reward component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_drawdown = config.get('max_drawdown', 0.1)
        self.penalty_factor = config.get('penalty_factor', 2.0)
        self.asymmetric_penalty = config.get('asymmetric_penalty', True)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate drawdown penalty reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Drawdown penalty reward (negative for penalty)
        """
        # Calculate current drawdown
        current_drawdown = portfolio_state.drawdown
        
        if current_drawdown <= 0:
            return 0.0  # No penalty if no drawdown
            
        # Calculate penalty
        if self.asymmetric_penalty:
            # Asymmetric penalty: higher penalty for larger drawdowns
            penalty = self.penalty_factor * (current_drawdown / self.max_drawdown) ** 2
        else:
            # Linear penalty
            penalty = self.penalty_factor * (current_drawdown / self.max_drawdown)
            
        # Apply penalty only if drawdown exceeds threshold
        if current_drawdown > self.max_drawdown:
            penalty *= (current_drawdown / self.max_drawdown)
            
        # Return negative penalty
        return self.normalize(-penalty)
        
    def get_name(self) -> str:
        return "drawdown_penalty"
```

#### 7. TurnoverPenaltyComponent
Reward component for turnover penalty.

```python
class TurnoverPenaltyComponent(RewardComponent):
    """
    Turnover penalty reward component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_turnover = config.get('max_turnover', 0.5)
        self.penalty_factor = config.get('penalty_factor', 1.0)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate turnover penalty reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Turnover penalty reward (negative for penalty)
        """
        # Calculate turnover
        turnover = 0.0
        
        for ticker, action in actions.items():
            if ticker not in market_data:
                continue
                
            # Calculate trade size
            previous_position = previous_portfolio_state.positions.get(ticker, 0.0)
            current_position = portfolio_state.positions.get(ticker, 0.0)
            trade_size = abs(current_position - previous_position)
            
            # Get current price
            current_price = market_data[ticker].close
            
            # Add to turnover
            turnover += trade_size * current_price
            
        # Normalize by portfolio value
        normalized_turnover = turnover / previous_portfolio_state.total_value
        
        # Calculate penalty
        if normalized_turnover > self.max_turnover:
            penalty = self.penalty_factor * (normalized_turnover / self.max_turnover - 1.0)
        else:
            penalty = 0.0
            
        # Return negative penalty
        return self.normalize(-penalty)
        
    def get_name(self) -> str:
        return "turnover_penalty"
```

#### 8. RiskAdjustedReturnComponent
Reward component based on risk-adjusted returns.

```python
class RiskAdjustedReturnComponent(RewardComponent):
    """
    Risk-adjusted return reward component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.risk_aversion = config.get('risk_aversion', 1.0)
        self.window = config.get('window', 20)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate risk-adjusted return reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Risk-adjusted return reward
        """
        # Get portfolio returns history
        returns_history = portfolio_state.returns_history
        
        if len(returns_history) < self.window:
            return 0.0  # Not enough data
            
        # Calculate recent returns
        recent_returns = returns_history[-self.window:]
        
        # Calculate expected return
        expected_return = np.mean(recent_returns)
        
        # Calculate risk (volatility)
        risk = np.std(recent_returns)
        
        # Calculate risk-adjusted return
        risk_adjusted_return = expected_return - 0.5 * self.risk_aversion * risk ** 2
        
        return self.normalize(risk_adjusted_return)
        
    def get_name(self) -> str:
        return "risk_adjusted_return"
```

### Regime Detection

#### Volatility-Trend Regime Detection

```python
def detect_volatility_trend_regime(self, market_data: Dict[str, MarketData],
                                  portfolio_state: PortfolioState) -> str:
    """
    Detect regime based on volatility and trend.
    
    Args:
        market_data: Current market data for all tickers
        portfolio_state: Current portfolio state
        
    Returns:
        Current regime name
    """
    # Calculate portfolio volatility
    returns_history = portfolio_state.returns_history
    if len(returns_history) < self.window:
        return 'normal'
        
    recent_returns = returns_history[-self.window:]
    volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
    
    # Calculate trend strength
    portfolio_values = portfolio_state.value_history[-self.window:]
    if len(portfolio_values) < 2:
        return 'normal'
        
    x = np.arange(len(portfolio_values))
    slope, _ = np.polyfit(x, portfolio_values, 1)
    trend_strength = slope / np.mean(portfolio_values) * 252  # Annualized
    
    # Determine regime based on thresholds
    high_vol_threshold = self.thresholds.get('volatility_high', 0.25)
    low_vol_threshold = self.thresholds.get('volatility_low', 0.1)
    high_trend_threshold = self.thresholds.get('trend_strength_high', 0.7)
    low_trend_threshold = self.thresholds.get('trend_strength_low', 0.3)
    
    if volatility > high_vol_threshold:
        if abs(trend_strength) > high_trend_threshold:
            return 'volatile_trending'
        else:
            return 'volatile'
    elif volatility < low_vol_threshold:
        if abs(trend_strength) < low_trend_threshold:
            return 'ranging'
        else:
            return 'normal'
    else:
        if abs(trend_strength) > high_trend_threshold:
            return 'trending'
        else:
            return 'normal'
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Portfolio       │    │ Market Data     │    │ Actions         │
│ State           │    │ (Input)         │    │ (Input)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MultiTicker     │    │ Reward          │    │ Previous        │
│ Reward          │    │ Components      │    │ Portfolio       │
│ Calculator      │◀───│ (Scoring)       │◀───│ State           │
│ (Coordinator)   │    └─────────────────┘    └─────────────────┘
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Regime          │
│ Detector        │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Reward          │
│ Normalizer      │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Reward          │
│ Smoother        │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Hybrid2         │
│ Reward          │
│ (Output)        │
└─────────────────┘
```

## Configuration Structure

### Hybrid2 Reward Configuration

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
      normalization:
        enabled: true
        method: clip  # standard, min_max, clip
        min: -0.1
        max: 0.1
        
    sharpe_ratio:
      enabled: true
      weight: 0.5
      window: 20
      risk_free_rate: 0.02
      annualization_factor: 252
      normalization:
        enabled: true
        method: standard
        mean: 0.0
        std: 1.0
        
    sortino_ratio:
      enabled: true
      weight: 0.3
      window: 20
      risk_free_rate: 0.02
      annualization_factor: 252
      minimum_acceptable_return: 0.0
      normalization:
        enabled: true
        method: standard
        mean: 0.0
        std: 1.0
        
    diversification:
      enabled: true
      weight: 0.4
      correlation_window: 20
      optimal_correlation: -0.2
      correlation_method: pearson  # pearson, spearman
      normalization:
        enabled: true
        method: min_max
        min: -1.0
        max: 1.0
        
    transaction_cost:
      enabled: true
      weight: 0.5
      include_commission: true
      include_slippage: true
      include_market_impact: true
      commission_rate: 0.001
      slippage_rate: 0.0005
      market_impact_factor: 0.1
      normalization:
        enabled: true
        method: clip
        min: -0.05
        max: 0.0
        
    drawdown_penalty:
      enabled: true
      weight: 0.3
      max_drawdown: 0.1
      penalty_factor: 2.0
      asymmetric_penalty: true
      normalization:
        enabled: true
        method: clip
        min: -1.0
        max: 0.0
        
    turnover_penalty:
      enabled: true
      weight: 0.2
      max_turnover: 0.5
      penalty_factor: 1.0
      normalization:
        enabled: true
        method: clip
        min: -0.5
        max: 0.0
        
    risk_adjusted_return:
      enabled: true
      weight: 0.4
      risk_free_rate: 0.02
      risk_aversion: 1.0
      window: 20
      normalization:
        enabled: true
        method: standard
        mean: 0.0
        std: 0.1
  
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
    regime_weights:
      normal:
        portfolio_pnl: 1.0
        sharpe_ratio: 0.5
        sortino_ratio: 0.3
        diversification: 0.4
        transaction_cost: 0.5
        drawdown_penalty: 0.3
        turnover_penalty: 0.2
        risk_adjusted_return: 0.4
      volatile:
        portfolio_pnl: 0.5
        sharpe_ratio: 1.0
        sortino_ratio: 0.8
        diversification: 0.8
        transaction_cost: 0.3
        drawdown_penalty: 0.6
        turnover_penalty: 0.1
        risk_adjusted_return: 0.7
      trending:
        portfolio_pnl: 1.2
        sharpe_ratio: 0.3
        sortino_ratio: 0.2
        diversification: 0.2
        transaction_cost: 0.6
        drawdown_penalty: 0.2
        turnover_penalty: 0.4
        risk_adjusted_return: 0.3
      ranging:
        portfolio_pnl: 0.3
        sharpe_ratio: 0.8
        sortino_ratio: 0.6
        diversification: 0.6
        transaction_cost: 0.4
        drawdown_penalty: 0.4
        turnover_penalty: 0.3
        risk_adjusted_return: 0.5
  
  # Reward normalization and scaling
  normalization:
    enabled: true
    method: rolling  # standard, robust, rolling, quantile
    window: 20
    min_periods: 10
    clip_outliers: true
    clip_threshold: 3.0
    
  # Reward smoothing
  smoothing:
    enabled: true
    method: exponential  # exponential, moving_average, none
    window: 5
    alpha: 0.2
```

## Implementation Details

### Reward Calculation Algorithm

```python
def calculate_reward(self, portfolio_state: PortfolioState,
                    market_data: Dict[str, MarketData],
                    actions: Dict[str, float],
                    previous_portfolio_state: PortfolioState) -> float:
    """
    Calculate the hybrid2 reward for the current step.
    
    Args:
        portfolio_state: Current portfolio state
        market_data: Current market data for all tickers
        actions: Actions taken for each ticker
        previous_portfolio_state: Previous portfolio state
        
    Returns:
        Hybrid2 reward value
    """
    # Detect current regime
    current_regime = self.regime_detector.detect_regime(market_data, portfolio_state)
    
    # Get regime-specific weights
    regime_weights = self.regime_detector.get_regime_weights(current_regime)
    
    # Calculate individual reward components
    component_rewards = {}
    for component_name, component in self.reward_components.items():
        if component.enabled:
            reward = component.calculate(
                portfolio_state, market_data, actions, previous_portfolio_state
            )
            component_rewards[component_name] = reward
    
    # Apply regime weights
    weighted_rewards = {}
    for component_name, reward in component_rewards.items():
        weight = regime_weights.get(component_name, component.weight)
        weighted_rewards[component_name] = reward * weight
    
    # Calculate total reward
    total_reward = sum(weighted_rewards.values())
    
    # Normalize total reward
    normalized_reward = self.reward_normalizer.normalize(total_reward)
    
    # Smooth reward
    smoothed_reward = self.reward_smoother.smooth(normalized_reward)
    
    # Store history
    self.reward_history.append(smoothed_reward)
    self.component_history[current_regime] = component_rewards.copy()
    
    return smoothed_reward
```

### Regime Weight Application

```python
def apply_regime_weights(self, component_rewards: Dict[str, float],
                        regime: str) -> Dict[str, float]:
    """
    Apply regime-specific weights to reward components.
    
    Args:
        component_rewards: Dictionary of component name -> reward value
        regime: Current regime name
        
    Returns:
        Dictionary of component name -> weighted reward value
    """
    # Get regime weights
    regime_weights = self.regime_detector.get_regime_weights(regime)
    
    # Apply weights
    weighted_rewards = {}
    for component_name, reward in component_rewards.items():
        # Use regime weight if available, otherwise use default weight
        weight = regime_weights.get(component_name, 1.0)
        weighted_rewards[component_name] = reward * weight
    
    return weighted_rewards
```

### Reward Normalization

```python
def normalize(self, reward: float) -> float:
    """
    Normalize the reward value.
    
    Args:
        reward: Raw reward value
        
    Returns:
        Normalized reward value
    """
    if not self.config.get('enabled', False):
        return reward
    
    method = self.config.get('method', 'rolling')
    
    if method == 'standard':
        # Standard normalization using rolling statistics
        if len(self.reward_history) >= self.window:
            recent_rewards = self.reward_history[-self.window:]
            mean = np.mean(recent_rewards)
            std = np.std(recent_rewards)
            normalized = (reward - mean) / std if std > 0 else reward
        else:
            normalized = reward
            
    elif method == 'robust':
        # Robust normalization using median and IQR
        if len(self.reward_history) >= self.window:
            recent_rewards = self.reward_history[-self.window:]
            median = np.median(recent_rewards)
            q75 = np.percentile(recent_rewards, 75)
            q25 = np.percentile(recent_rewards, 25)
            iqr = q75 - q25
            normalized = (reward - median) / iqr if iqr > 0 else reward
        else:
            normalized = reward
            
    elif method == 'rolling':
        # Rolling normalization with exponential weighting
        if len(self.reward_history) > 0:
            alpha = 2.0 / (self.window + 1)  # Exponential smoothing factor
            if not hasattr(self, 'rolling_mean'):
                self.rolling_mean = 0.0
                self.rolling_std = 1.0
                
            # Update rolling statistics
            self.rolling_mean = alpha * reward + (1 - alpha) * self.rolling_mean
            self.rolling_std = alpha * abs(reward - self.rolling_mean) + (1 - alpha) * self.rolling_std
            
            normalized = (reward - self.rolling_mean) / self.rolling_std if self.rolling_std > 0 else reward
        else:
            normalized = reward
            
    elif method == 'quantile':
        # Quantile normalization
        if len(self.reward_history) >= self.window:
            recent_rewards = self.reward_history[-self.window:]
            percentile = np.searchsorted(np.sort(recent_rewards), reward) / len(recent_rewards)
            normalized = (percentile - 0.5) * 2  # Scale to [-1, 1]
        else:
            normalized = reward
    else:
        normalized = reward
    
    # Clip outliers if enabled
    if self.config.get('clip_outliers', True):
        clip_threshold = self.config.get('clip_threshold', 3.0)
        normalized = np.clip(normalized, -clip_threshold, clip_threshold)
    
    return normalized
```

### Reward Smoothing

```python
def smooth(self, reward: float) -> float:
    """
    Smooth the reward value.
    
    Args:
        reward: Raw reward value
        
    Returns:
        Smoothed reward value
    """
    if not self.config.get('enabled', False):
        return reward
    
    method = self.config.get('method', 'exponential')
    
    if method == 'exponential':
        # Exponential smoothing
        alpha = self.config.get('alpha', 0.2)
        if len(self.reward_history) > 0:
            smoothed = alpha * reward + (1 - alpha) * self.reward_history[-1]
        else:
            smoothed = reward
            
    elif method == 'moving_average':
        # Simple moving average
        window = self.config.get('window', 5)
        if len(self.reward_history) >= window:
            recent_rewards = self.reward_history[-window:] + [reward]
            smoothed = np.mean(recent_rewards)
        else:
            smoothed = reward
            
    elif method == 'none':
        # No smoothing
        smoothed = reward
    else:
        smoothed = reward
    
    return smoothed
```

## Best Practices

### Reward Component Design
1. **Normalization**: Always normalize reward components to ensure stable training
2. **Interpretability**: Make reward components interpretable and meaningful
3. **Balance**: Balance between different reward components to avoid dominance
4. **Regime Awareness**: Consider market regimes when designing reward components
5. **Risk Management**: Include risk management components in the reward system

### Configuration Management
1. **Default Values**: Provide sensible default values for all parameters
2. **Validation**: Validate configuration parameters to ensure they are within reasonable ranges
3. **Documentation**: Document all configuration parameters and their effects
4. **Environment Overrides**: Support environment-specific configuration overrides
5. **Versioning**: Version configuration schemas to ensure backward compatibility

### Performance Optimization
1. **Caching**: Cache expensive calculations to improve performance
2. **Vectorization**: Use vectorized operations for efficient computation
3. **Lazy Evaluation**: Only calculate reward components when needed
4. **Parallel Processing**: Use parallel processing for independent calculations
5. **Memory Management**: Manage memory usage for large datasets

### Monitoring and Debugging
1. **Logging**: Log reward component values for debugging and analysis
2. **Visualization**: Visualize reward components and their contributions
3. **Alerting**: Alert on abnormal reward values or component behavior
4. **Metrics**: Track reward-related metrics over time
5. **A/B Testing**: Test different reward configurations to find optimal settings