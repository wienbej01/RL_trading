# Potential-Based Reward Shaping

## Overview

This document outlines the design for potential-based reward shaping in the multi-ticker RL trading system. Potential-based reward shaping is a technique that uses potential functions to guide the agent toward desirable states without changing the optimal policy. It helps accelerate learning by providing additional reward signals based on the potential value of states.

## Requirements

### Functional Requirements
1. **Potential Function Design**: Design potential functions that capture desirable trading states
2. **Multi-Ticker Support**: Apply potential-based shaping across multiple tickers
3. **State Representation**: Define state representations for potential calculation
4. **Reward Shaping**: Calculate shaped rewards based on potential differences
5. **Policy Invariance**: Ensure the shaping doesn't change the optimal policy
6. **Configurable Parameters**: Allow configuration of potential function parameters
7. **Adaptive Potentials**: Adapt potential functions based on market conditions

### Non-Functional Requirements
1. **Performance**: Efficient potential calculation with minimal computational overhead
2. **Stability**: Stable reward shaping that doesn't introduce excessive noise
3. **Interpretability**: Clear, interpretable potential functions
4. **Configurability**: Flexible configuration of potential function parameters
5. **Adaptability**: Adaptive to changing market conditions

## Architecture Overview

### Core Components

#### 1. PotentialBasedRewardShaper
The main class responsible for implementing potential-based reward shaping.

```python
class PotentialBasedRewardShaper(RewardComponent):
    """
    Potential-based reward shaping component.
    
    Uses potential functions to guide the agent toward desirable states
    without changing the optimal policy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the potential-based reward shaper.
        
        Args:
            config: Configuration dictionary with shaping settings
        """
        super().__init__(config)
        self.potential_functions = {}  # Potential functions for each component
        self.potential_weights = config.get('potential_weights', {})
        self.gamma = config.get('gamma', 0.99)  # Discount factor for potential calculation
        self.adaptive_potentials = config.get('adaptive_potentials', True)
        self.regime_aware = config.get('regime_aware', True)
        self.multi_ticker_correlation = config.get('multi_ticker_correlation', True)
        self.potential_history = {}  # Potential history for each ticker
        self.potential_stats = {}  # Potential statistics for each ticker
        self.regime_detector = RegimeDetector(config.get('regime_detection', {}))
        
        # Initialize potential functions
        self._initialize_potential_functions()
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate the potential-based shaped reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Potential-based shaped reward
        """
        # Calculate current potential
        current_potential = self._calculate_potential(portfolio_state, market_data)
        
        # Calculate previous potential
        previous_potential = self._calculate_potential(previous_portfolio_state, market_data)
        
        # Calculate shaped reward
        shaped_reward = self._calculate_shaped_reward(current_potential, previous_potential)
        
        # Apply regime adjustment if enabled
        if self.regime_aware:
            shaped_reward = self._apply_regime_adjustment(shaped_reward, market_data)
            
        # Apply multi-ticker correlation adjustment if enabled
        if self.multi_ticker_correlation:
            shaped_reward = self._apply_multi_ticker_correlation_adjustment(
                shaped_reward, portfolio_state, market_data
            )
            
        return self.normalize(shaped_reward)
        
    def get_name(self) -> str:
        return "potential_based_reward_shaping"
        
    def _initialize_potential_functions(self):
        """Initialize potential functions for different components."""
        # Portfolio potential function
        self.potential_functions['portfolio'] = PortfolioPotentialFunction(
            self.config.get('portfolio_potential', {})
        )
        
        # Market potential function
        self.potential_functions['market'] = MarketPotentialFunction(
            self.config.get('market_potential', {})
        )
        
        # Risk potential function
        self.potential_functions['risk'] = RiskPotentialFunction(
            self.config.get('risk_potential', {})
        )
        
        # Execution potential function
        self.potential_functions['execution'] = ExecutionPotentialFunction(
            self.config.get('execution_potential', {})
        )
        
        # Set default weights if not provided
        default_weights = {
            'portfolio': 0.3,
            'market': 0.3,
            'risk': 0.2,
            'execution': 0.2
        }
        
        for component, weight in default_weights.items():
            if component not in self.potential_weights:
                self.potential_weights[component] = weight
                
    def _calculate_potential(self, portfolio_state: PortfolioState,
                           market_data: Dict[str, MarketData]) -> Dict[str, float]:
        """
        Calculate potential for the current state.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Dictionary of component -> potential value
        """
        potential = {}
        
        # Calculate portfolio potential
        potential['portfolio'] = self.potential_functions['portfolio'].calculate(
            portfolio_state, market_data
        )
        
        # Calculate market potential
        potential['market'] = self.potential_functions['market'].calculate(
            portfolio_state, market_data
        )
        
        # Calculate risk potential
        potential['risk'] = self.potential_functions['risk'].calculate(
            portfolio_state, market_data
        )
        
        # Calculate execution potential
        potential['execution'] = self.potential_functions['execution'].calculate(
            portfolio_state, market_data
        )
        
        # Update potential history
        self._update_potential_history(potential)
        
        # Update potential statistics
        self._update_potential_stats(potential)
        
        return potential
        
    def _calculate_shaped_reward(self, current_potential: Dict[str, float],
                               previous_potential: Dict[str, float]) -> float:
        """
        Calculate shaped reward based on potential difference.
        
        Args:
            current_potential: Current potential values
            previous_potential: Previous potential values
            
        Returns:
            Shaped reward
        """
        shaped_reward = 0.0
        
        for component in current_potential:
            if component in previous_potential:
                # Calculate potential difference
                potential_diff = current_potential[component] - previous_potential[component]
                
                # Apply discount factor
                discounted_diff = self.gamma * potential_diff
                
                # Apply component weight
                component_reward = self.potential_weights[component] * discounted_diff
                
                shaped_reward += component_reward
                
        return shaped_reward
        
    def _update_potential_history(self, potential: Dict[str, float]):
        """
        Update potential history.
        
        Args:
            potential: Current potential values
        """
        timestamp = pd.Timestamp.now()
        
        for component, value in potential.items():
            if component not in self.potential_history:
                self.potential_history[component] = []
                
            self.potential_history[component].append({
                'timestamp': timestamp,
                'value': value
            })
            
            # Keep history limited
            max_history = self.config.get('max_history', 1000)
            if len(self.potential_history[component]) > max_history:
                self.potential_history[component].pop(0)
                
    def _update_potential_stats(self, potential: Dict[str, float]):
        """
        Update potential statistics.
        
        Args:
            potential: Current potential values
        """
        for component, value in potential.items():
            if component not in self.potential_stats:
                self.potential_stats[component] = {
                    'mean': value,
                    'std': 0.0,
                    'min': value,
                    'max': value,
                    'count': 1
                }
            else:
                stats = self.potential_stats[component]
                count = stats['count']
                
                # Update mean
                stats['mean'] = (stats['mean'] * count + value) / (count + 1)
                
                # Update std
                if count > 1:
                    variance = ((count - 1) * stats['std']**2 + 
                               (value - stats['mean'])**2 * count / (count + 1)) / count
                    stats['std'] = np.sqrt(variance)
                
                # Update min and max
                stats['min'] = min(stats['min'], value)
                stats['max'] = max(stats['max'], value)
                
                # Update count
                stats['count'] += 1
                
    def _apply_regime_adjustment(self, reward: float, market_data: Dict[str, MarketData]) -> float:
        """
        Apply regime-based adjustment to the reward.
        
        Args:
            reward: Original reward value
            market_data: Current market data for all tickers
            
        Returns:
            Adjusted reward value
        """
        if not self.regime_aware or not market_data:
            return reward
            
        # Detect current regime
        regime = self.regime_detector.detect_regime(market_data)
        
        # Adjust reward based on regime
        if regime == 'high_volatility':
            adjustment_factor = 0.8  # Reduce reward in high volatility
        elif regime == 'low_volatility':
            adjustment_factor = 1.2  # Increase reward in low volatility
        elif regime == 'trending':
            adjustment_factor = 1.1  # Slightly increase reward in trending markets
        elif regime == 'range_bound':
            adjustment_factor = 0.9  # Slightly reduce reward in range-bound markets
        else:  # normal regime
            adjustment_factor = 1.0
            
        return reward * adjustment_factor
        
    def _apply_multi_ticker_correlation_adjustment(self, reward: float,
                                                 portfolio_state: PortfolioState,
                                                 market_data: Dict[str, MarketData]) -> float:
        """
        Apply multi-ticker correlation adjustment to the reward.
        
        Args:
            reward: Original reward
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Adjusted reward
        """
        if not self.multi_ticker_correlation or not market_data:
            return reward
            
        # Calculate portfolio diversification
        diversification = self._calculate_portfolio_diversification(portfolio_state, market_data)
        
        # Adjust reward based on diversification
        # Higher diversification means higher reward
        diversification_adjustment = 1.0 + (diversification * 0.3)
        reward *= diversification_adjustment
        
        return reward
        
    def _calculate_portfolio_diversification(self, portfolio_state: PortfolioState,
                                           market_data: Dict[str, MarketData]) -> float:
        """
        Calculate portfolio diversification score.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Diversification score (0-1)
        """
        positions = portfolio_state.positions
        
        if len(positions) <= 1:
            return 0.0  # No diversification with single position
            
        # Calculate position weights
        total_value = sum(abs(pos) for pos in positions.values())
        if total_value == 0:
            return 0.0
            
        weights = [abs(pos) / total_value for pos in positions.values()]
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = sum(w**2 for w in weights)
        
        # Convert to diversification score (1 - HHI)
        diversification = 1.0 - hhi
        
        return diversification
```

#### 2. PortfolioPotentialFunction
Component for calculating portfolio-based potential.

```python
class PortfolioPotentialFunction:
    """
    Portfolio potential function.
    
    Calculates potential based on portfolio characteristics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize portfolio potential function.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pnl_weight = config.get('pnl_weight', 0.4)
        self.drawdown_weight = config.get('drawdown_weight', 0.3)
        self.concentration_weight = config.get('concentration_weight', 0.3)
        self.target_return = config.get('target_return', 0.001)  # Daily target return
        self.max_drawdown = config.get('max_drawdown', 0.1)  # Maximum acceptable drawdown
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData]) -> float:
        """
        Calculate portfolio potential.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Portfolio potential value
        """
        # Calculate PnL potential
        pnl_potential = self._calculate_pnl_potential(portfolio_state)
        
        # Calculate drawdown potential
        drawdown_potential = self._calculate_drawdown_potential(portfolio_state)
        
        # Calculate concentration potential
        concentration_potential = self._calculate_concentration_potential(portfolio_state)
        
        # Combine potentials
        potential = (
            self.pnl_weight * pnl_potential +
            self.drawdown_weight * drawdown_potential +
            self.concentration_weight * concentration_potential
        )
        
        return potential
        
    def _calculate_pnl_potential(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate PnL-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            PnL potential value
        """
        if not hasattr(portfolio_state, 'pnl') or portfolio_state.pnl is None:
            return 0.0
            
        # Calculate PnL relative to target
        pnl_ratio = portfolio_state.pnl / self.target_return if self.target_return != 0 else 0.0
        
        # Use sigmoid to bound potential
        pnl_potential = 2.0 / (1.0 + np.exp(-pnl_ratio)) - 1.0
        
        return pnl_potential
        
    def _calculate_drawdown_potential(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate drawdown-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Drawdown potential value
        """
        if not hasattr(portfolio_state, 'drawdown') or portfolio_state.drawdown is None:
            return 0.0
            
        # Calculate drawdown relative to maximum
        drawdown_ratio = portfolio_state.drawdown / self.max_drawdown if self.max_drawdown != 0 else 0.0
        
        # Higher drawdown means lower potential
        drawdown_potential = -drawdown_ratio
        
        return drawdown_potential
        
    def _calculate_concentration_potential(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate concentration-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Concentration potential value
        """
        positions = portfolio_state.positions
        
        if len(positions) <= 1:
            return 0.0  # No concentration with single position
            
        # Calculate position weights
        total_value = sum(abs(pos) for pos in positions.values())
        if total_value == 0:
            return 0.0
            
        weights = [abs(pos) / total_value for pos in positions.values()]
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = sum(w**2 for w in weights)
        
        # Higher concentration means lower potential
        concentration_potential = 1.0 - hhi
        
        return concentration_potential
```

#### 3. MarketPotentialFunction
Component for calculating market-based potential.

```python
class MarketPotentialFunction:
    """
    Market potential function.
    
    Calculates potential based on market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize market potential function.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.trend_weight = config.get('trend_weight', 0.4)
        self.volatility_weight = config.get('volatility_weight', 0.3)
        self.liquidity_weight = config.get('liquidity_weight', 0.3)
        self.optimal_volatility = config.get('optimal_volatility', 0.2)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData]) -> float:
        """
        Calculate market potential.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Market potential value
        """
        # Calculate trend potential
        trend_potential = self._calculate_trend_potential(portfolio_state, market_data)
        
        # Calculate volatility potential
        volatility_potential = self._calculate_volatility_potential(market_data)
        
        # Calculate liquidity potential
        liquidity_potential = self._calculate_liquidity_potential(market_data)
        
        # Combine potentials
        potential = (
            self.trend_weight * trend_potential +
            self.volatility_weight * volatility_potential +
            self.liquidity_weight * liquidity_potential
        )
        
        return potential
        
    def _calculate_trend_potential(self, portfolio_state: PortfolioState,
                                 market_data: Dict[str, MarketData]) -> float:
        """
        Calculate trend-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Trend potential value
        """
        trend_scores = []
        
        for ticker, data in market_data.items():
            if hasattr(data, 'trend') and data.trend is not None:
                # Align trend with portfolio position
                position = portfolio_state.positions.get(ticker, 0.0)
                if position != 0:
                    aligned_trend = data.trend * np.sign(position)
                    trend_scores.append(aligned_trend)
                    
        if not trend_scores:
            return 0.0
            
        # Average aligned trend score
        avg_trend = np.mean(trend_scores)
        
        # Use sigmoid to bound potential
        trend_potential = 2.0 / (1.0 + np.exp(-avg_trend * 5)) - 1.0
        
        return trend_potential
        
    def _calculate_volatility_potential(self, market_data: Dict[str, MarketData]) -> float:
        """
        Calculate volatility-based potential.
        
        Args:
            market_data: Current market data for all tickers
            
        Returns:
            Volatility potential value
        """
        volatilities = []
        
        for ticker, data in market_data.items():
            if hasattr(data, 'volatility') and data.volatility is not None:
                volatilities.append(data.volatility)
                
        if not volatilities:
            return 0.0
            
        # Average volatility
        avg_volatility = np.mean(volatilities)
        
        # Optimal volatility is best
        volatility_score = 1.0 - abs(avg_volatility - self.optimal_volatility) / self.optimal_volatility
        volatility_score = max(0, volatility_score)
        
        return volatility_score
        
    def _calculate_liquidity_potential(self, market_data: Dict[str, MarketData]) -> float:
        """
        Calculate liquidity-based potential.
        
        Args:
            market_data: Current market data for all tickers
            
        Returns:
            Liquidity potential value
        """
        liquidity_scores = []
        
        for ticker, data in market_data.items():
            if hasattr(data, 'volume') and hasattr(data, 'avg_volume') and data.avg_volume > 0:
                volume_ratio = data.volume / data.avg_volume
                # Higher volume is better for liquidity
                liquidity_score = min(volume_ratio / 2.0, 1.0)
                liquidity_scores.append(liquidity_score)
                
        if not liquidity_scores:
            return 0.0
            
        # Average liquidity score
        avg_liquidity = np.mean(liquidity_scores)
        
        return avg_liquidity
```

#### 4. RiskPotentialFunction
Component for calculating risk-based potential.

```python
class RiskPotentialFunction:
    """
    Risk potential function.
    
    Calculates potential based on risk characteristics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk potential function.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.variance_weight = config.get('variance_weight', 0.4)
        self.beta_weight = config.get('beta_weight', 0.3)
        self.value_at_risk_weight = config.get('value_at_risk_weight', 0.3)
        self.target_variance = config.get('target_variance', 0.1)
        self.target_beta = config.get('target_beta', 1.0)
        self.var_threshold = config.get('var_threshold', 0.05)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData]) -> float:
        """
        Calculate risk potential.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Risk potential value
        """
        # Calculate variance potential
        variance_potential = self._calculate_variance_potential(portfolio_state)
        
        # Calculate beta potential
        beta_potential = self._calculate_beta_potential(portfolio_state, market_data)
        
        # Calculate VaR potential
        var_potential = self._calculate_var_potential(portfolio_state)
        
        # Combine potentials
        potential = (
            self.variance_weight * variance_potential +
            self.beta_weight * beta_potential +
            self.value_at_risk_weight * var_potential
        )
        
        return potential
        
    def _calculate_variance_potential(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate variance-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Variance potential value
        """
        if not hasattr(portfolio_state, 'variance') or portfolio_state.variance is None:
            return 0.0
            
        # Calculate variance relative to target
        variance_ratio = portfolio_state.variance / self.target_variance if self.target_variance != 0 else 0.0
        
        # Lower variance is better (up to a point)
        if variance_ratio < 1.0:
            variance_potential = variance_ratio
        else:
            variance_potential = 1.0 / variance_ratio
            
        return variance_potential
        
    def _calculate_beta_potential(self, portfolio_state: PortfolioState,
                                market_data: Dict[str, MarketData]) -> float:
        """
        Calculate beta-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Beta potential value
        """
        if not hasattr(portfolio_state, 'beta') or portfolio_state.beta is None:
            return 0.0
            
        # Calculate beta relative to target
        beta_diff = abs(portfolio_state.beta - self.target_beta)
        
        # Lower beta difference is better
        beta_potential = 1.0 / (1.0 + beta_diff)
        
        return beta_potential
        
    def _calculate_var_potential(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate VaR-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            VaR potential value
        """
        if not hasattr(portfolio_state, 'var') or portfolio_state.var is None:
            return 0.0
            
        # Calculate VaR relative to threshold
        var_ratio = portfolio_state.var / self.var_threshold if self.var_threshold != 0 else 0.0
        
        # Lower VaR is better
        if var_ratio < 1.0:
            var_potential = 1.0 - var_ratio
        else:
            var_potential = 0.0
            
        return var_potential
```

#### 5. ExecutionPotentialFunction
Component for calculating execution-based potential.

```python
class ExecutionPotentialFunction:
    """
    Execution potential function.
    
    Calculates potential based on execution quality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize execution potential function.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.slippage_weight = config.get('slippage_weight', 0.4)
        self.timing_weight = config.get('timing_weight', 0.3)
        self.participation_weight = config.get('participation_weight', 0.3)
        self.target_slippage = config.get('target_slippage', 0.001)
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData]) -> float:
        """
        Calculate execution potential.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Execution potential value
        """
        # Calculate slippage potential
        slippage_potential = self._calculate_slippage_potential(portfolio_state)
        
        # Calculate timing potential
        timing_potential = self._calculate_timing_potential(portfolio_state, market_data)
        
        # Calculate participation potential
        participation_potential = self._calculate_participation_potential(portfolio_state)
        
        # Combine potentials
        potential = (
            self.slippage_weight * slippage_potential +
            self.timing_weight * timing_potential +
            self.participation_weight * participation_potential
        )
        
        return potential
        
    def _calculate_slippage_potential(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate slippage-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Slippage potential value
        """
        if not hasattr(portfolio_state, 'slippage') or portfolio_state.slippage is None:
            return 0.0
            
        # Calculate slippage relative to target
        slippage_ratio = portfolio_state.slippage / self.target_slippage if self.target_slippage != 0 else 0.0
        
        # Lower slippage is better
        if slippage_ratio < 1.0:
            slippage_potential = 1.0 - slippage_ratio
        else:
            slippage_potential = 0.0
            
        return slippage_potential
        
    def _calculate_timing_potential(self, portfolio_state: PortfolioState,
                                  market_data: Dict[str, MarketData]) -> float:
        """
        Calculate timing-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            
        Returns:
            Timing potential value
        """
        timing_scores = []
        
        for ticker, data in market_data.items():
            if hasattr(data, 'price') and hasattr(data, 'vwap') and data.vwap > 0:
                # Compare execution price to VWAP
                price_vwap_ratio = data.price / data.vwap
                
                # Closer to VWAP is better
                timing_score = 1.0 - abs(price_vwap_ratio - 1.0)
                timing_scores.append(timing_score)
                
        if not timing_scores:
            return 0.0
            
        # Average timing score
        avg_timing = np.mean(timing_scores)
        
        return avg_timing
        
    def _calculate_participation_potential(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate participation-based potential.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Participation potential value
        """
        if not hasattr(portfolio_state, 'participation_rate') or portfolio_state.participation_rate is None:
            return 0.0
            
        # Optimal participation rate is around 20%
        optimal_participation = 0.2
        participation_diff = abs(portfolio_state.participation_rate - optimal_participation)
        
        # Lower participation difference is better
        participation_potential = 1.0 / (1.0 + participation_diff * 5)
        
        return participation_potential
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
│ Current         │    │ Previous        │    │ Potential       │
│ Potential       │    │ Potential       │    │ History         │
│ Calculation     │◀───│ Calculation     │◀───│ (Update)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Shaped Reward   │
│ Calculation     │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Regime          │
│ Adjustment      │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Multi-Ticker    │
│ Correlation     │
│ Adjustment     │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Normalized      │
│ Reward          │
│ (Output)        │
└─────────────────┘
```

## Configuration Structure

### Potential-Based Reward Shaping Configuration

```yaml
# Potential-based reward shaping configuration
potential_based_reward_shaping:
  enabled: true
  weight: 0.2
  
  # General parameters
  gamma: 0.99  # Discount factor for potential calculation
  adaptive_potentials: true  # Enable adaptive potential functions
  regime_aware: true  # Enable regime-aware adjustments
  multi_ticker_correlation: true  # Enable multi-ticker correlation adjustments
  max_history: 1000  # Maximum potential history to keep
  
  # Potential function weights
  potential_weights:
    portfolio: 0.3  # Weight for portfolio potential
    market: 0.3  # Weight for market potential
    risk: 0.2  # Weight for risk potential
    execution: 0.2  # Weight for execution potential
    
  # Portfolio potential configuration
  portfolio_potential:
    pnl_weight: 0.4  # Weight for PnL component
    drawdown_weight: 0.3  # Weight for drawdown component
    concentration_weight: 0.3  # Weight for concentration component
    target_return: 0.001  # Daily target return
    max_drawdown: 0.1  # Maximum acceptable drawdown
    
  # Market potential configuration
  market_potential:
    trend_weight: 0.4  # Weight for trend component
    volatility_weight: 0.3  # Weight for volatility component
    liquidity_weight: 0.3  # Weight for liquidity component
    optimal_volatility: 0.2  # Optimal volatility level
    
  # Risk potential configuration
  risk_potential:
    variance_weight: 0.4  # Weight for variance component
    beta_weight: 0.3  # Weight for beta component
    value_at_risk_weight: 0.3  # Weight for VaR component
    target_variance: 0.1  # Target variance level
    target_beta: 1.0  # Target beta level
    var_threshold: 0.05  # VaR threshold
    
  # Execution potential configuration
  execution_potential:
    slippage_weight: 0.4  # Weight for slippage component
    timing_weight: 0.3  # Weight for timing component
    participation_weight: 0.3  # Weight for participation component
    target_slippage: 0.001  # Target slippage level
    
  # Regime detection parameters
  regime_detection:
    window: 20  # Window for regime detection
    volatility_threshold: 0.2  # Threshold for high volatility regime
    trend_threshold: 0.1  # Threshold for trending regime
    
  # Normalization
  normalization:
    enabled: true
    method: standard  # standard, min_max, clip
    clip_threshold: 3.0
```

## Implementation Details

### Potential Calculation Algorithm

```python
def _calculate_potential(self, portfolio_state: PortfolioState,
                       market_data: Dict[str, MarketData]) -> Dict[str, float]:
    """
    Calculate potential for the current state.
    
    Args:
        portfolio_state: Current portfolio state
        market_data: Current market data for all tickers
        
    Returns:
        Dictionary of component -> potential value
    """
    potential = {}
    
    # Calculate portfolio potential
    potential['portfolio'] = self.potential_functions['portfolio'].calculate(
        portfolio_state, market_data
    )
    
    # Calculate market potential
    potential['market'] = self.potential_functions['market'].calculate(
        portfolio_state, market_data
    )
    
    # Calculate risk potential
    potential['risk'] = self.potential_functions['risk'].calculate(
        portfolio_state, market_data
    )
    
    # Calculate execution potential
    potential['execution'] = self.potential_functions['execution'].calculate(
        portfolio_state, market_data
    )
    
    # Update potential history
    self._update_potential_history(potential)
    
    # Update potential statistics
    self._update_potential_stats(potential)
    
    return potential
```

### Shaped Reward Calculation Algorithm

```python
def _calculate_shaped_reward(self, current_potential: Dict[str, float],
                           previous_potential: Dict[str, float]) -> float:
    """
    Calculate shaped reward based on potential difference.
    
    Args:
        current_potential: Current potential values
        previous_potential: Previous potential values
        
    Returns:
        Shaped reward
    """
    shaped_reward = 0.0
    
    for component in current_potential:
        if component in previous_potential:
            # Calculate potential difference
            potential_diff = current_potential[component] - previous_potential[component]
            
            # Apply discount factor
            discounted_diff = self.gamma * potential_diff
            
            # Apply component weight
            component_reward = self.potential_weights[component] * discounted_diff
            
            shaped_reward += component_reward
            
    return shaped_reward
```

### Portfolio Potential Calculation Algorithm

```python
def calculate(self, portfolio_state: PortfolioState,
             market_data: Dict[str, MarketData]) -> float:
    """
    Calculate portfolio potential.
    
    Args:
        portfolio_state: Current portfolio state
        market_data: Current market data for all tickers
        
    Returns:
        Portfolio potential value
    """
    # Calculate PnL potential
    pnl_potential = self._calculate_pnl_potential(portfolio_state)
    
    # Calculate drawdown potential
    drawdown_potential = self._calculate_drawdown_potential(portfolio_state)
    
    # Calculate concentration potential
    concentration_potential = self._calculate_concentration_potential(portfolio_state)
    
    # Combine potentials
    potential = (
        self.pnl_weight * pnl_potential +
        self.drawdown_weight * drawdown_potential +
        self.concentration_weight * concentration_potential
    )
    
    return potential
```

### Market Potential Calculation Algorithm

```python
def calculate(self, portfolio_state: PortfolioState,
             market_data: Dict[str, MarketData]) -> float:
    """
    Calculate market potential.
    
    Args:
        portfolio_state: Current portfolio state
        market_data: Current market data for all tickers
        
    Returns:
        Market potential value
    """
    # Calculate trend potential
    trend_potential = self._calculate_trend_potential(portfolio_state, market_data)
    
    # Calculate volatility potential
    volatility_potential = self._calculate_volatility_potential(market_data)
    
    # Calculate liquidity potential
    liquidity_potential = self._calculate_liquidity_potential(market_data)
    
    # Combine potentials
    potential = (
        self.trend_weight * trend_potential +
        self.volatility_weight * volatility_potential +
        self.liquidity_weight * liquidity_potential
    )
    
    return potential
```

## Best Practices

### Potential Function Design
1. **Policy Invariance**: Ensure potential functions satisfy the policy invariance property
2. **Bounded Potentials**: Keep potential values bounded to avoid numerical issues
3. **Smooth Transitions**: Design potential functions with smooth transitions
4. **Interpretability**: Make potential functions interpretable and meaningful
5. **Adaptability**: Adapt potential functions to changing market conditions

### Multi-Ticker Considerations
1. **Cross-Ticker Effects**: Consider cross-ticker effects in potential calculations
2. **Portfolio-Level Metrics**: Use portfolio-level metrics for potential calculation
3. **Diversification**: Encourage diversification through potential functions
4. **Correlation Awareness**: Be aware of correlations between tickers
5. **Risk Parity**: Consider risk parity in potential calculations

### Configuration Management
1. **Default Values**: Provide sensible default values for all parameters
2. **Validation**: Validate configuration parameters to ensure they are within reasonable ranges
3. **Documentation**: Document all configuration parameters and their effects
4. **Environment Overrides**: Support environment-specific configuration overrides
5. **Versioning**: Version configuration schemas to ensure backward compatibility

### Performance Optimization
1. **Efficient Calculation**: Use efficient algorithms for potential calculation
2. **Caching**: Cache expensive calculations to improve performance
3. **Vectorization**: Use vectorized operations for efficient computation
4. **Memory Management**: Manage memory usage for potential histories
5. **Lazy Evaluation**: Only evaluate potentials when needed

### Monitoring and Debugging
1. **Logging**: Log potential values for debugging and analysis
2. **Visualization**: Visualize potential values over time
3. **Alerting**: Alert on unusual potential values
4. **Metrics**: Track potential-related metrics over time
5. **A/B Testing**: Test different potential function configurations to find optimal settings