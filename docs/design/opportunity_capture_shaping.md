# Opportunity/Capture Shaping

## Overview

This document outlines the design for opportunity/capture shaping in the multi-ticker RL trading system. Opportunity/capture shaping is a reward shaping technique that encourages the agent to identify and capitalize on trading opportunities while balancing the trade-off between opportunity identification and successful execution.

## Requirements

### Functional Requirements
1. **Opportunity Identification**: Identify potential trading opportunities based on market conditions
2. **Capture Efficiency**: Reward successful capture of identified opportunities
3. **Multi-Ticker Support**: Apply opportunity/capture shaping across multiple tickers
4. **Dynamic Thresholds**: Adjust opportunity thresholds based on market conditions
5. **Regime Awareness**: Adapt opportunity/capture criteria based on market regimes
6. **Configurable Parameters**: Allow configuration of opportunity/capture parameters
7. **Performance Tracking**: Track opportunity identification and capture performance

### Non-Functional Requirements
1. **Performance**: Efficient opportunity detection with minimal computational overhead
2. **Stability**: Stable reward calculation that doesn't introduce excessive noise
3. **Interpretability**: Clear, interpretable opportunity/capture metrics
4. **Configurability**: Flexible configuration of opportunity/capture parameters
5. **Adaptability**: Adaptive to changing market conditions

## Architecture Overview

### Core Components

#### 1. OpportunityCaptureShaper
The main class responsible for implementing opportunity/capture shaping.

```python
class OpportunityCaptureShaper(RewardComponent):
    """
    Opportunity/capture shaping component.
    
    Encourages the agent to identify and capitalize on trading opportunities
    while balancing the trade-off between opportunity identification and successful execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the opportunity/capture shaper.
        
        Args:
            config: Configuration dictionary with shaping settings
        """
        super().__init__(config)
        self.opportunity_threshold = config.get('opportunity_threshold', 0.02)  # 2% minimum opportunity
        self.capture_threshold = config.get('capture_threshold', 0.5)  # 50% capture ratio
        self.opportunity_window = config.get('opportunity_window', 10)  # Lookback window for opportunity detection
        self.capture_window = config.get('capture_window', 5)  # Window for capture evaluation
        self.regime_aware = config.get('regime_aware', True)
        self.dynamic_thresholds = config.get('dynamic_thresholds', True)
        self.multi_ticker_correlation = config.get('multi_ticker_correlation', True)
        self.opportunity_history = {}  # Opportunity history for each ticker
        self.capture_history = {}  # Capture history for each ticker
        self.performance_metrics = {}  # Performance metrics for each ticker
        self.regime_detector = RegimeDetector(config.get('regime_detection', {}))
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate the opportunity/capture shaping reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Opportunity/capture shaping reward
        """
        # Identify opportunities
        opportunities = self._identify_opportunities(market_data)
        
        # Evaluate capture efficiency
        capture_efficiency = self._evaluate_capture_efficiency(
            portfolio_state, market_data, actions, previous_portfolio_state, opportunities
        )
        
        # Calculate opportunity/capture reward
        reward = self._calculate_opportunity_capture_reward(opportunities, capture_efficiency)
        
        # Apply regime adjustment if enabled
        if self.regime_aware:
            reward = self._apply_regime_adjustment(reward, market_data)
            
        # Apply dynamic threshold adjustment if enabled
        if self.dynamic_thresholds:
            reward = self._apply_dynamic_threshold_adjustment(reward, market_data)
            
        return self.normalize(reward)
        
    def get_name(self) -> str:
        return "opportunity_capture_shaping"
        
    def _identify_opportunities(self, market_data: Dict[str, MarketData]) -> Dict[str, Opportunity]:
        """
        Identify trading opportunities for each ticker.
        
        Args:
            market_data: Current market data for all tickers
            
        Returns:
            Dictionary of ticker -> opportunity
        """
        opportunities = {}
        
        for ticker, data in market_data.items():
            # Calculate opportunity score
            opportunity_score = self._calculate_opportunity_score(ticker, data)
            
            # Determine if this is an opportunity
            is_opportunity = opportunity_score >= self.opportunity_threshold
            
            # Create opportunity object
            opportunity = Opportunity(
                ticker=ticker,
                score=opportunity_score,
                is_opportunity=is_opportunity,
                timestamp=data.timestamp if hasattr(data, 'timestamp') else pd.Timestamp.now(),
                expected_return=self._calculate_expected_return(ticker, data),
                risk_level=self._calculate_risk_level(ticker, data),
                time_horizon=self._calculate_time_horizon(ticker, data)
            )
            
            opportunities[ticker] = opportunity
            
            # Update opportunity history
            if ticker not in self.opportunity_history:
                self.opportunity_history[ticker] = []
                
            self.opportunity_history[ticker].append(opportunity)
            
            # Keep history limited
            if len(self.opportunity_history[ticker]) > self.opportunity_window:
                self.opportunity_history[ticker].pop(0)
                
        return opportunities
        
    def _calculate_opportunity_score(self, ticker: str, data: MarketData) -> float:
        """
        Calculate opportunity score for a ticker.
        
        Args:
            ticker: Ticker symbol
            data: Market data for the ticker
            
        Returns:
            Opportunity score
        """
        score = 0.0
        
        # Price momentum component
        if hasattr(data, 'momentum') and data.momentum is not None:
            score += abs(data.momentum) * 0.3
            
        # Volatility component
        if hasattr(data, 'volatility') and data.volatility is not None:
            # Moderate volatility is good for opportunities
            optimal_volatility = 0.2
            volatility_score = 1.0 - abs(data.volatility - optimal_volatility) / optimal_volatility
            score += max(0, volatility_score) * 0.2
            
        # Volume component
        if hasattr(data, 'volume') and hasattr(data, 'avg_volume') and data.avg_volume > 0:
            volume_ratio = data.volume / data.avg_volume
            # High volume is good for opportunities
            volume_score = min(volume_ratio, 2.0) / 2.0
            score += volume_score * 0.2
            
        # Technical indicators component
        if hasattr(data, 'rsi') and data.rsi is not None:
            # Extreme RSI values indicate potential reversal opportunities
            rsi_score = max(0, (abs(data.rsi - 50) - 30) / 20)
            score += rsi_score * 0.15
            
        # Spread component
        if hasattr(data, 'spread') and hasattr(data, 'price') and data.price > 0:
            relative_spread = data.spread / data.price
            # Lower spreads are better for opportunities
            spread_score = max(0, 1.0 - relative_spread * 1000)
            score += spread_score * 0.15
            
        return score
        
    def _calculate_expected_return(self, ticker: str, data: MarketData) -> float:
        """
        Calculate expected return for a ticker.
        
        Args:
            ticker: Ticker symbol
            data: Market data for the ticker
            
        Returns:
            Expected return
        """
        # Simple expected return based on momentum and volatility
        if hasattr(data, 'momentum') and data.momentum is not None:
            if hasattr(data, 'volatility') and data.volatility is not None and data.volatility > 0:
                # Expected return proportional to momentum and inversely proportional to volatility
                expected_return = data.momentum / data.volatility
                return expected_return
                
        return 0.0
        
    def _calculate_risk_level(self, ticker: str, data: MarketData) -> float:
        """
        Calculate risk level for a ticker.
        
        Args:
            ticker: Ticker symbol
            data: Market data for the ticker
            
        Returns:
            Risk level (0-1)
        """
        risk = 0.5  # Default medium risk
        
        # Volatility component
        if hasattr(data, 'volatility') and data.volatility is not None:
            # Higher volatility means higher risk
            risk += min(data.volatility * 2, 0.5)
            
        # Trend component
        if hasattr(data, 'trend_strength') and data.trend_strength is not None:
            # Stronger trends might mean lower risk
            risk -= data.trend_strength * 0.2
            
        # Volume component
        if hasattr(data, 'volume') and hasattr(data, 'avg_volume') and data.avg_volume > 0:
            volume_ratio = data.volume / data.avg_volume
            # Higher volume might mean lower risk
            risk -= min(volume_ratio / 5, 0.2)
            
        return max(0, min(risk, 1.0))
        
    def _calculate_time_horizon(self, ticker: str, data: MarketData) -> int:
        """
        Calculate time horizon for an opportunity.
        
        Args:
            ticker: Ticker symbol
            data: Market data for the ticker
            
        Returns:
            Time horizon in periods
        """
        # Simple time horizon based on volatility and trend
        if hasattr(data, 'volatility') and data.volatility is not None:
            if hasattr(data, 'trend_strength') and data.trend_strength is not None:
                # Higher volatility and stronger trend suggest shorter time horizon
                base_horizon = 10
                volatility_factor = max(0.5, 1.0 - data.volatility)
                trend_factor = max(0.5, 1.0 - data.trend_strength)
                time_horizon = int(base_horizon * volatility_factor * trend_factor)
                return max(1, time_horizon)
                
        return 5  # Default time horizon
        
    def _evaluate_capture_efficiency(self, portfolio_state: PortfolioState,
                                   market_data: Dict[str, MarketData],
                                   actions: Dict[str, float],
                                   previous_portfolio_state: PortfolioState,
                                   opportunities: Dict[str, Opportunity]) -> Dict[str, float]:
        """
        Evaluate capture efficiency for each ticker.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            opportunities: Current opportunities
            
        Returns:
            Dictionary of ticker -> capture efficiency
        """
        capture_efficiency = {}
        
        for ticker, opportunity in opportunities.items():
            if ticker in actions:
                action = actions[ticker]
                
                # Calculate capture efficiency
                efficiency = self._calculate_ticker_capture_efficiency(
                    ticker, action, portfolio_state, previous_portfolio_state, opportunity
                )
                
                capture_efficiency[ticker] = efficiency
                
                # Update capture history
                if ticker not in self.capture_history:
                    self.capture_history[ticker] = []
                    
                self.capture_history[ticker].append({
                    'timestamp': portfolio_state.timestamp,
                    'efficiency': efficiency,
                    'action': action,
                    'opportunity_score': opportunity.score
                })
                
                # Keep history limited
                if len(self.capture_history[ticker]) > self.capture_window:
                    self.capture_history[ticker].pop(0)
                    
        return capture_efficiency
        
    def _calculate_ticker_capture_efficiency(self, ticker: str, action: float,
                                           portfolio_state: PortfolioState,
                                           previous_portfolio_state: PortfolioState,
                                           opportunity: Opportunity) -> float:
        """
        Calculate capture efficiency for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            action: Action taken
            portfolio_state: Current portfolio state
            previous_portfolio_state: Previous portfolio state
            opportunity: Current opportunity
            
        Returns:
            Capture efficiency (0-1)
        """
        # If no opportunity, efficiency is 0
        if not opportunity.is_opportunity:
            return 0.0
            
        # Calculate position change
        previous_position = previous_portfolio_state.positions.get(ticker, 0.0)
        current_position = portfolio_state.positions.get(ticker, 0.0)
        position_change = current_position - previous_position
        
        # Calculate capture efficiency based on action alignment with opportunity
        if opportunity.expected_return > 0:
            # Positive opportunity, should go long
            if action > 0 and position_change > 0:
                # Correct action, efficiency based on action size
                efficiency = min(abs(action), 1.0)
            else:
                # Incorrect action or no action
                efficiency = 0.0
        else:
            # Negative opportunity, should go short
            if action < 0 and position_change < 0:
                # Correct action, efficiency based on action size
                efficiency = min(abs(action), 1.0)
            else:
                # Incorrect action or no action
                efficiency = 0.0
                
        # Adjust efficiency based on risk level
        # Higher risk opportunities should have higher efficiency requirements
        risk_adjustment = 1.0 - (opportunity.risk_level * 0.3)
        efficiency *= risk_adjustment
        
        return efficiency
        
    def _calculate_opportunity_capture_reward(self, opportunities: Dict[str, Opportunity],
                                            capture_efficiency: Dict[str, float]) -> float:
        """
        Calculate opportunity/capture reward.
        
        Args:
            opportunities: Current opportunities
            capture_efficiency: Capture efficiency for each ticker
            
        Returns:
            Opportunity/capture reward
        """
        reward = 0.0
        
        for ticker, opportunity in opportunities.items():
            if ticker in capture_efficiency:
                efficiency = capture_efficiency[ticker]
                
                if opportunity.is_opportunity:
                    # Reward for identifying opportunities
                    opportunity_reward = opportunity.score * 0.3
                    
                    # Reward for capturing opportunities
                    capture_reward = efficiency * opportunity.score * 0.7
                    
                    # Total reward for this ticker
                    ticker_reward = opportunity_reward + capture_reward
                    
                    # Apply multi-ticker correlation adjustment if enabled
                    if self.multi_ticker_correlation:
                        ticker_reward = self._apply_multi_ticker_correlation_adjustment(
                            ticker_reward, ticker, opportunities
                        )
                        
                    reward += ticker_reward
                    
        # Normalize by number of tickers
        if opportunities:
            reward /= len(opportunities)
            
        return reward
        
    def _apply_multi_ticker_correlation_adjustment(self, reward: float, ticker: str,
                                                 opportunities: Dict[str, Opportunity]) -> float:
        """
        Apply multi-ticker correlation adjustment to reward.
        
        Args:
            reward: Original reward
            ticker: Current ticker
            opportunities: All opportunities
            
        Returns:
            Adjusted reward
        """
        # Calculate correlation with other tickers
        correlation_sum = 0.0
        correlation_count = 0
        
        for other_ticker, other_opportunity in opportunities.items():
            if other_ticker != ticker and other_opportunity.is_opportunity:
                # Simple correlation based on opportunity scores
                if hasattr(self, 'correlation_matrix'):
                    correlation = self.correlation_matrix.get((ticker, other_ticker), 0.0)
                else:
                    # Default correlation if not available
                    correlation = 0.0
                    
                correlation_sum += abs(correlation)
                correlation_count += 1
                
        if correlation_count > 0:
            avg_correlation = correlation_sum / correlation_count
            
            # Adjust reward based on correlation
            # Higher correlation means lower reward (diversification penalty)
            correlation_adjustment = 1.0 - (avg_correlation * 0.3)
            reward *= correlation_adjustment
            
        return reward
        
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
            adjustment_factor = 1.2  # Increase reward in high volatility (more opportunities)
        elif regime == 'low_volatility':
            adjustment_factor = 0.8  # Decrease reward in low volatility (fewer opportunities)
        elif regime == 'trending':
            adjustment_factor = 1.1  # Slightly increase reward in trending markets
        elif regime == 'range_bound':
            adjustment_factor = 0.9  # Slightly decrease reward in range-bound markets
        else:  # normal regime
            adjustment_factor = 1.0
            
        return reward * adjustment_factor
        
    def _apply_dynamic_threshold_adjustment(self, reward: float, market_data: Dict[str, MarketData]) -> float:
        """
        Apply dynamic threshold adjustment to the reward.
        
        Args:
            reward: Original reward value
            market_data: Current market data for all tickers
            
        Returns:
            Adjusted reward value
        """
        if not self.dynamic_thresholds or not market_data:
            return reward
            
        # Calculate average opportunity score
        opportunity_scores = []
        for ticker, data in market_data.items():
            score = self._calculate_opportunity_score(ticker, data)
            opportunity_scores.append(score)
            
        if opportunity_scores:
            avg_opportunity_score = np.mean(opportunity_scores)
            
            # Adjust opportunity threshold based on average score
            if avg_opportunity_score > 0.7:
                # High opportunity environment, increase threshold
                threshold_adjustment = 1.2
            elif avg_opportunity_score < 0.3:
                # Low opportunity environment, decrease threshold
                threshold_adjustment = 0.8
            else:
                # Normal environment
                threshold_adjustment = 1.0
                
            # Apply adjustment to reward
            reward *= threshold_adjustment
            
        return reward
```

#### 2. Opportunity
Data class for representing trading opportunities.

```python
@dataclass
class Opportunity:
    """
    Trading opportunity data class.
    """
    ticker: str
    score: float
    is_opportunity: bool
    timestamp: pd.Timestamp
    expected_return: float
    risk_level: float
    time_horizon: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert opportunity to dictionary."""
        return {
            'ticker': self.ticker,
            'score': self.score,
            'is_opportunity': self.is_opportunity,
            'timestamp': self.timestamp,
            'expected_return': self.expected_return,
            'risk_level': self.risk_level,
            'time_horizon': self.time_horizon
        }
```

#### 3. RegimeDetector
Component for detecting market regimes based on opportunity characteristics.

```python
class RegimeDetector:
    """
    Market regime detector based on opportunity characteristics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize regime detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.window = config.get('window', 20)
        self.volatility_threshold = config.get('volatility_threshold', 0.2)
        self.opportunity_threshold = config.get('opportunity_threshold', 0.5)
        self.feature_history = []
        
    def detect_regime(self, market_data: Dict[str, MarketData]) -> str:
        """
        Detect current market regime.
        
        Args:
            market_data: Current market data for all tickers
            
        Returns:
            Regime name
        """
        # Calculate regime features
        regime_features = self._calculate_regime_features(market_data)
        
        # Update feature history
        self.feature_history.append(regime_features)
        
        # Keep history limited
        if len(self.feature_history) > self.window:
            self.feature_history.pop(0)
            
        # Detect regime if we have enough history
        if len(self.feature_history) >= self.window:
            return self._classify_regime()
        else:
            return 'normal'
            
    def _calculate_regime_features(self, market_data: Dict[str, MarketData]) -> Dict[str, float]:
        """
        Calculate features for regime detection.
        
        Args:
            market_data: Current market data for all tickers
            
        Returns:
            Dictionary of regime features
        """
        features = {}
        
        # Calculate average volatility
        volatilities = []
        for ticker, data in market_data.items():
            if hasattr(data, 'volatility') and data.volatility is not None:
                volatilities.append(data.volatility)
                
        if volatilities:
            features['avg_volatility'] = np.mean(volatilities)
            features['volatility_dispersion'] = np.std(volatilities)
        else:
            features['avg_volatility'] = 0.0
            features['volatility_dispersion'] = 0.0
            
        # Calculate opportunity scores
        opportunity_scores = []
        for ticker, data in market_data.items():
            # Simple opportunity score based on momentum and volatility
            if hasattr(data, 'momentum') and data.momentum is not None:
                if hasattr(data, 'volatility') and data.volatility is not None and data.volatility > 0:
                    score = abs(data.momentum) / data.volatility
                    opportunity_scores.append(score)
                    
        if opportunity_scores:
            features['avg_opportunity_score'] = np.mean(opportunity_scores)
            features['opportunity_dispersion'] = np.std(opportunity_scores)
        else:
            features['avg_opportunity_score'] = 0.0
            features['opportunity_dispersion'] = 0.0
            
        # Calculate trend strength
        trends = []
        for ticker, data in market_data.items():
            if hasattr(data, 'trend_strength') and data.trend_strength is not None:
                trends.append(data.trend_strength)
                
        if trends:
            features['avg_trend_strength'] = np.mean(trends)
            features['trend_dispersion'] = np.std(trends)
        else:
            features['avg_trend_strength'] = 0.0
            features['trend_dispersion'] = 0.0
            
        return features
        
    def _classify_regime(self) -> str:
        """
        Classify current market regime based on feature history.
        
        Returns:
            Regime name
        """
        # Calculate feature averages over the window
        avg_volatility = np.mean([f['avg_volatility'] for f in self.feature_history])
        avg_opportunity = np.mean([f['avg_opportunity_score'] for f in self.feature_history])
        avg_trend = np.mean([f['avg_trend_strength'] for f in self.feature_history])
        
        # Classify regime
        if avg_volatility > self.volatility_threshold:
            return 'high_volatility'
        elif avg_volatility < self.volatility_threshold * 0.5:
            return 'low_volatility'
        elif avg_opportunity > self.opportunity_threshold:
            return 'high_opportunity'
        elif avg_opportunity < self.opportunity_threshold * 0.5:
            return 'low_opportunity'
        elif abs(avg_trend) > 0.7:
            return 'trending'
        else:
            return 'normal'
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Market Data     │    │ Portfolio       │    │ Actions         │
│ (Input)         │    │ State           │    │ (Input)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Opportunity     │    │ Capture         │    │ Opportunity     │
│ Identification  │    │ Efficiency      │    │ History         │
│                 │◀───│ Evaluation      │◀───│ (Update)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Opportunity/    │
│ Capture Reward  │
│ Calculation     │
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
│ Regime          │
│ Adjustment      │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Dynamic         │
│ Threshold       │
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

### Opportunity/Capture Shaping Configuration

```yaml
# Opportunity/capture shaping configuration
opportunity_capture_shaping:
  enabled: true
  weight: 0.3
  
  # Threshold parameters
  opportunity_threshold: 0.02  # 2% minimum opportunity
  capture_threshold: 0.5  # 50% capture ratio
  opportunity_window: 10  # Lookback window for opportunity detection
  capture_window: 5  # Window for capture evaluation
  
  # Feature parameters
  regime_aware: true  # Enable regime-aware adjustments
  dynamic_thresholds: true  # Enable dynamic threshold adjustments
  multi_ticker_correlation: true  # Enable multi-ticker correlation adjustments
  
  # Opportunity score components
  opportunity_components:
    momentum_weight: 0.3  # Weight for momentum component
    volatility_weight: 0.2  # Weight for volatility component
    volume_weight: 0.2  # Weight for volume component
    technical_weight: 0.15  # Weight for technical indicators component
    spread_weight: 0.15  # Weight for spread component
    
  # Regime detection parameters
  regime_detection:
    window: 20  # Window for regime detection
    volatility_threshold: 0.2  # Threshold for high volatility regime
    opportunity_threshold: 0.5  # Threshold for high opportunity regime
    
  # Reward calculation parameters
  reward_weights:
    opportunity_weight: 0.3  # Weight for opportunity identification
    capture_weight: 0.7  # Weight for opportunity capture
    
  # Multi-ticker correlation parameters
  correlation_adjustment:
    enabled: true
    correlation_matrix_file: "data/correlation_matrix.csv"  # Path to correlation matrix
    correlation_window: 252  # Window for correlation calculation
    correlation_threshold: 0.7  # Threshold for significant correlation
    
  # Normalization
  normalization:
    enabled: true
    method: standard  # standard, min_max, clip
    clip_threshold: 3.0
```

## Implementation Details

### Opportunity Identification Algorithm

```python
def _identify_opportunities(self, market_data: Dict[str, MarketData]) -> Dict[str, Opportunity]:
    """
    Identify trading opportunities for each ticker.
    
    Args:
        market_data: Current market data for all tickers
        
    Returns:
        Dictionary of ticker -> opportunity
    """
    opportunities = {}
    
    for ticker, data in market_data.items():
        # Calculate opportunity score
        opportunity_score = self._calculate_opportunity_score(ticker, data)
        
        # Determine if this is an opportunity
        is_opportunity = opportunity_score >= self.opportunity_threshold
        
        # Create opportunity object
        opportunity = Opportunity(
            ticker=ticker,
            score=opportunity_score,
            is_opportunity=is_opportunity,
            timestamp=data.timestamp if hasattr(data, 'timestamp') else pd.Timestamp.now(),
            expected_return=self._calculate_expected_return(ticker, data),
            risk_level=self._calculate_risk_level(ticker, data),
            time_horizon=self._calculate_time_horizon(ticker, data)
        )
        
        opportunities[ticker] = opportunity
        
        # Update opportunity history
        if ticker not in self.opportunity_history:
            self.opportunity_history[ticker] = []
            
        self.opportunity_history[ticker].append(opportunity)
        
        # Keep history limited
        if len(self.opportunity_history[ticker]) > self.opportunity_window:
            self.opportunity_history[ticker].pop(0)
            
    return opportunities
```

### Opportunity Score Calculation Algorithm

```python
def _calculate_opportunity_score(self, ticker: str, data: MarketData) -> float:
    """
    Calculate opportunity score for a ticker.
    
    Args:
        ticker: Ticker symbol
        data: Market data for the ticker
        
    Returns:
        Opportunity score
    """
    score = 0.0
    
    # Price momentum component
    if hasattr(data, 'momentum') and data.momentum is not None:
        score += abs(data.momentum) * 0.3
        
    # Volatility component
    if hasattr(data, 'volatility') and data.volatility is not None:
        # Moderate volatility is good for opportunities
        optimal_volatility = 0.2
        volatility_score = 1.0 - abs(data.volatility - optimal_volatility) / optimal_volatility
        score += max(0, volatility_score) * 0.2
        
    # Volume component
    if hasattr(data, 'volume') and hasattr(data, 'avg_volume') and data.avg_volume > 0:
        volume_ratio = data.volume / data.avg_volume
        # High volume is good for opportunities
        volume_score = min(volume_ratio, 2.0) / 2.0
        score += volume_score * 0.2
        
    # Technical indicators component
    if hasattr(data, 'rsi') and data.rsi is not None:
        # Extreme RSI values indicate potential reversal opportunities
        rsi_score = max(0, (abs(data.rsi - 50) - 30) / 20)
        score += rsi_score * 0.15
        
    # Spread component
    if hasattr(data, 'spread') and hasattr(data, 'price') and data.price > 0:
        relative_spread = data.spread / data.price
        # Lower spreads are better for opportunities
        spread_score = max(0, 1.0 - relative_spread * 1000)
        score += spread_score * 0.15
        
    return score
```

### Capture Efficiency Evaluation Algorithm

```python
def _evaluate_capture_efficiency(self, portfolio_state: PortfolioState,
                               market_data: Dict[str, MarketData],
                               actions: Dict[str, float],
                               previous_portfolio_state: PortfolioState,
                               opportunities: Dict[str, Opportunity]) -> Dict[str, float]:
    """
    Evaluate capture efficiency for each ticker.
    
    Args:
        portfolio_state: Current portfolio state
        market_data: Current market data for all tickers
        actions: Actions taken for each ticker
        previous_portfolio_state: Previous portfolio state
        opportunities: Current opportunities
        
    Returns:
        Dictionary of ticker -> capture efficiency
    """
    capture_efficiency = {}
    
    for ticker, opportunity in opportunities.items():
        if ticker in actions:
            action = actions[ticker]
            
            # Calculate capture efficiency
            efficiency = self._calculate_ticker_capture_efficiency(
                ticker, action, portfolio_state, previous_portfolio_state, opportunity
            )
            
            capture_efficiency[ticker] = efficiency
            
            # Update capture history
            if ticker not in self.capture_history:
                self.capture_history[ticker] = []
                
            self.capture_history[ticker].append({
                'timestamp': portfolio_state.timestamp,
                'efficiency': efficiency,
                'action': action,
                'opportunity_score': opportunity.score
            })
            
            # Keep history limited
            if len(self.capture_history[ticker]) > self.capture_window:
                self.capture_history[ticker].pop(0)
                
    return capture_efficiency
```

### Opportunity/Capture Reward Calculation Algorithm

```python
def _calculate_opportunity_capture_reward(self, opportunities: Dict[str, Opportunity],
                                        capture_efficiency: Dict[str, float]) -> float:
    """
    Calculate opportunity/capture reward.
    
    Args:
        opportunities: Current opportunities
        capture_efficiency: Capture efficiency for each ticker
        
    Returns:
        Opportunity/capture reward
    """
    reward = 0.0
    
    for ticker, opportunity in opportunities.items():
        if ticker in capture_efficiency:
            efficiency = capture_efficiency[ticker]
            
            if opportunity.is_opportunity:
                # Reward for identifying opportunities
                opportunity_reward = opportunity.score * 0.3
                
                # Reward for capturing opportunities
                capture_reward = efficiency * opportunity.score * 0.7
                
                # Total reward for this ticker
                ticker_reward = opportunity_reward + capture_reward
                
                # Apply multi-ticker correlation adjustment if enabled
                if self.multi_ticker_correlation:
                    ticker_reward = self._apply_multi_ticker_correlation_adjustment(
                        ticker_reward, ticker, opportunities
                    )
                    
                reward += ticker_reward
                
    # Normalize by number of tickers
    if opportunities:
        reward /= len(opportunities)
        
    return reward
```

## Best Practices

### Opportunity Identification
1. **Diverse Signals**: Use a diverse set of signals to identify opportunities
2. **Risk-Adjusted**: Consider risk when evaluating opportunities
3. **Time Horizon**: Define appropriate time horizons for different opportunity types
4. **Market Regimes**: Adjust opportunity criteria based on market regimes
5. **Multi-Ticker**: Consider cross-ticker relationships when identifying opportunities

### Capture Efficiency
1. **Action Alignment**: Evaluate how well actions align with identified opportunities
2. **Position Sizing**: Consider position size when evaluating capture efficiency
3. **Timing**: Evaluate the timing of actions relative to opportunities
4. **Execution Quality**: Consider execution quality in capture efficiency
5. **Persistence**: Track capture efficiency over time to identify patterns

### Configuration Management
1. **Default Values**: Provide sensible default values for all parameters
2. **Validation**: Validate configuration parameters to ensure they are within reasonable ranges
3. **Documentation**: Document all configuration parameters and their effects
4. **Environment Overrides**: Support environment-specific configuration overrides
5. **Versioning**: Version configuration schemas to ensure backward compatibility

### Performance Optimization
1. **Efficient Calculation**: Use efficient algorithms for opportunity detection
2. **Caching**: Cache expensive calculations to improve performance
3. **Vectorization**: Use vectorized operations for efficient computation
4. **Memory Management**: Manage memory usage for opportunity and capture histories
5. **Lazy Evaluation**: Only evaluate opportunities when needed

### Monitoring and Debugging
1. **Logging**: Log opportunity identification and capture efficiency for debugging
2. **Visualization**: Visualize opportunity scores and capture efficiency over time
3. **Alerting**: Alert on persistent low capture efficiency
4. **Metrics**: Track opportunity-related metrics over time
5. **A/B Testing**: Test different opportunity/capture configurations to find optimal settings