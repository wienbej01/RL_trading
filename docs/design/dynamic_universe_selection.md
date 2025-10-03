
# Dynamic Universe Selection Mechanism Design

## Overview

This document outlines the design for a dynamic universe selection mechanism that enables the RL trading system to adaptively select which tickers to trade based on market conditions, liquidity, volatility, and other factors. The dynamic universe selection will provide the flexibility to focus on the most promising trading opportunities while managing risk and computational resources.

## Requirements

### Functional Requirements
1. **Dynamic Selection**: Dynamically select tickers for trading based on configurable criteria
2. **Multiple Selection Criteria**: Support various selection criteria (liquidity, volatility, trend, etc.)
3. **Regime Awareness**: Adapt universe selection based on market regimes
4. **Performance-Based Selection**: Include/exclude tickers based on historical performance
5. **Risk Management**: Ensure selected universe meets risk constraints
6. **Diversity Maintenance**: Maintain diversity in the selected universe
7. **Universe Stability**: Balance between responsiveness and stability

### Non-Functional Requirements
1. **Performance**: Efficient universe selection computation
2. **Scalability**: Support for large ticker universes (100+ tickers)
3. **Stability**: Avoid frequent universe changes
4. **Transparency**: Clear, interpretable selection criteria
5. **Configurability**: Flexible configuration of selection parameters

## Architecture Overview

### Core Components

#### 1. UniverseSelector
The main class responsible for coordinating the universe selection process.

```python
class UniverseSelector:
    """
    Dynamic universe selector for multi-ticker trading.
    
    Coordinates the selection of tickers based on multiple criteria,
    market conditions, and performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize universe selector.
        
        Args:
            config: Configuration dictionary with universe selection settings
        """
        self.config = config
        self.selection_criteria = self._initialize_selection_criteria()
        self.regime_aware_selector = RegimeAwareSelector(config.get('regime_aware', {}))
        self.performance_tracker = PerformanceTracker(config.get('performance_tracking', {}))
        self.diversity_manager = DiversityManager(config.get('diversity', {}))
        self.stability_controller = StabilityController(config.get('stability', {}))
        self.current_universe = []
        self.selection_history = []
        
    def select_universe(self, market_data: Dict[str, MarketData],
                       performance_data: Dict[str, PerformanceData],
                       current_regime: str) -> List[str]:
        """
        Select the trading universe for the current period.
        
        Args:
            market_data: Current market data for all candidate tickers
            performance_data: Performance data for all candidate tickers
            current_regime: Current market regime
            
        Returns:
            List of selected ticker symbols
        """
        pass
        
    def get_selection_rationale(self) -> Dict[str, Any]:
        """
        Get rationale for the current universe selection.
        
        Returns:
            Dictionary with selection rationale
        """
        pass
```

#### 2. SelectionCriterion
Base class for individual selection criteria.

```python
class SelectionCriterion(ABC):
    """
    Base class for universe selection criteria.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize selection criterion.
        
        Args:
            config: Criterion-specific configuration
        """
        self.config = config
        self.weight = config.get('weight', 1.0)
        self.threshold = config.get('threshold', 0.5)
        self.enabled = config.get('enabled', True)
        self.name = self.get_name()
        
    @abstractmethod
    def calculate_score(self, ticker: str, market_data: MarketData,
                       performance_data: PerformanceData) -> float:
        """
        Calculate selection score for a ticker.
        
        Args:
            ticker: Ticker symbol
            market_data: Market data for the ticker
            performance_data: Performance data for the ticker
            
        Returns:
            Selection score (higher is better)
        """
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """
        Get criterion name.
        
        Returns:
            Criterion name
        """
        pass
        
    def is_passing_threshold(self, score: float) -> bool:
        """
        Check if score passes threshold.
        
        Args:
            score: Selection score
            
        Returns:
            True if score passes threshold
        """
        return score >= self.threshold
```

#### 3. RegimeAwareSelector
Component for adapting universe selection based on market regimes.

```python
class RegimeAwareSelector:
    """
    Regime-aware universe selection component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize regime-aware selector.
        
        Args:
            config: Regime-aware selection configuration
        """
        self.config = config
        self.regime_criteria_weights = config.get('regime_criteria_weights', {})
        self.regime_universe_sizes = config.get('regime_universe_sizes', {})
        self.regime_sector_preferences = config.get('regime_sector_preferences', {})
        
    def get_criteria_weights(self, regime: str) -> Dict[str, float]:
        """
        Get criteria weights for the current regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary of criterion name -> weight
        """
        pass
        
    def get_universe_size(self, regime: str) -> int:
        """
        Get target universe size for the current regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Target universe size
        """
        pass
        
    def get_sector_preferences(self, regime: str) -> Dict[str, float]:
        """
        Get sector preferences for the current regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary of sector -> preference weight
        """
        pass
```

#### 4. PerformanceTracker
Component for tracking and evaluating ticker performance.

```python
class PerformanceTracker:
    """
    Performance tracking component for universe selection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance tracker.
        
        Args:
            config: Performance tracking configuration
        """
        self.config = config
        self.lookback_window = config.get('lookback_window', 20)
        self.performance_metrics = config.get('metrics', ['sharpe', 'sortino', 'calmar'])
        self.performance_history = {}
        self.performance_scores = {}
        
    def update_performance(self, ticker: str, performance_data: PerformanceData):
        """
        Update performance data for a ticker.
        
        Args:
            ticker: Ticker symbol
            performance_data: Performance data for the ticker
        """
        pass
        
    def calculate_performance_score(self, ticker: str) -> float:
        """
        Calculate composite performance score for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Composite performance score
        """
        pass
        
    def get_performance_ranking(self, tickers: List[str]) -> List[Tuple[str, float]]:
        """
        Get performance ranking for a list of tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            List of (ticker, score) tuples sorted by score
        """
        pass
```

#### 5. DiversityManager
Component for ensuring diversity in the selected universe.

```python
class DiversityManager:
    """
    Diversity management component for universe selection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize diversity manager.
        
        Args:
            config: Diversity management configuration
        """
        self.config = config
        self.max_sector_concentration = config.get('max_sector_concentration', 0.3)
        self.max_correlation = config.get('max_correlation', 0.7)
        self.min_sectors = config.get('min_sectors', 3)
        self.sector_mapping = config.get('sector_mapping', {})
        
    def ensure_diversity(self, candidates: List[Tuple[str, float]],
                         market_data: Dict[str, MarketData]) -> List[str]:
        """
        Ensure diversity in the selected universe.
        
        Args:
            candidates: List of (ticker, score) tuples
            market_data: Market data for all tickers
            
        Returns:
            Diversified list of ticker symbols
        """
        pass
        
    def calculate_diversity_score(self, universe: List[str],
                                market_data: Dict[str, MarketData]) -> float:
        """
        Calculate diversity score for a universe.
        
        Args:
            universe: List of ticker symbols
            market_data: Market data for all tickers
            
        Returns:
            Diversity score (higher is better)
        """
        pass
```

#### 6. StabilityController
Component for controlling universe stability.

```python
class StabilityController:
    """
    Stability control component for universe selection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stability controller.
        
        Args:
            config: Stability control configuration
        """
        self.config = config
        self.max_turnover = config.get('max_turnover', 0.2)
        self.min_hold_period = config.get('min_hold_period', 5)
        self.stability_weight = config.get('stability_weight', 0.3)
        self.previous_universe = []
        self.hold_periods = {}
        
    def apply_stability_constraints(self, new_universe: List[str],
                                  current_universe: List[str]) -> List[str]:
        """
        Apply stability constraints to universe selection.
        
        Args:
            new_universe: Newly selected universe
            current_universe: Current universe
            
        Returns:
            Stability-adjusted universe
        """
        pass
        
    def update_hold_periods(self, universe: List[str]):
        """
        Update hold periods for tickers in the universe.
        
        Args:
            universe: Current universe
        """
        pass
```

### Selection Criteria

#### 1. LiquidityCriterion
Selection criterion based on liquidity metrics.

```python
class LiquidityCriterion(SelectionCriterion):
    """
    Liquidity-based selection criterion.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_volume = config.get('min_volume', 1_000_000)
        self.min_trades = config.get('min_trades', 100)
        self.max_spread = config.get('max_spread', 0.01)
        
    def calculate_score(self, ticker: str, market_data: MarketData,
                       performance_data: PerformanceData) -> float:
        """
        Calculate liquidity score.
        
        Args:
            ticker: Ticker symbol
            market_data: Market data for the ticker
            performance_data: Performance data for the ticker
            
        Returns:
            Liquidity score
        """
        # Calculate volume score
        volume_score = min(market_data.volume / self.min_volume, 1.0)
        
        # Calculate trades score
        trades_score = min(market_data.trades / self.min_trades, 1.0) if hasattr(market_data, 'trades') else 1.0
        
        # Calculate spread score (inverse of spread)
        spread = (market_data.ask - market_data.bid) / market_data.mid_price if hasattr(market_data, 'bid') else 0
        spread_score = max(1.0 - spread / self.max_spread, 0.0)
        
        # Composite score
        score = (volume_score + trades_score + spread_score) / 3.0
        return score * self.weight
        
    def get_name(self) -> str:
        return "liquidity"
```

#### 2. VolatilityCriterion
Selection criterion based on volatility metrics.

```python
class VolatilityCriterion(SelectionCriterion):
    """
    Volatility-based selection criterion.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_volatility = config.get('target_volatility', 0.2)
        self.volatility_window = config.get('volatility_window', 20)
        self.volatility_tolerance = config.get('volatility_tolerance', 0.1)
        
    def calculate_score(self, ticker: str, market_data: MarketData,
                       performance_data: PerformanceData) -> float:
        """
        Calculate volatility score.
        
        Args:
            ticker: Ticker symbol
            market_data: Market data for the ticker
            performance_data: Performance data for the ticker
            
        Returns:
            Volatility score
        """
        # Calculate rolling volatility
        if len(performance_data.returns) >= self.volatility_window:
            recent_returns = performance_data.returns[-self.volatility_window:]
            volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
            
            # Score based on distance from target volatility
            distance = abs(volatility - self.target_volatility)
            score = max(1.0 - distance / self.volatility_tolerance, 0.0)
        else:
            score = 0.5  # Neutral score if insufficient data
            
        return score * self.weight
        
    def get_name(self) -> str:
        return "volatility"
```

#### 3. TrendCriterion
Selection criterion based on trend strength.

```python
class TrendCriterion(SelectionCriterion):
    """
    Trend-based selection criterion.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trend_window = config.get('trend_window', 20)
        self.min_trend_strength = config.get('min_trend_strength', 0.1)
        
    def calculate_score(self, ticker: str, market_data: MarketData,
                       performance_data: PerformanceData) -> float:
        """
        Calculate trend score.
        
        Args:
            ticker: Ticker symbol
            market_data: Market data for the ticker
            performance_data: Performance data for the ticker
            
        Returns:
            Trend score
        """
        # Calculate trend strength using linear regression
        if len(market_data.close) >= self.trend_window:
            recent_prices = market_data.close[-self.trend_window:]
            x = np.arange(len(recent_prices))
            slope, _ = np.polyfit(x, recent_prices, 1)
            
            # Normalize slope by price level
            normalized_slope = slope / np.mean(recent_prices)
            
            # Score based on trend strength
            score = min(abs(normalized_slope) / self.min_trend_strength, 1.0)
        else:
            score = 0.0  # No score if insufficient data
            
        return score * self.weight
        
    def get_name(self) -> str:
        return "trend"
```

#### 4. PerformanceCriterion
Selection criterion based on historical performance.

```python
class PerformanceCriterion(SelectionCriterion):
    """
    Performance-based selection criterion.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.performance_window = config.get('performance_window', 20)
        self.performance_metrics = config.get('metrics', ['sharpe', 'sortino'])
        self.metric_weights = config.get('metric_weights', {'sharpe': 0.6, 'sortino': 0.4})
        
    def calculate_score(self, ticker: str, market_data: MarketData,
                       performance_data: PerformanceData) -> float:
        """
        Calculate performance score.
        
        Args:
            ticker: Ticker symbol
            market_data: Market data for the ticker
            performance_data: Performance data for the ticker
            
        Returns:
            Performance score
        """
        if len(performance_data.returns) < self.performance_window:
            return 0.0  # No score if insufficient data
            
        recent_returns = performance_data.returns[-self.performance_window:]
        
        # Calculate individual metric scores
        metric_scores = {}
        
        # Sharpe ratio
        if 'sharpe' in self.performance_metrics:
            sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
            metric_scores['sharpe'] = max(sharpe / 2.0, 0.0)  # Normalize by target Sharpe of 2.0
            
        # Sortino ratio
        if 'sortino' in self.performance_metrics:
            downside_returns = recent_returns[recent_returns < 0]
            if len(downside_returns) > 0:
                sortino = np.mean(recent_returns) / np.std(downside_returns) * np.sqrt(252)
                metric_scores['sortino'] = max(sortino / 2.0, 0.0)  # Normalize by target Sortino of 2.0
            else:
                metric_scores['sortino'] = 1.0  # Perfect Sortino if no downside
                
        # Weighted composite score
        score = sum(metric_scores[metric] * self.metric_weights.get(metric, 0.0)
                   for metric in self.performance_metrics if metric in metric_scores)
        
        return score * self.weight
        
    def get_name(self) -> str:
        return "performance"
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Universe        │    │ Selection       │    │ Market Data     │
│ Selector        │───▶│ Criteria        │───▶│ (Input)         │
│ (Coordinator)   │    │ (Scoring)       │    └─────────────────┘
└─────────────────┘    └─────────────────┘         │
         │                      │                  │
         │                      │                  ▼
         │                      │            ┌─────────────────┐
         │                      │            │ Performance     │
         │                      │            │ Data           │
         │                      │            └─────────────────┘
         │                      │                  │
         │                      ▼                  ▼
         │                ┌─────────────────┐    ┌─────────────────┐
         │                │ RegimeAware     │    │ Current Regime  │
         │                │ Selector        │    │ (Input)         │
         │                └─────────────────┘    └─────────────────┘
         │                      │
         │                      ▼
         │                ┌─────────────────┐
         │                │ Performance     │
         │                │ Tracker        │
         │                └─────────────────┘
         │                      │
         │                      ▼
         │                ┌─────────────────┐
         │                │ Diversity       │
         │                │ Manager        │
         │                └─────────────────┘
         │                      │
         │                      ▼
         │                ┌─────────────────┐
         └────────────────▶│ Stability       │
                          │ Controller     │
                          └─────────────────┘
                                 │
                                 ▼
                          ┌─────────────────┐
                          │ Selected        │
                          │ Universe       │
                          │ (Output)        │
                          └─────────────────┘
```

## Configuration Structure

### Dynamic Universe Selection Configuration

```yaml
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
        correlation: 0.