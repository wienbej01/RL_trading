# Multi-Ticker RL Environment with Portfolio Management Design

## Overview

This document outlines the design for extending the existing single-ticker RL environment to support multiple tickers with portfolio management capabilities. The multi-ticker RL environment will provide a realistic trading simulation where the agent can manage positions across multiple instruments simultaneously.

## Requirements

### Functional Requirements
1. **Multi-Ticker Support**: Support trading across multiple tickers simultaneously
2. **Portfolio Management**: Track and manage positions across all tickers
3. **Risk Management**: Implement portfolio-level risk controls
4. **Position Sizing**: Dynamic position sizing based on portfolio constraints
5. **Capital Allocation**: Allocate capital across tickers based on strategy signals
6. **Transaction Costs**: Model realistic transaction costs for multi-ticker trading
7. **Performance Metrics**: Track portfolio-level performance metrics

### Non-Functional Requirements
1. **Performance**: Efficient simulation of multi-ticker trading
2. **Scalability**: Support for 10+ tickers with complex portfolio interactions
3. **Realism**: Realistic market microstructure and execution modeling
4. **Flexibility**: Configurable portfolio management strategies
5. **Backward Compatibility**: Maintain compatibility with existing single-ticker workflows

## Architecture Overview

### Core Components

#### 1. MultiTickerRLEnvironment
The main environment class that extends the existing IntradayRLEnv.

```python
class MultiTickerRLEnvironment(IntradayRLEnv):
    """
    Multi-ticker RL environment with portfolio management.
    
    Extends the single-ticker environment to support trading across multiple
    instruments with portfolio-level risk management and position sizing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-ticker RL environment.
        
        Args:
            config: Configuration dictionary with environment settings
        """
        super().__init__(config)
        self.tickers = config.get('tickers', [])
        self.portfolio_manager = PortfolioManager(config.get('portfolio', {}))
        self.position_sizer = PositionSizer(config.get('position_sizing', {}))
        self.risk_manager = RiskManager(config.get('risk_management', {}))
        self.execution_simulator = ExecutionSimulator(config.get('execution', {}))
        self.multi_ticker_observation_space = self._build_multi_ticker_observation_space()
        self.multi_ticker_action_space = self._build_multi_ticker_action_space()
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial observation
        """
        pass
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Action to take (portfolio allocation across tickers)
            
        Returns:
            observation, reward, done, info
        """
        pass
```

#### 2. PortfolioManager
Component responsible for tracking and managing the portfolio state.

```python
class PortfolioManager:
    """
    Portfolio management component for multi-ticker trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize portfolio manager.
        
        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.initial_capital = config.get('initial_capital', 1_000_000)
        self.current_capital = self.initial_capital
        self.positions = {}  # ticker -> Position
        self.portfolio_history = []
        self.performance_metrics = {}
        
    def update_portfolio(self, trades: List[Trade], market_data: Dict[str, MarketData]):
        """
        Update portfolio state with new trades and market data.
        
        Args:
            trades: List of trades executed
            market_data: Current market data for all tickers
        """
        pass
        
    def get_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state.
        
        Returns:
            PortfolioState object with current portfolio information
        """
        pass
        
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        pass
```

#### 3. PositionSizer
Component responsible for determining position sizes across tickers.

```python
class PositionSizer:
    """
    Position sizing component for multi-ticker trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize position sizer.
        
        Args:
            config: Position sizing configuration
        """
        self.config = config
        self.sizing_method = config.get('method', 'risk_parity')  # risk_parity, equal_weight, kelly, signal_strength
        self.max_position_size = config.get('max_position_size', 0.2)  # 20% of portfolio
        self.min_position_size = config.get('min_position_size', 0.01)  # 1% of portfolio
        
    def calculate_position_sizes(self, signals: Dict[str, float], 
                                portfolio_state: PortfolioState,
                                market_data: Dict[str, MarketData]) -> Dict[str, float]:
        """
        Calculate position sizes for all tickers.
        
        Args:
            signals: Trading signals for each ticker
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Dictionary of ticker -> position size (as fraction of portfolio)
        """
        pass
```

#### 4. RiskManager
Component responsible for portfolio-level risk management.

```python
class RiskManager:
    """
    Risk management component for multi-ticker trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 2% daily VaR
        self.max_drawdown = config.get('max_drawdown', 0.1)  # 10% max drawdown
        self.max_correlation = config.get('max_correlation', 0.7)  # Max correlation between positions
        self.risk_models = {
            'var': self._calculate_var,
            'expected_shortfall': self._calculate_expected_shortfall,
            'max_drawdown': self._calculate_max_drawdown,
            'correlation': self._calculate_correlation_matrix
        }
        
    def check_risk_constraints(self, portfolio_state: PortfolioState,
                             proposed_trades: List[Trade]) -> RiskCheckResult:
        """
        Check if proposed trades violate risk constraints.
        
        Args:
            portfolio_state: Current portfolio state
            proposed_trades: List of proposed trades
            
        Returns:
            RiskCheckResult with constraint violations
        """
        pass
        
    def adjust_positions_for_risk(self, position_sizes: Dict[str, float],
                                portfolio_state: PortfolioState,
                                market_data: Dict[str, MarketData]) -> Dict[str, float]:
        """
        Adjust position sizes to meet risk constraints.
        
        Args:
            position_sizes: Proposed position sizes
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Adjusted position sizes
        """
        pass
```

#### 5. ExecutionSimulator
Component responsible for simulating trade execution across multiple tickers.

```python
class ExecutionSimulator:
    """
    Execution simulator for multi-ticker trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize execution simulator.
        
        Args:
            config: Execution configuration
        """
        self.config = config
        self.commission_model = CommissionModel(config.get('commission', {}))
        self.slippage_model = SlippageModel(config.get('slippage', {}))
        self.market_impact_model = MarketImpactModel(config.get('market_impact', {}))
        
    def execute_trades(self, trade_requests: List[TradeRequest],
                      market_data: Dict[str, MarketData]) -> List[Trade]:
        """
        Execute trade requests with realistic execution modeling.
        
        Args:
            trade_requests: List of trade requests
            market_data: Current market data
            
        Returns:
            List of executed trades
        """
        pass
```

### Data Structures

#### 1. PortfolioState
Data structure representing the current portfolio state.

```python
@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_value: float
    cash: float
    positions: Dict[str, Position]  # ticker -> Position
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    metrics: Dict[str, float]
```

#### 2. Position
Data structure representing a position in a single ticker.

```python
@dataclass
class Position:
    """Position in a single ticker."""
    ticker: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    trades: List[Trade]
```

#### 3. Trade
Data structure representing a single trade.

```python
@dataclass
class Trade:
    """Single trade execution."""
    ticker: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    trade_id: str
```

#### 4. TradeRequest
Data structure representing a trade request.

```python
@dataclass
class TradeRequest:
    """Trade request for execution."""
    ticker: str
    quantity: float
    order_type: str  # market, limit, etc.
    time_in_force: str  # day, gtc, etc.
    limit_price: Optional[float] = None
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MultiTicker     │    │ Portfolio       │    │ Position        │
│ RLEnvironment   │───▶│ Manager        │───▶│ Sizer          │
│ (Main Env)      │    │ (State Mgmt)    │    │ (Sizing)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      │                      │
         │                      ▼                      ▼
         │                ┌─────────────────┐    ┌─────────────────┐
         │                │ Risk Manager    │    │ Execution       │
         │                │ (Risk Control)  │    │ Simulator       │
         │                └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      │                      │
         │                      ▼                      ▼
         │                ┌─────────────────┐    ┌─────────────────┐
         └────────────────▶│ Market Data     │    │ Trade History   │
                          │ (Input)         │    │ (Output)        │
                          └─────────────────┘    └─────────────────┘
```

## Configuration Structure

### Multi-Ticker Environment Configuration

```yaml
# Multi-ticker RL environment configuration
multi_ticker_env:
  enabled: true
  
  # Ticker configuration
  tickers:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - META
  
  # Portfolio management
  portfolio:
    initial_capital: 1_000_000
    base_currency: USD
    benchmark: SPY
    rebalance_frequency: 1min  # How often to rebalance portfolio
    
  # Position sizing
  position_sizing:
    method: risk_parity  # risk_parity, equal_weight, kelly, signal_strength
    max_position_size: 0.2  # Maximum position size as fraction of portfolio
    min_position_size: 0.01  # Minimum position size as fraction of portfolio
    max_leverage: 1.0  # Maximum leverage
    risk_parity_params:
      target_risk: 0.02  # Target risk per position
      risk_budget: equal  # equal, custom
    kelly_params:
      fraction: 0.25  # Kelly fraction to use
      min_kelly: 0.01  # Minimum Kelly allocation
    
  # Risk management
  risk_management:
    max_portfolio_risk: 0.02  # Maximum daily portfolio risk (VaR)
    max_drawdown: 0.1  # Maximum drawdown threshold
    max_correlation: 0.7  # Maximum correlation between positions
    position_limits:
      max_long_positions: 5  # Maximum number of long positions
      max_short_positions: 2  # Maximum number of short positions
    stop_loss:
      enabled: true
      threshold: 0.05  # 5% stop loss
    take_profit:
      enabled: true
      threshold: 0.1  # 10% take profit
    
  # Execution simulation
  execution:
    commission:
      model: tiered  # fixed, percentage, tiered
      fixed_fee: 1.0  # Fixed fee per trade
      percentage_fee: 0.001  # Percentage fee
      tiered_rates:
        - {min_volume: 0, max_volume: 1000, rate: 0.001}
        - {min_volume: 1000, max_volume: 10000, rate: 0.0008}
        - {min_volume: 10000, max_volume: inf, rate: 0.0005}
    slippage:
      model: linear  # linear, square_root, percentage
      base_slippage: 0.0001  # Base slippage rate
      volume_impact: 0.00000001  # Volume impact factor
    market_impact:
      model: square_root  # linear, square_root, permanent_temporary
      permanent_impact: 0.1  # Permanent market impact coefficient
      temporary_impact: 0.05  # Temporary market impact coefficient
      participation_rate_limit: 0.1  # Maximum participation rate
    
  # Reward configuration
  reward:
    type: portfolio_pnl  # portfolio_pnl, sharpe, sortino, calmar, custom
    risk_adjusted: true
    transaction_cost_penalty: 0.5  # Penalty factor for transaction costs
    drawdown_penalty: 1.0  # Penalty factor for drawdown
    reward_scaling: 1.0  # Scaling factor for rewards
    
  # Observation space
  observation_space:
    include_portfolio_state: true
    include_positions: true
    include_market_data: true
    include_technical_features: true
    include_risk_metrics: true
    normalization:
      method: rolling  # standard, robust, rolling
      window: 20
    
  # Action space
  action_space:
    type: continuous  # continuous, discrete, multi_discrete
    action_type: portfolio_weights  # portfolio_weights, position_changes, trade_signals
    constraints:
      sum_to_one: true  # Portfolio weights must sum to 1
      no_shorting: false  # Allow short positions
      max_leverage: 1.0  # Maximum leverage
```

## Implementation Details

### Multi-Ticker Observation Space

#### 1. Observation Structure
The observation space will include:
- Portfolio state (total value, cash, positions)
- Market data for all tickers (prices, volumes, spreads)
- Technical features for all tickers
- Risk metrics (VaR, correlation, drawdown)
- Position information for all tickers

#### 2. Observation Normalization
- Rolling normalization for market data
- Portfolio value normalization
- Position size normalization
- Risk metric normalization

#### 3. Observation Encoding
- Ticker-specific features with ticker prefix
- Portfolio-level features without prefix
- Time-aware features (time of day, day of week)
- Regime indicators

### Multi-Ticker Action Space

#### 1. Action Types
- **Portfolio Weights**: Direct allocation to each ticker
- **Position Changes**: Changes to current positions
- **Trade Signals**: Buy/sell/hold signals for each ticker

#### 2. Action Constraints
- Sum-to-one constraint for portfolio weights
- Position limits (long/short)
- Leverage constraints
- Risk-based constraints

#### 3. Action Processing
- Convert actions to trade requests
- Apply position sizing rules
- Apply risk management constraints
- Execute trades with simulation

### Portfolio Management

#### 1. Portfolio State Tracking
- Track positions for all tickers
- Calculate unrealized and realized P&L
- Track portfolio metrics
- Maintain trade history

#### 2. Performance Metrics
- Total return and annualized return
- Volatility and risk-adjusted metrics
- Drawdown analysis
- Attribution analysis

#### 3. Rebalancing Logic
- Time-based rebalancing
- Threshold-based rebalancing
- Risk-based rebalancing
- Cost-aware rebalancing

### Risk Management

#### 1. Risk Models
- Value at Risk (VaR)
- Expected Shortfall (ES)
- Maximum Drawdown
- Correlation analysis

#### 2. Risk Constraints
- Portfolio-level risk limits
- Position-level risk limits
- Concentration limits
- Leverage limits

#### 3. Risk Adjustment
- Position size adjustment
- Portfolio rebalancing
- Hedging strategies
- Stop-loss mechanisms

### Execution Simulation

#### 1. Commission Models
- Fixed fee per trade
- Percentage of trade value
- Tiered pricing
- Exchange-specific fees

#### 2. Slippage Models
- Linear slippage
- Square root slippage
- Percentage slippage
- Volume-dependent slippage

#### 3. Market Impact
- Permanent vs. temporary impact
- Participation rate limits
- Price impact functions
- Liquidity considerations

## Performance Considerations

### Computational Efficiency
1. **Vectorization**: Use vectorized operations for portfolio calculations
2. **Caching**: Cache expensive calculations
3. **Incremental Updates**: Update portfolio state incrementally
4. **Parallel Processing**: Process multiple tickers in parallel

### Memory Optimization
1. **Data Structures**: Use efficient data structures
2. **Memory Pooling**: Reuse memory objects
3. **Data Compression**: Compress historical data
4. **Garbage Collection**: Implement proper memory management

### Simulation Speed
1. **Event-Driven Architecture**: Use event-driven simulation
2. **Batch Processing**: Process multiple time steps in batches
3. **Lazy Evaluation**: Defer expensive calculations
4. **Optimization**: Profile and optimize hot paths

## Testing Strategy

### Unit Tests
1. **Individual Components**: Test each component in isolation
2. **Edge Cases**: Test boundary conditions and error scenarios
3. **Configuration**: Test various configuration combinations
4. **Data Structures**: Test data structure operations

### Integration Tests
1. **End-to-End**: Test complete environment workflow
2. **Multi-Ticker**: Test with multiple tickers
3. **Portfolio Management**: Test portfolio operations
4. **Risk Management**: Test risk constraint enforcement

### Performance Tests
1. **Scalability**: Test with increasing numbers of tickers
2. **Memory Usage**: Monitor memory consumption
3. **Simulation Speed**: Measure simulation performance
4. **Throughput**: Test action processing throughput

## Migration Path

### Phase 1: Backward Compatibility
1. Maintain existing single-ticker environment
2. Add multi-ticker functionality as an option
3. Ensure existing tests continue to pass

### Phase 2: Coexistence
1. Support both single and multi-ticker environments
2. Provide migration utilities
3. Update documentation and examples

### Phase 3: Multi-Ticker First
1. Make multi-ticker the primary environment
2. Maintain single-ticker compatibility layer
3. Deprecate single-ticker specific features

## Future Enhancements

### Advanced Portfolio Management
1. **Multi-Period Optimization**: Multi-period portfolio optimization
2. **Transaction Cost Optimization**: Optimize trading costs
3. **Tax Optimization**: Tax-aware portfolio management
4. **ESG Integration**: Environmental, Social, Governance factors

### Advanced Risk Management
1. **Dynamic Risk Models**: Adaptive risk models
2. **Stress Testing**: Portfolio stress testing
3. **Liquidity Risk**: Liquidity risk management
4. **Counterparty Risk**: Counterparty risk modeling

### Advanced Execution
1. **Algorithmic Trading**: Advanced execution algorithms
2. **Market Making**: Market making strategies
3. **Dark Pools**: Dark pool execution
4. **Cross-Exchange Trading**: Multi-exchange execution

## Conclusion

The multi-ticker RL environment with portfolio management provides a realistic and flexible simulation environment for training reinforcement learning agents. By extending the existing single-ticker environment with portfolio management capabilities, we can train agents that make intelligent trading decisions across multiple instruments.

The design emphasizes modularity, with clear separation of concerns between portfolio management, position sizing, risk management, and execution simulation. This approach allows for independent development and testing of each component while ensuring they work together seamlessly.

The architecture is designed to scale to support 10+ tickers with complex portfolio interactions, realistic execution modeling, and sophisticated risk management. This provides a solid foundation for developing advanced multi-ticker trading strategies.