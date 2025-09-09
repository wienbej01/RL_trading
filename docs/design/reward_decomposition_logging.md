# Reward Decomposition Logging

## Overview

This document outlines the design for reward decomposition logging in the multi-ticker RL trading system. Reward decomposition logging provides detailed insights into the contribution of each reward component to the total reward, enabling better understanding, debugging, and optimization of the reward function.

## Requirements

### Functional Requirements
1. **Component Tracking**: Track individual reward components (pnl, dsr, sharpe, blend, directional, hybrid, hybrid2, potential-based)
2. **Multi-Ticker Support**: Log reward decomposition for each ticker individually and at portfolio level
3. **Time Series Logging**: Maintain time series of reward components for analysis
4. **Statistical Analysis**: Calculate statistics for each reward component
5. **Visualization Support**: Provide data structures suitable for visualization
6. **Export Capability**: Enable export of reward decomposition data for external analysis
7. **Real-time Monitoring**: Support real-time monitoring of reward components

### Non-Functional Requirements
1. **Performance**: Efficient logging with minimal impact on training performance
2. **Scalability**: Handle multiple tickers and high-frequency data
3. **Storage Efficiency**: Optimize storage usage for large amounts of log data
4. **Accessibility**: Easy access to historical reward decomposition data
5. **Configurability**: Configurable level of detail and logging frequency

## Architecture Overview

### Core Components

#### 1. RewardDecompositionLogger
The main class responsible for logging reward decomposition data.

```python
class RewardDecompositionLogger:
    """
    Logger for reward decomposition data.
    
    Tracks individual reward components and their contributions
    to the total reward for analysis and debugging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward decomposition logger.
        
        Args:
            config: Configuration dictionary with logging settings
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.log_frequency = config.get('log_frequency', 1)  # Log every N steps
        self.max_history = config.get('max_history', 10000)  # Maximum entries to keep
        self.storage_path = config.get('storage_path', 'data/reward_decomposition')
        self.compression = config.get('compression', True)
        self.real_time_monitoring = config.get('real_time_monitoring', True)
        
        # Data structures for storing decomposition data
        self.reward_history = []  # List of reward decomposition entries
        self.component_stats = {}  # Statistics for each component
        self.ticker_stats = {}  # Statistics for each ticker
        self.portfolio_stats = {}  # Portfolio-level statistics
        
        # Create storage directory if it doesn't exist
        if self.enabled and self.storage_path:
            os.makedirs(self.storage_path, exist_ok=True)
            
        # Initialize monitoring if enabled
        if self.real_time_monitoring:
            self.monitor = RewardDecompositionMonitor(config.get('monitoring', {}))
            
        # Step counter for frequency-based logging
        self.step_count = 0
        
    def log_reward(self, reward_components: Dict[str, float],
                  portfolio_state: PortfolioState,
                  market_data: Dict[str, MarketData],
                  actions: Dict[str, float],
                  timestamp: Optional[pd.Timestamp] = None):
        """
        Log reward decomposition data.
        
        Args:
            reward_components: Dictionary of component -> reward value
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            timestamp: Optional timestamp for the log entry
        """
        if not self.enabled:
            return
            
        self.step_count += 1
        
        # Only log at specified frequency
        if self.step_count % self.log_frequency != 0:
            return
            
        # Use current timestamp if not provided
        if timestamp is None:
            timestamp = pd.Timestamp.now()
            
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'step': self.step_count,
            'reward_components': reward_components.copy(),
            'total_reward': sum(reward_components.values()),
            'portfolio_state': self._serialize_portfolio_state(portfolio_state),
            'market_data': self._serialize_market_data(market_data),
            'actions': actions.copy()
        }
        
        # Add to history
        self.reward_history.append(log_entry)
        
        # Limit history size
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
            
        # Update statistics
        self._update_statistics(reward_components, portfolio_state)
        
        # Update real-time monitor if enabled
        if self.real_time_monitoring:
            self.monitor.update(log_entry)
            
        # Periodically save to disk
        if self.step_count % (self.log_frequency * 100) == 0:
            self.save_to_disk()
            
    def get_component_history(self, component: str) -> List[Dict[str, Any]]:
        """
        Get history for a specific reward component.
        
        Args:
            component: Name of the reward component
            
        Returns:
            List of historical entries for the component
        """
        history = []
        for entry in self.reward_history:
            if component in entry['reward_components']:
                history.append({
                    'timestamp': entry['timestamp'],
                    'step': entry['step'],
                    'value': entry['reward_components'][component]
                })
        return history
        
    def get_ticker_history(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get history for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            List of historical entries for the ticker
        """
        history = []
        for entry in self.reward_history:
            if ticker in entry['actions']:
                history.append({
                    'timestamp': entry['timestamp'],
                    'step': entry['step'],
                    'action': entry['actions'][ticker],
                    'total_reward': entry['total_reward']
                })
        return history
        
    def get_component_stats(self, component: str) -> Dict[str, float]:
        """
        Get statistics for a specific reward component.
        
        Args:
            component: Name of the reward component
            
        Returns:
            Dictionary of statistics for the component
        """
        return self.component_stats.get(component, {})
        
    def get_ticker_stats(self, ticker: str) -> Dict[str, float]:
        """
        Get statistics for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary of statistics for the ticker
        """
        return self.ticker_stats.get(ticker, {})
        
    def get_portfolio_stats(self) -> Dict[str, float]:
        """
        Get portfolio-level statistics.
        
        Returns:
            Dictionary of portfolio-level statistics
        """
        return self.portfolio_stats
        
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix of reward components.
        
        Returns:
            DataFrame with correlation matrix
        """
        if not self.reward_history:
            return pd.DataFrame()
            
        # Extract component values
        component_data = {}
        for entry in self.reward_history:
            for component, value in entry['reward_components'].items():
                if component not in component_data:
                    component_data[component] = []
                component_data[component].append(value)
                
        # Create DataFrame
        df = pd.DataFrame(component_data)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        return correlation_matrix
        
    def save_to_disk(self):
        """Save reward decomposition data to disk."""
        if not self.enabled or not self.storage_path:
            return
            
        # Save history
        history_path = os.path.join(self.storage_path, 'reward_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.reward_history, f, indent=2, default=str)
            
        # Save statistics
        stats_path = os.path.join(self.storage_path, 'reward_stats.json')
        stats = {
            'component_stats': self.component_stats,
            'ticker_stats': self.ticker_stats,
            'portfolio_stats': self.portfolio_stats
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        # Save correlation matrix
        correlation_matrix = self.get_correlation_matrix()
        if not correlation_matrix.empty:
            correlation_path = os.path.join(self.storage_path, 'correlation_matrix.csv')
            correlation_matrix.to_csv(correlation_path)
            
    def load_from_disk(self):
        """Load reward decomposition data from disk."""
        if not self.enabled or not self.storage_path:
            return
            
        # Load history
        history_path = os.path.join(self.storage_path, 'reward_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.reward_history = json.load(f)
                
        # Load statistics
        stats_path = os.path.join(self.storage_path, 'reward_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                self.component_stats = stats.get('component_stats', {})
                self.ticker_stats = stats.get('ticker_stats', {})
                self.portfolio_stats = stats.get('portfolio_stats', {})
                
    def export_to_csv(self, output_path: str):
        """
        Export reward decomposition data to CSV.
        
        Args:
            output_path: Path to save the CSV file
        """
        if not self.reward_history:
            return
            
        # Flatten data for CSV export
        rows = []
        for entry in self.reward_history:
            row = {
                'timestamp': entry['timestamp'],
                'step': entry['step'],
                'total_reward': entry['total_reward']
            }
            
            # Add reward components
            for component, value in entry['reward_components'].items():
                row[f'component_{component}'] = value
                
            # Add actions
            for ticker, action in entry['actions'].items():
                row[f'action_{ticker}'] = action
                
            rows.append(row)
            
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
    def _serialize_portfolio_state(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """
        Serialize portfolio state for logging.
        
        Args:
            portfolio_state: Portfolio state to serialize
            
        Returns:
            Serialized portfolio state
        """
        return {
            'pnl': getattr(portfolio_state, 'pnl', None),
            'drawdown': getattr(portfolio_state, 'drawdown', None),
            'positions': getattr(portfolio_state, 'positions', {}),
            'cash': getattr(portfolio_state, 'cash', None),
            'total_value': getattr(portfolio_state, 'total_value', None)
        }
        
    def _serialize_market_data(self, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """
        Serialize market data for logging.
        
        Args:
            market_data: Market data to serialize
            
        Returns:
            Serialized market data
        """
        serialized = {}
        for ticker, data in market_data.items():
            serialized[ticker] = {
                'price': getattr(data, 'price', None),
                'volume': getattr(data, 'volume', None),
                'volatility': getattr(data, 'volatility', None),
                'trend': getattr(data, 'trend', None)
            }
        return serialized
        
    def _update_statistics(self, reward_components: Dict[str, float],
                          portfolio_state: PortfolioState):
        """
        Update statistics for reward components and portfolio.
        
        Args:
            reward_components: Dictionary of component -> reward value
            portfolio_state: Current portfolio state
        """
        # Update component statistics
        for component, value in reward_components.items():
            if component not in self.component_stats:
                self.component_stats[component] = {
                    'count': 0,
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'mean': 0.0,
                    'std': 0.0
                }
                
            stats = self.component_stats[component]
            stats['count'] += 1
            stats['sum'] += value
            stats['sum_sq'] += value * value
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['mean'] = stats['sum'] / stats['count']
            
            if stats['count'] > 1:
                variance = (stats['sum_sq'] - stats['sum']**2 / stats['count']) / (stats['count'] - 1)
                stats['std'] = np.sqrt(max(0, variance))
                
        # Update portfolio statistics
        if hasattr(portfolio_state, 'pnl') and portfolio_state.pnl is not None:
            if 'pnl' not in self.portfolio_stats:
                self.portfolio_stats['pnl'] = {
                    'count': 0,
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'mean': 0.0,
                    'std': 0.0
                }
                
            stats = self.portfolio_stats['pnl']
            stats['count'] += 1
            stats['sum'] += portfolio_state.pnl
            stats['sum_sq'] += portfolio_state.pnl * portfolio_state.pnl
            stats['min'] = min(stats['min'], portfolio_state.pnl)
            stats['max'] = max(stats['max'], portfolio_state.pnl)
            stats['mean'] = stats['sum'] / stats['count']
            
            if stats['count'] > 1:
                variance = (stats['sum_sq'] - stats['sum']**2 / stats['count']) / (stats['count'] - 1)
                stats['std'] = np.sqrt(max(0, variance))
```

#### 2. RewardDecompositionMonitor
Component for real-time monitoring of reward decomposition.

```python
class RewardDecompositionMonitor:
    """
    Real-time monitor for reward decomposition.
    
    Provides real-time visualization and alerting for reward components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward decomposition monitor.
        
        Args:
            config: Configuration dictionary with monitoring settings
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.dashboard_port = config.get('dashboard_port', 8080)
        self.update_interval = config.get('update_interval', 1)  # seconds
        
        # Data structures for monitoring
        self.current_values = {}  # Current reward component values
        self.recent_history = deque(maxlen=100)  # Recent history for visualization
        self.alerts = []  # List of active alerts
        
        # Start dashboard if enabled
        if self.enabled:
            self._start_dashboard()
            
    def update(self, log_entry: Dict[str, Any]):
        """
        Update monitor with new log entry.
        
        Args:
            log_entry: New log entry to monitor
        """
        if not self.enabled:
            return
            
        # Update current values
        self.current_values = log_entry['reward_components'].copy()
        
        # Add to recent history
        self.recent_history.append({
            'timestamp': log_entry['timestamp'],
            'reward_components': log_entry['reward_components'].copy(),
            'total_reward': log_entry['total_reward']
        })
        
        # Check for alerts
        self._check_alerts(log_entry['reward_components'])
        
    def get_current_values(self) -> Dict[str, float]:
        """
        Get current reward component values.
        
        Returns:
            Dictionary of component -> current value
        """
        return self.current_values.copy()
        
    def get_recent_history(self) -> List[Dict[str, Any]]:
        """
        Get recent history for visualization.
        
        Returns:
            List of recent entries
        """
        return list(self.recent_history)
        
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get active alerts.
        
        Returns:
            List of active alerts
        """
        return self.alerts.copy()
        
    def _start_dashboard(self):
        """Start the monitoring dashboard."""
        # This would typically start a web-based dashboard
        # For now, we'll just log that it would start
        print(f"Reward decomposition monitor dashboard would start on port {self.dashboard_port}")
        
    def _check_alerts(self, reward_components: Dict[str, float]):
        """
        Check for alert conditions.
        
        Args:
            reward_components: Dictionary of component -> reward value
        """
        for component, value in reward_components.items():
            if component in self.alert_thresholds:
                thresholds = self.alert_thresholds[component]
                
                # Check upper threshold
                if 'upper' in thresholds and value > thresholds['upper']:
                    alert = {
                        'component': component,
                        'type': 'upper_threshold',
                        'value': value,
                        'threshold': thresholds['upper'],
                        'timestamp': pd.Timestamp.now(),
                        'message': f"{component} value {value} exceeds upper threshold {thresholds['upper']}"
                    }
                    self.alerts.append(alert)
                    
                # Check lower threshold
                if 'lower' in thresholds and value < thresholds['lower']:
                    alert = {
                        'component': component,
                        'type': 'lower_threshold',
                        'value': value,
                        'threshold': thresholds['lower'],
                        'timestamp': pd.Timestamp.now(),
                        'message': f"{component} value {value} below lower threshold {thresholds['lower']}"
                    }
                    self.alerts.append(alert)
                    
        # Limit number of alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
```

#### 3. RewardDecompositionAnalyzer
Component for analyzing reward decomposition data.

```python
class RewardDecompositionAnalyzer:
    """
    Analyzer for reward decomposition data.
    
    Provides tools for analyzing reward component behavior and relationships.
    """
    
    def __init__(self, logger: RewardDecompositionLogger):
        """
        Initialize the reward decomposition analyzer.
        
        Args:
            logger: RewardDecompositionLogger instance with data to analyze
        """
        self.logger = logger
        
    def analyze_component_trends(self, component: str, window: int = 100) -> Dict[str, Any]:
        """
        Analyze trends for a specific reward component.
        
        Args:
            component: Name of the reward component
            window: Window size for trend analysis
            
        Returns:
            Dictionary with trend analysis results
        """
        history = self.logger.get_component_history(component)
        if len(history) < window:
            return {'error': 'Insufficient data for trend analysis'}
            
        # Extract values
        values = [entry['value'] for entry in history[-window:]]
        
        # Calculate trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Calculate momentum
        momentum = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
        
        # Calculate volatility
        volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        
        return {
            'component': component,
            'trend_slope': slope,
            'trend_r_squared': r_value**2,
            'trend_p_value': p_value,
            'momentum': momentum,
            'volatility': volatility,
            'current_value': values[-1],
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'min_value': np.min(values),
            'max_value': np.max(values)
        }
        
    def analyze_component_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlations between reward components.
        
        Returns:
            Dictionary with correlation analysis results
        """
        correlation_matrix = self.logger.get_correlation_matrix()
        if correlation_matrix.empty:
            return {'error': 'No data available for correlation analysis'}
            
        # Find strongest correlations
        strongest_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                component1 = correlation_matrix.columns[i]
                component2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]
                
                strongest_correlations.append({
                    'component1': component1,
                    'component2': component2,
                    'correlation': correlation
                })
                
        # Sort by absolute correlation
        strongest_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strongest_correlations': strongest_correlations[:10]  # Top 10
        }
        
    def analyze_reward_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of total rewards.
        
        Returns:
            Dictionary with distribution analysis results
        """
        if not self.logger.reward_history:
            return {'error': 'No data available for distribution analysis'}
            
        # Extract total rewards
        total_rewards = [entry['total_reward'] for entry in self.logger.reward_history]
        
        # Calculate distribution statistics
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        skewness = stats.skew(total_rewards)
        kurtosis = stats.kurtosis(total_rewards)
        
        # Calculate percentiles
        percentiles = {
            'p5': np.percentile(total_rewards, 5),
            'p25': np.percentile(total_rewards, 25),
            'p50': np.percentile(total_rewards, 50),
            'p75': np.percentile(total_rewards, 75),
            'p95': np.percentile(total_rewards, 95)
        }
        
        # Test for normality
        _, normality_p = stats.normaltest(total_rewards)
        
        return {
            'mean': mean_reward,
            'std': std_reward,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'percentiles': percentiles,
            'normality_p_value': normality_p,
            'is_normal': normality_p > 0.05,
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards)
        }
        
    def analyze_component_importance(self) -> Dict[str, Any]:
        """
        Analyze the importance of each reward component.
        
        Returns:
            Dictionary with component importance analysis
        """
        if not self.logger.reward_history:
            return {'error': 'No data available for importance analysis'}
            
        # Calculate component contributions
        component_contributions = {}
        for component in self.logger.component_stats:
            abs_values = [abs(entry['reward_components'].get(component, 0)) 
                         for entry in self.logger.reward_history 
                         if component in entry['reward_components']]
            
            if abs_values:
                component_contributions[component] = {
                    'mean_abs_contribution': np.mean(abs_values),
                    'std_abs_contribution': np.std(abs_values),
                    'max_abs_contribution': np.max(abs_values),
                    'total_abs_contribution': np.sum(abs_values)
                }
                
        # Calculate importance scores
        total_abs_contribution = sum(data['total_abs_contribution'] 
                                   for data in component_contributions.values())
        
        if total_abs_contribution > 0:
            for component, data in component_contributions.items():
                data['importance_score'] = data['total_abs_contribution'] / total_abs_contribution
        else:
            for component, data in component_contributions.items():
                data['importance_score'] = 0
                
        # Sort by importance score
        sorted_components = sorted(component_contributions.items(), 
                                key=lambda x: x[1]['importance_score'], 
                                reverse=True)
        
        return {
            'component_importance': dict(sorted_components),
            'total_abs_contribution': total_abs_contribution
        }
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Dictionary with complete analysis report
        """
        report = {
            'timestamp': pd.Timestamp.now(),
            'total_entries': len(self.logger.reward_history),
            'component_trends': {},
            'component_correlations': self.analyze_component_correlations(),
            'reward_distribution': self.analyze_reward_distribution(),
            'component_importance': self.analyze_component_importance()
        }
        
        # Analyze trends for each component
        for component in self.logger.component_stats:
            report['component_trends'][component] = self.analyze_component_trends(component)
            
        return report
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Reward          │    │ Portfolio       │    │ Market Data     │
│ Components      │    │ State           │    │ (Input)         │
│ (Input)         │    │ (Input)         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Reward          │    │ Log Entry       │    │ Actions         │
│ Decomposition   │    │ Creation        │    │ (Input)         │
│ Logger          │◀───│                 │◀───│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Statistics      │
│ Update          │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Real-time       │
│ Monitor         │
│ Update          │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Periodic        │
│ Save to Disk    │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Analysis &      │
│ Reporting       │
│ (Output)        │
└─────────────────┘
```

## Configuration Structure

### Reward Decomposition Logging Configuration

```yaml
# Reward decomposition logging configuration
reward_decomposition_logging:
  enabled: true
  weight: 0.0  # Not a reward component, just for configuration
  
  # Logging parameters
  log_frequency: 1  # Log every N steps
  max_history: 10000  # Maximum entries to keep in memory
  storage_path: data/reward_decomposition  # Path for storing log data
  compression: true  # Enable compression for stored data
  
  # Real-time monitoring
  real_time_monitoring: true
  monitoring:
    enabled: true
    dashboard_port: 8080  # Port for monitoring dashboard
    update_interval: 1  # Update interval in seconds
    
    # Alert thresholds for each component
    alert_thresholds:
      pnl:
        upper: 0.1  # Alert if PnL component exceeds this value
        lower: -0.1  # Alert if PnL component falls below this value
      dsr:
        upper: 0.05
        lower: -0.05
      sharpe:
        upper: 0.1
        lower: -0.1
      potential_based_reward_shaping:
        upper: 0.2
        lower: -0.2
        
  # Analysis parameters
  analysis:
    trend_window: 100  # Window size for trend analysis
    correlation_threshold: 0.5  # Threshold for significant correlations
    normality_threshold: 0.05  # P-value threshold for normality test
    
  # Export settings
  export:
    csv_enabled: true
    csv_path: data/reward_decomposition/export.csv
    json_enabled: true
    json_path: data/reward_decomposition/export.json
    plot_enabled: true
    plot_path: data/reward_decomposition/plots/
```

## Implementation Details

### Logging Algorithm

```python
def log_reward(self, reward_components: Dict[str, float],
              portfolio_state: PortfolioState,
              market_data: Dict[str, MarketData],
              actions: Dict[str, float],
              timestamp: Optional[pd.Timestamp] = None):
    """
    Log reward decomposition data.
    
    Args:
        reward_components: Dictionary of component -> reward value
        portfolio_state: Current portfolio state
        market_data: Current market data for all tickers
        actions: Actions taken for each ticker
        timestamp: Optional timestamp for the log entry
    """
    if not self.enabled:
        return
        
    self.step_count += 1
    
    # Only log at specified frequency
    if self.step_count % self.log_frequency != 0:
        return
        
    # Use current timestamp if not provided
    if timestamp is None:
        timestamp = pd.Timestamp.now()
        
    # Create log entry
    log_entry = {
        'timestamp': timestamp,
        'step': self.step_count,
        'reward_components': reward_components.copy(),
        'total_reward': sum(reward_components.values()),
        'portfolio_state': self._serialize_portfolio_state(portfolio_state),
        'market_data': self._serialize_market_data(market_data),
        'actions': actions.copy()
    }
    
    # Add to history
    self.reward_history.append(log_entry)
    
    # Limit history size
    if len(self.reward_history) > self.max_history:
        self.reward_history.pop(0)
        
    # Update statistics
    self._update_statistics(reward_components, portfolio_state)
    
    # Update real-time monitor if enabled
    if self.real_time_monitoring:
        self.monitor.update(log_entry)
        
    # Periodically save to disk
    if self.step_count % (self.log_frequency * 100) == 0:
        self.save_to_disk()
```

### Statistics Update Algorithm

```python
def _update_statistics(self, reward_components: Dict[str, float],
                      portfolio_state: PortfolioState):
    """
    Update statistics for reward components and portfolio.
    
    Args:
        reward_components: Dictionary of component -> reward value
        portfolio_state: Current portfolio state
    """
    # Update component statistics
    for component, value in reward_components.items():
        if component not in self.component_stats:
            self.component_stats[component] = {
                'count': 0,
                'sum': 0.0,
                'sum_sq': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'mean': 0.0,
                'std': 0.0
            }
            
        stats = self.component_stats[component]
        stats['count'] += 1
        stats['sum'] += value
        stats['sum_sq'] += value * value
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
        stats['mean'] = stats['sum'] / stats['count']
        
        if stats['count'] > 1:
            variance = (stats['sum_sq'] - stats['sum']**2 / stats['count']) / (stats['count'] - 1)
            stats['std'] = np.sqrt(max(0, variance))
            
    # Update portfolio statistics
    if hasattr(portfolio_state, 'pnl') and portfolio_state.pnl is not None:
        if 'pnl' not in self.portfolio_stats:
            self.portfolio_stats['pnl'] = {
                'count': 0,
                'sum': 0.0,
                'sum_sq': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'mean': 0.0,
                'std': 0.0
            }
            
        stats = self.portfolio_stats['pnl']
        stats['count'] += 1
        stats['sum'] += portfolio_state.pnl
        stats['sum_sq'] += portfolio_state.pnl * portfolio_state.pnl
        stats['min'] = min(stats['min'], portfolio_state.pnl)
        stats['max'] = max(stats['max'], portfolio_state.pnl)
        stats['mean'] = stats['sum'] / stats['count']
        
        if stats['count'] > 1:
            variance = (stats['sum_sq'] - stats['sum']**2 / stats['count']) / (stats['count'] - 1)
            stats['std'] = np.sqrt(max(0, variance))
```

### Trend Analysis Algorithm

```python
def analyze_component_trends(self, component: str, window: int = 100) -> Dict[str, Any]:
    """
    Analyze trends for a specific reward component.
    
    Args:
        component: Name of the reward component
        window: Window size for trend analysis
        
    Returns:
        Dictionary with trend analysis results
    """
    history = self.logger.get_component_history(component)
    if len(history) < window:
        return {'error': 'Insufficient data for trend analysis'}
        
    # Extract values
    values = [entry['value'] for entry in history[-window:]]
    
    # Calculate trend
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    
    # Calculate momentum
    momentum = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
    
    # Calculate volatility
    volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
    
    return {
        'component': component,
        'trend_slope': slope,
        'trend_r_squared': r_value**2,
        'trend_p_value': p_value,
        'momentum': momentum,
        'volatility': volatility,
        'current_value': values[-1],
        'mean_value': np.mean(values),
        'std_value': np.std(values),
        'min_value': np.min(values),
        'max_value': np.max(values)
    }
```

## Best Practices

### Logging Strategy
1. **Frequency Management**: Use appropriate logging frequency to balance detail and performance
2. **Data Compression**: Compress stored data to save disk space
3. **Memory Management**: Limit in-memory history to prevent memory issues
4. **Asynchronous Logging**: Consider asynchronous logging to minimize performance impact
5. **Selective Logging**: Log only relevant data to reduce storage requirements

### Monitoring and Alerting
1. **Meaningful Thresholds**: Set alert thresholds based on historical data analysis
2. **Alert Aggregation**: Aggregate similar alerts to reduce noise
3. **Escalation Rules**: Implement escalation rules for critical alerts
4. **Alert Suppression**: Suppress alerts during expected events (e.g., market open/close)
5. **Notification Channels**: Support multiple notification channels (email, Slack, etc.)

### Analysis and Reporting
1. **Statistical Rigor**: Use appropriate statistical methods for analysis
2. **Visualization**: Create clear, informative visualizations
3. **Interpretability**: Make analysis results easy to understand
4. **Actionable Insights**: Focus on insights that can drive action
5. **Historical Context**: Always provide historical context for analysis

### Performance Optimization
1. **Efficient Data Structures**: Use efficient data structures for storing and accessing data
2. **Caching**: Cache frequently accessed data to improve performance
3. **Parallel Processing**: Use parallel processing for computationally intensive analysis
4. **Incremental Updates**: Update statistics incrementally rather than recalculating
5. **Lazy Evaluation**: Only perform analysis when requested

### Data Management
1. **Backup Strategy**: Implement regular backups of log data
2. **Retention Policy**: Define clear retention policies for log data
3. **Data Validation**: Validate data integrity before analysis
4. **Metadata Management**: Maintain comprehensive metadata for logged data
5. **Access Control**: Implement appropriate access controls for sensitive data