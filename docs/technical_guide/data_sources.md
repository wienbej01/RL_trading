# Data Sources in RL Trading System

## Overview

This document provides a comprehensive technical guide to the data sources used in the RL trading system, including both single-ticker and multi-ticker implementations. It covers data acquisition, storage, processing, and integration with the reward mechanisms.

## Primary Data Sources

### Polygon.io

#### Overview
Polygon.io is the primary data source for intraday OHLCV (Open, High, Low, Close, Volume) data used in the trading system.

#### Data Specifications
- **Frequency**: 1-minute bars for intraday trading
- **Assets**: US equities (stocks, ETFs)
- **Time Range**: Historical data from 2020 onwards
- **Market Hours**: Regular trading hours (9:30 AM - 4:00 PM ET)

#### Data Structure
```
data/polygon/historical/symbol={TICKER}/year={YEAR}/month={MONTH}/day={DAY}/data.parquet
```

#### Data Schema
Each parquet file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | int64 | Unix timestamp in milliseconds |
| open | float64 | Opening price for the minute |
| high | float64 | Highest price during the minute |
| low | float64 | Lowest price during the minute |
| close | float64 | Closing price for the minute |
| volume | float64 | Trading volume during the minute |
| vwap | float64 | Volume-weighted average price |
| transactions | int64 | Number of transactions during the minute |

#### Data Access
```python
from src.data.data_loader import UnifiedDataLoader

# Initialize data loader
data_loader = UnifiedDataLoader(config_path="configs/settings.yaml")

# Load single ticker data
data = data_loader.load_ohlcv(
    symbol="AAPL",
    start=pd.Timestamp("2020-01-01"),
    end=pd.Timestamp("2024-12-31"),
    frequency="1min"
)
```

### Databento (Alternative)

#### Overview
Databento is an alternative high-frequency data source that can be used as a backup or supplement to Polygon.io.

#### Data Specifications
- **Frequency**: Tick-level or 1-minute bars
- **Assets**: US equities, futures, options
- **Time Range**: Historical data from 2019 onwards
- **Market Hours**: Extended hours including pre-market and post-market

#### Data Structure
```
data/databento/symbol={TICKER}/year={YEAR}/month={MONTH}/day={DAY}/data.parquet
```

## Secondary Data Sources

### VIX Data

#### Overview
VIX (Volatility Index) data is used for market regime detection and volatility-based reward adjustments.

#### Data Specifications
- **Frequency**: Daily
- **Source**: CBOE via Polygon or direct download
- **Time Range**: Historical data from 2020 onwards

#### Data Structure
```
data/external/vix.parquet
```

#### Data Schema
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime64 | Date |
| vix_close | float64 | VIX closing value |
| vix_ret | float64 | VIX daily return |
| vix_ma20 | float64 | 20-day moving average of VIX |
| vix_z20 | float64 | Z-score of VIX relative to 20-day MA |

### Sector Data

#### Overview
Sector classification data is used for portfolio diversification calculations and sector-specific reward adjustments.

#### Data Specifications
- **Frequency**: Static (updated quarterly)
- **Source**: GICS classification system
- **Coverage**: All US equities

#### Data Structure
```
data/metadata/sector_classifications.parquet
```

#### Data Schema
| Column | Type | Description |
|--------|------|-------------|
| symbol | str | Ticker symbol |
| sector | str | GICS sector name |
| industry | str | GICS industry group |
| industry_group | str | GICS industry group |

## Multi-Ticker Data Integration

### Data Alignment

#### Timestamp Alignment
For multi-ticker trading, data from different tickers must be aligned on a common timestamp grid:

```python
def align_multiticker_data(data_dict, frequency="1min"):
    """
    Align data from multiple tickers on a common timestamp grid.
    
    Args:
        data_dict: Dictionary of {ticker: DataFrame}
        frequency: Target frequency for alignment
        
    Returns:
        Aligned DataFrame with multi-index (timestamp, ticker)
    """
    # Create common timestamp grid
    all_timestamps = set()
    for ticker, df in data_dict.items():
        all_timestamps.update(df.index)
    
    common_timestamps = sorted(all_timestamps)
    
    # Reindex each ticker's data
    aligned_data = {}
    for ticker, df in data_dict.items():
        aligned_df = df.reindex(common_timestamps)
        aligned_df['ticker'] = ticker
        aligned_data[ticker] = aligned_df
    
    # Combine into single DataFrame
    combined = pd.concat(aligned_data.values(), axis=0)
    combined = combined.set_index('ticker', append=True)
    
    return combined
```

### Missing Data Handling

#### Forward Fill Strategy
```python
def handle_missing_data(data, method='ffill', limit=5):
    """
    Handle missing data in multi-ticker datasets.
    
    Args:
        data: Multi-ticker DataFrame
        method: Filling method ('ffill', 'bfill', 'interpolate')
        limit: Maximum number of consecutive periods to fill
        
    Returns:
        Cleaned DataFrame
    """
    # Group by ticker and apply filling
    if method == 'ffill':
        return data.groupby(level='ticker').fillna(method='ffill', limit=limit)
    elif method == 'bfill':
        return data.groupby(level='ticker').fillna(method='bfill', limit=limit)
    elif method == 'interpolate':
        return data.groupby(level='ticker').apply(
            lambda x: x.interpolate(method='linear', limit=limit)
        )
    else:
        raise ValueError(f"Unknown filling method: {method}")
```

### Correlation Matrix Calculation

#### Rolling Correlations
```python
def calculate_rolling_correlations(data, window=20):
    """
    Calculate rolling correlation matrix for multiple tickers.
    
    Args:
        data: Multi-ticker DataFrame with price data
        window: Rolling window size in days
        
    Returns:
        Dictionary of {timestamp: correlation_matrix}
    """
    # Extract close prices for each ticker
    close_prices = data['close'].unstack(level='ticker')
    
    # Calculate rolling correlations
    rolling_corrs = {}
    for date in close_prices.index[window:]:
        window_data = close_prices.loc[date-window+1:date]
        corr_matrix = window_data.pct_change().corr()
        rolling_corrs[date] = corr_matrix
    
    return rolling_corrs
```

## Data Processing Pipeline

### Single-Ticker Pipeline

```
Raw Data → Quality Checks → Resampling → Feature Engineering → Normalization → Cache
```

#### Quality Checks
```python
def perform_quality_checks(data):
    """
    Perform data quality checks on single-ticker data.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Cleaned DataFrame and quality report
    """
    quality_report = {}
    
    # Check for missing values
    missing_values = data.isnull().sum()
    quality_report['missing_values'] = missing_values.to_dict()
    
    # Check for outliers
    z_scores = (data - data.mean()) / data.std()
    outliers = (np.abs(z_scores) > 3).sum()
    quality_report['outliers'] = outliers.to_dict()
    
    # Check for price anomalies
    price_anomalies = (
        (data['high'] < data['low']) |
        (data['high'] < data['open']) |
        (data['high'] < data['close']) |
        (data['low'] > data['open']) |
        (data['low'] > data['close'])
    )
    quality_report['price_anomalies'] = price_anomalies.sum()
    
    # Remove anomalies
    clean_data = data[~price_anomalies]
    
    return clean_data, quality_report
```

### Multi-Ticker Pipeline

```
Raw Data → Alignment → Quality Checks → Missing Data Handling → 
Feature Engineering → Correlation Calculation → Normalization → Cache
```

#### Multi-Ticker Quality Checks
```python
def perform_multiticker_quality_checks(data):
    """
    Perform data quality checks on multi-ticker data.
    
    Args:
        data: Multi-ticker DataFrame
        
    Returns:
        Cleaned DataFrame and quality report
    """
    quality_report = {}
    
    # Check for missing data by ticker
    missing_by_ticker = data.isnull().groupby(level='ticker').sum()
    quality_report['missing_by_ticker'] = missing_by_ticker.to_dict()
    
    # Check for cross-ticker data availability
    timestamp_counts = data.groupby(level='timestamp').size()
    complete_timestamps = timestamp_counts[timestamp_counts == data.index.get_level_values('ticker').nunique()]
    quality_report['data_completeness'] = len(complete_timestamps) / len(timestamp_counts)
    
    # Check for synchronized trading hours
    trading_hours_sync = check_trading_hours_synchronization(data)
    quality_report['trading_hours_sync'] = trading_hours_sync
    
    return data, quality_report
```

## Data Caching

### Cache Structure

```
data/cache/
├── single_ticker/
│   ├── {TICKER}_{START}_{END}_{FREQUENCY}.parquet
│   └── ...
└── multi_ticker/
    ├── {TICKER_LIST}_{START}_{END}_{FREQUENCY}.parquet
    └── ...
```

### Cache Management

```python
class DataCache:
    """
    Manages caching of processed data for both single and multi-ticker scenarios.
    """
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, tickers, start_date, end_date, frequency):
        """
        Generate cache file path based on parameters.
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        ticker_str = "_".join(sorted(tickers))
        date_str = f"{start_date}_{end_date}"
        filename = f"{ticker_str}_{date_str}_{frequency}.parquet"
        
        if len(tickers) == 1:
            return self.cache_dir / "single_ticker" / filename
        else:
            return self.cache_dir / "multi_ticker" / filename
    
    def load_from_cache(self, tickers, start_date, end_date, frequency):
        """
        Load data from cache if available.
        """
        cache_path = self.get_cache_path(tickers, start_date, end_date, frequency)
        
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        else:
            return None
    
    def save_to_cache(self, data, tickers, start_date, end_date, frequency):
        """
        Save processed data to cache.
        """
        cache_path = self.get_cache_path(tickers, start_date, end_date, frequency)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_path)
```

## Data Integration with Reward Mechanisms

### Single-Ticker Integration

```python
class SingleTickerRewardCalculator:
    """
    Calculates rewards for single-ticker trading.
    """
    
    def __init__(self, config):
        self.config = config
        self.reward_type = config.get('reward', {}).get('type', 'hybrid2')
    
    def calculate_reward(self, data, positions, portfolio_value):
        """
        Calculate reward based on data and positions.
        """
        if self.reward_type == 'hybrid2':
            return self._calculate_hybrid2_reward(data, positions, portfolio_value)
        elif self.reward_type == 'sharpe':
            return self._calculate_sharpe_reward(data, positions, portfolio_value)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _calculate_hybrid2_reward(self, data, positions, portfolio_value):
        """
        Calculate Hybrid2 reward with multiple components.
        """
        # Extract necessary data
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # Calculate base PnL
        pnl = (portfolio_value - portfolio_value.shift(1)) / portfolio_value.shift(1)
        
        # Calculate risk penalty
        drawdown = (portfolio_value / portfolio_value.cummax() - 1)
        risk_penalty = self.config['reward']['hybrid2']['drawdown_penalty'] * drawdown
        
        # Calculate regime adjustment
        regime_weight = self._calculate_regime_weight(data)
        
        # Combine components
        reward = pnl * regime_weight - risk_penalty
        
        return reward
```

### Multi-Ticker Integration

```python
class MultiTickerRewardCalculator:
    """
    Calculates rewards for multi-ticker portfolio trading.
    """
    
    def __init__(self, config):
        self.config = config
        self.reward_type = config.get('reward', {}).get('type', 'multiticker_hybrid2')
    
    def calculate_reward(self, data, positions, portfolio_value):
        """
        Calculate portfolio-level reward.
        """
        if self.reward_type == 'multiticker_hybrid2':
            return self._calculate_multiticker_hybrid2_reward(data, positions, portfolio_value)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _calculate_multiticker_hybrid2_reward(self, data, positions, portfolio_value):
        """
        Calculate multi-ticker Hybrid2 reward.
        """
        # Calculate portfolio returns
        portfolio_returns = portfolio_value.pct_change()
        
        # Calculate portfolio risk
        portfolio_risk = self._calculate_portfolio_risk(data, positions)
        
        # Calculate diversification bonus
        diversification_bonus = self._calculate_diversification_bonus(positions, data)
        
        # Calculate correlation penalty
        correlation_penalty = self._calculate_correlation_penalty(positions, data)
        
        # Combine components
        reward = (
            portfolio_returns - 
            self.config['reward']['multiticker_hybrid2']['risk_adjustment_weight'] * portfolio_risk +
            self.config['reward']['multiticker_hybrid2']['diversification_bonus_weight'] * diversification_bonus -
            self.config['reward']['multiticker_hybrid2']['correlation_penalty_weight'] * correlation_penalty
        )
        
        return reward
```

## Data Quality Monitoring

### Monitoring Metrics

```python
class DataQualityMonitor:
    """
    Monitors data quality and generates alerts.
    """
    
    def __init__(self, config):
        self.config = config
        self.alert_thresholds = config.get('data_quality', {}).get('alert_thresholds', {})
    
    def check_data_quality(self, data):
        """
        Check data quality and generate alerts.
        """
        alerts = []
        
        # Check missing data
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_pct > self.alert_thresholds.get('missing_data', 0.05):
            alerts.append(f"High missing data percentage: {missing_pct:.2%}")
        
        # Check stale data
        if hasattr(data.index, 'max'):
            latest_timestamp = data.index.max()
            time_since_update = pd.Timestamp.now() - latest_timestamp
            if time_since_update > pd.Timedelta(days=1):
                alerts.append(f"Stale data: last update {time_since_update}")
        
        # Check data anomalies
        anomaly_score = self._calculate_anomaly_score(data)
        if anomaly_score > self.alert_thresholds.get('anomaly_score', 3.0):
            alerts.append(f"High anomaly score: {anomaly_score:.2f}")
        
        return alerts
```

## Outstanding Issues and Solutions

### Known Issues

1. **Feature Selection Bug**
   - **Issue**: TypeError in feature selection when selected_features is None
   - **Location**: `src/features/pipeline.py` line 1080
   - **Solution**: Add null check before feature selection
   - **Status**: Requires immediate fix

2. **Data Alignment for Multi-Ticker**
   - **Issue**: Timestamp misalignment between different tickers
   - **Impact**: Incorrect correlation calculations
   - **Solution**: Implement robust timestamp alignment
   - **Status**: Partially implemented

3. **Memory Usage with Large Datasets**
   - **Issue**: High memory consumption with multiple tickers
   - **Impact**: Limits scalability
   - **Solution**: Implement data chunking and lazy loading
   - **Status**: In progress

### Data Quality Issues

1. **Missing Trading Days**
   - **Issue**: Some tickers have missing trading days
   - **Impact**: Incomplete training data
   - **Solution**: Implement calendar-based data filling
   - **Status**: Not implemented

2. **Corporate Actions**
   - **Issue**: No adjustment for splits, dividends, or other corporate actions
   - **Impact**: Incorrect price calculations
   - **Solution**: Implement corporate action adjustments
   - **Status**: Not implemented

3. **Data Source Reliability**
   - **Issue**: Occasional data quality issues from Polygon
   - **Impact**: Training instability
   - **Solution**: Implement data validation and fallback mechanisms
   - **Status**: Partially implemented

## Future Enhancements

### Data Sources

1. **Alternative Data Integration**
   - News sentiment data
   - Social media sentiment
   - Satellite imagery
   - Supply chain data

2. **Real-Time Data Feeds**
   - WebSocket connections for real-time data
   - Low-latency data processing
   - Real-time feature calculation

3. **Options Data**
   - Implied volatility surfaces
   - Options flow analysis
   - Put/call ratios

### Processing Enhancements

1. **Distributed Processing**
   - Spark or Dask for large-scale data processing
   - Parallel feature calculation
   - Distributed training data preparation

2. **Stream Processing**
   - Kafka or similar for real-time data streams
   - Stream-based feature engineering
   - Real-time reward calculation

3. **Data Versioning**
   - DVC or similar for data version control
   - Reproducible data pipelines
   - Data lineage tracking

## Best Practices

### Data Management

1. **Data Validation**
   - Implement comprehensive data validation checks
   - Log all data quality issues
   - Set up alerts for critical data issues

2. **Data Backup**
   - Regular backups of raw and processed data
   - Version control for data processing pipelines
   - Disaster recovery procedures

3. **Data Documentation**
   - Maintain data dictionaries
   - Document data sources and processing steps
   - Keep track of data schema changes

### Performance Optimization

1. **Memory Management**
   - Use appropriate data types
   - Implement data chunking for large datasets
   - Clear unused data from memory

2. **I/O Optimization**
   - Use efficient file formats (Parquet)
   - Implement data compression
   - Optimize read/write operations

3. **Caching Strategy**
   - Cache frequently accessed data
   - Implement cache invalidation
   - Monitor cache hit rates

## Conclusion

The data sources and processing pipeline form the foundation of the RL trading system. Proper data management, quality control, and integration with reward mechanisms are essential for successful trading strategies. The system supports both single-ticker and multi-ticker trading with flexible configuration options.

Future enhancements will focus on expanding data sources, improving processing efficiency, and implementing more sophisticated data quality monitoring. Addressing the outstanding issues, particularly the feature selection bug and data alignment problems, should be prioritized to ensure system stability and performance.