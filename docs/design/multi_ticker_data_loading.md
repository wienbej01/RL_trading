# Multi-Ticker Data Loading Architecture Design

## Overview

This document outlines the design for extending the existing single-ticker data loading system to support multiple tickers simultaneously. The multi-ticker data loading architecture will provide efficient, scalable, and flexible data access for the RL trading system.

## Requirements

### Functional Requirements
1. **Multi-Ticker Support**: Load and manage data for multiple tickers simultaneously
2. **Data Alignment**: Align data across tickers by timestamp
3. **Flexible Data Sources**: Support both Polygon and Databento data sources
4. **Caching**: Implement intelligent caching for performance optimization
5. **Data Validation**: Ensure data quality and consistency across tickers
6. **Metadata Management**: Track and manage ticker-specific metadata

### Non-Functional Requirements
1. **Performance**: Efficient memory usage and fast data access
2. **Scalability**: Support for 10+ tickers simultaneously
3. **Reliability**: Robust error handling and data recovery
4. **Maintainability**: Clean, well-documented code with clear interfaces
5. **Backward Compatibility**: Maintain compatibility with existing single-ticker workflows

## Architecture Overview

### Core Components

#### 1. MultiTickerDataLoader
The main class responsible for coordinating data loading across multiple tickers.

```python
class MultiTickerDataLoader:
    """
    Multi-ticker data loader for RL trading system.
    
    Coordinates loading, alignment, and management of data for multiple tickers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-ticker data loader.
        
        Args:
            config: Configuration dictionary with ticker settings
        """
        self.config = config
        self.ticker_configs = config.get('tickers', {})
        self.data_loaders = {}  # Individual ticker data loaders
        self.aligned_data = None
        self.metadata = {}
        self.cache_manager = CacheManager(config.get('cache', {}))
        
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load and align data for all configured tickers.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            Aligned multi-ticker DataFrame
        """
        pass
        
    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Get data for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            DataFrame for the specified ticker
        """
        pass
```

#### 2. TickerDataLoader
Individual ticker data loader that extends the existing UnifiedDataLoader.

```python
class TickerDataLoader(UnifiedDataLoader):
    """
    Individual ticker data loader with multi-ticker specific enhancements.
    """
    
    def __init__(self, ticker: str, config: Dict[str, Any]):
        """
        Initialize ticker data loader.
        
        Args:
            ticker: Ticker symbol
            config: Ticker-specific configuration
        """
        super().__init__(config)
        self.ticker = ticker
        self.metadata = self._load_ticker_metadata()
        
    def _load_ticker_metadata(self) -> Dict[str, Any]:
        """
        Load ticker-specific metadata.
        
        Returns:
            Ticker metadata dictionary
        """
        pass
```

#### 3. DataAligner
Component responsible for aligning data across tickers.

```python
class DataAligner:
    """
    Data alignment component for multi-ticker data.
    """
    
    def __init__(self, alignment_method: str = 'outer_join'):
        """
        Initialize data aligner.
        
        Args:
            alignment_method: Method for aligning data ('outer_join', 'inner_join', 'forward_fill')
        """
        self.alignment_method = alignment_method
        
    def align_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align data across multiple tickers.
        
        Args:
            data_dict: Dictionary of ticker -> DataFrame
            
        Returns:
            Aligned multi-ticker DataFrame
        """
        pass
```

#### 4. CacheManager
Intelligent caching system for performance optimization.

```python
class CacheManager:
    """
    Cache manager for multi-ticker data loading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.cache_dir = Path(config.get('cache_dir', './cache'))
        self.max_size_gb = config.get('max_size_gb', 10)
        self.ttl_days = config.get('ttl_days', 7)
        
    def get_cached_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get cached data if available and valid.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Cached DataFrame or None
        """
        pass
        
    def cache_data(self, ticker: str, data: pd.DataFrame, start_date: str, end_date: str):
        """
        Cache data for future use.
        
        Args:
            ticker: Ticker symbol
            data: DataFrame to cache
            start_date: Start date
            end_date: End date
        """
        pass
```

#### 5. MetadataManager
Component for managing ticker-specific metadata.

```python
class MetadataManager:
    """
    Metadata manager for multi-ticker data.
    """
    
    def __init__(self, metadata_dir: Path):
        """
        Initialize metadata manager.
        
        Args:
            metadata_dir: Directory for metadata storage
        """
        self.metadata_dir = metadata_dir
        self.metadata_cache = {}
        
    def get_ticker_metadata(self, ticker: str) -> Dict[str, Any]:
        """
        Get metadata for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Ticker metadata dictionary
        """
        pass
        
    def update_ticker_metadata(self, ticker: str, metadata: Dict[str, Any]):
        """
        Update metadata for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            metadata: Updated metadata
        """
        pass
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MultiTicker     │    │ TickerData      │    │ Data            │
│ DataLoader      │───▶│ Loader          │───▶│ Source          │
│ (Coordinator)   │    │ (Individual)    │    │ (Polygon/       │
└─────────────────┘    └─────────────────┘    │ Databento)      │
         │                                           └─────────────────┘
         │                                                  │
         │                                                  ▼
         │                                            ┌─────────────────┐
         │                                            │ Raw Data        │
         │                                            │ (Per ticker)    │
         │                                            └─────────────────┘
         │                                                  │
         │                                                  ▼
         │                                            ┌─────────────────┐
         └────────────────────────────────────────────▶│ DataAligner     │
                                                      │ (Alignment)     │
                                                      └─────────────────┘
                                                             │
                                                             ▼
                                                      ┌─────────────────┐
                                                      │ Aligned Data    │
                                                      │ (Multi-ticker)  │
                                                      └─────────────────┘
                                                             │
                                                             ▼
                                                      ┌─────────────────┐
                                                      │ CacheManager    │
                                                      │ (Caching)       │
                                                      └─────────────────┘
```

## Configuration Structure

### Multi-Ticker Configuration

```yaml
# Multi-ticker configuration
multi_ticker:
  enabled: true
  tickers:
    AAPL:
      data_source: polygon
      priority: 1
      weight: 0.3
      metadata:
        sector: Technology
        market_cap: large
    MSFT:
      data_source: polygon
      priority: 2
      weight: 0.3
      metadata:
        sector: Technology
        market_cap: large
    GOOGL:
      data_source: databento
      priority: 3
      weight: 0.4
      metadata:
        sector: Technology
        market_cap: large
  
  # Data alignment settings
  alignment:
    method: outer_join  # outer_join, inner_join, forward_fill
    timezone: US/Eastern
    resampling: 1min  # Resample to this frequency
    
  # Cache settings
  cache:
    enabled: true
    cache_dir: ./cache/multi_ticker
    max_size_gb: 20
    ttl_days: 7
    compression: true
    
  # Data validation
  validation:
    enabled: true
    check_gaps: true
    check_outliers: true
    max_gap_minutes: 5
    outlier_std_threshold: 3.0
```

### Ticker-Specific Configuration

```yaml
# Ticker-specific configuration (can override global settings)
ticker_config:
  # Default settings for all tickers
  defaults:
    data_source: polygon
    timezone: US/Eastern
    session_start: 09:30
    session_end: 16:00
    resampling: 1min
    
  # Ticker-specific overrides
  overrides:
    SPY:
      data_source: databento
      resampling: 30s
    TSLA:
      session_start: 09:30
      session_end: 16:00
      outlier_std_threshold: 4.0
```

## Implementation Details

### Data Alignment Strategy

#### 1. Timestamp Alignment
- Convert all timestamps to a common timezone
- Resample all data to a consistent frequency
- Handle missing data according to alignment method

#### 2. Alignment Methods
- **Outer Join**: Keep all timestamps from all tickers, fill missing values
- **Inner Join**: Keep only timestamps present in all tickers
- **Forward Fill**: Forward fill missing values for each ticker

#### 3. Column Naming Convention
- Prefix columns with ticker symbol to avoid conflicts
- Example: `AAPL_close`, `MSFT_close`, `GOOGL_close`

### Caching Strategy

#### 1. Cache Structure
```
cache/
├── multi_ticker/
│   ├── AAPL/
│   │   ├── 2023-01-01_2023-01-31.parquet
│   │   └── metadata.json
│   ├── MSFT/
│   │   ├── 2023-01-01_2023-01-31.parquet
│   │   └── metadata.json
│   └── cache_index.json
```

#### 2. Cache Key Generation
- Generate cache keys based on ticker, date range, and configuration hash
- Include data source and processing parameters in key

#### 3. Cache Invalidation
- Time-based invalidation (TTL)
- Configuration-based invalidation
- Manual invalidation options

### Error Handling and Recovery

#### 1. Data Loading Errors
- Retry failed loads with exponential backoff
- Log errors with detailed context
- Skip problematic tickers when possible

#### 2. Data Quality Issues
- Detect and handle gaps in data
- Identify and flag outliers
- Provide data quality metrics

#### 3. Resource Management
- Monitor memory usage
- Implement graceful degradation
- Provide cleanup mechanisms

## Performance Considerations

### Memory Optimization
1. **Lazy Loading**: Load data on demand when possible
2. **Data Types**: Use appropriate data types to minimize memory usage
3. **Chunking**: Process data in chunks for large date ranges
4. **Compression**: Use compressed file formats for cached data

### Parallel Processing
1. **Concurrent Loading**: Load data for multiple tickers in parallel
2. **Async Operations**: Use async I/O for network operations
3. **Thread Pool**: Manage concurrent operations efficiently

### Data Access Patterns
1. **Pre-fetching**: Anticipate and pre-load commonly accessed data
2. **Hot Data**: Keep frequently accessed data in memory
3. **Cold Data**: Store infrequently accessed data on disk

## Testing Strategy

### Unit Tests
1. **Individual Components**: Test each component in isolation
2. **Edge Cases**: Test boundary conditions and error scenarios
3. **Configuration**: Test various configuration combinations

### Integration Tests
1. **End-to-End**: Test complete data loading pipeline
2. **Multi-Ticker**: Test with multiple tickers simultaneously
3. **Data Sources**: Test with both Polygon and Databento

### Performance Tests
1. **Scalability**: Test with increasing numbers of tickers
2. **Memory Usage**: Monitor memory consumption
3. **Load Time**: Measure data loading performance

## Migration Path

### Phase 1: Backward Compatibility
1. Maintain existing single-ticker interface
2. Add multi-ticker functionality as an option
3. Ensure existing tests continue to pass

### Phase 2: Coexistence
1. Support both single and multi-ticker workflows
2. Provide migration utilities
3. Update documentation and examples

### Phase 3: Multi-Ticker First
1. Make multi-ticker the primary interface
2. Maintain single-ticker compatibility layer
3. Deprecate single-ticker specific features

## Future Enhancements

### Real-time Data Support
1. Streaming data integration
2. Real-time alignment and normalization
3. Event-driven architecture

### Advanced Features
1. Dynamic ticker universe
2. Smart data sampling
3. Cross-ticker feature calculation

### Performance Optimizations
1. GPU acceleration
2. Distributed processing
3. Advanced caching strategies

## Conclusion

The multi-ticker data loading architecture provides a robust, scalable foundation for the enhanced RL trading system. By building on the existing single-ticker infrastructure and adding multi-ticker specific components, we can efficiently support multiple tickers while maintaining backward compatibility and performance.

The design emphasizes modularity, with clear separation of concerns between data loading, alignment, caching, and metadata management. This approach allows for independent development and testing of each component while ensuring they work together seamlessly.

The architecture is designed to scale to support 10+ tickers simultaneously, with intelligent caching and performance optimizations to ensure efficient operation even with large datasets.