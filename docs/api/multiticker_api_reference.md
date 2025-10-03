
# Multi-Ticker RL Trading System API Reference

## Overview

This document provides a comprehensive API reference for the Multi-Ticker RL Trading System. It covers all major classes, methods, and functions available in the system, with detailed parameter descriptions and usage examples.

## Table of Contents

1. [Data Loading](#data-loading)
2. [Feature Engineering](#feature-engineering)
3. [Environment](#environment)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Monitoring](#monitoring)
7. [Configuration](#configuration)
8. [Utilities](#utilities)

## Data Loading

### MultiTickerDataLoader

The `MultiTickerDataLoader` class is responsible for loading and preprocessing market data for multiple tickers.

#### `__init__(self, config)`

Initialize the data loader with configuration.

**Parameters**:
- `config` (dict): Configuration dictionary containing data loading parameters

**Returns**:
- `MultiTickerDataLoader`: Initialized data loader

**Example**:
```python
from src.data.multiticker_data_loader import MultiTickerDataLoader
from src.utils.config_loader import load_config

config = load_config()
data_loader = MultiTickerDataLoader(config['data'])
```

#### `load_data(self, tickers=None, start_date=None, end_date=None)`

Load market data for specified tickers and date range.

**Parameters**:
- `tickers` (list, optional): List of tickers to load. If None, uses tickers from config
- `start_date` (str, optional): Start date for data loading (format: 'YYYY-MM-DD')
- `end_date` (str, optional): End date for data loading (format: 'YYYY-MM-DD')

**Returns**:
- `pd.DataFrame`: DataFrame with multi-level columns (ticker, feature)

**Example**:
```python
# Load data for specific tickers
data = data_loader.load_data(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

#### `get_universe(self, date=None)`

Get current universe of tickers.

**Parameters**:
- `date` (str, optional): Date for universe selection. If None, uses current date

**Returns**:
- `list`: List of tickers in the current universe

**Example**:
```python
# Get current universe
universe = data_loader.get_universe()
print(f"Current universe: {universe}")
```

#### `get_ticker_data(self, ticker, start_date=None, end_date=None)`

Get data for a specific ticker.

**Parameters**:
- `ticker` (str): Ticker symbol
- `start_date` (str, optional): Start date for data loading
- `end_date` (str, optional): End date for data loading

**Returns**:
- `pd.DataFrame`: DataFrame with data for the specified ticker

**Example**:
```python
# Get data for a specific ticker
aapl_data = data_loader.get_ticker_data('AAPL')
print(f"AAPL data shape: {aapl_data.shape}")
```

### DynamicUniverseSelector

The `DynamicUniverseSelector` class implements dynamic universe selection strategies.

#### `__init__(self, config)`

Initialize the universe selector with configuration.

**Parameters**:
- `config` (dict): Configuration dictionary containing universe selection parameters

**Returns**:
- `DynamicUniverseSelector`: Initialized universe selector

**Example**:
```python
from src.data.multiticker_data_loader import DynamicUniverseSelector

config = {
    'max_tickers': 10,
    'min_tickers': 3,
    'rebalance_freq': '1M',
    'selection_metrics': ['liquidity', 'volatility', 'trend_strength']
}
universe_selector = DynamicUniverseSelector(config)
```

#### `select_universe(self, data, current_universe=None, date=None