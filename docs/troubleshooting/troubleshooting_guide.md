
# Multi-Ticker RL Trading System Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when using the Multi-Ticker RL Trading System. It covers installation problems, runtime errors, performance issues, and debugging techniques.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Loading Problems](#data-loading-problems)
3. [Training Issues](#training-issues)
4. [Evaluation Problems](#evaluation-problems)
5. [Performance Issues](#performance-issues)
6. [Configuration Errors](#configuration-errors)
7. [Memory and GPU Issues](#memory-and-gpu-issues)
8. [Debugging Techniques](#debugging-techniques)
9. [Common Error Messages](#common-error-messages)
10. [Getting Help](#getting-help)

## Installation Issues

### Python Environment Setup

#### Problem: Virtual environment creation fails
```bash
Error: Command '['/usr/bin/python3', '-m', 'venv', '.venv']' returned non-zero exit status 1.
```

**Solution**:
```bash
# Ensure python3-venv is installed
sudo apt install python3-venv  # Ubuntu/Debian
sudo yum install python3-virtualenv  # CentOS/RHEL

# Try alternative creation method
python3 -m virtualenv .venv

# Or use conda
conda create -n rl-intraday python=3.8
```

#### Problem: Package installation fails
```bash
ERROR: Could not build wheels for torch which use PEP 517 and cannot be installed directly
```

**Solution**:
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install PyTorch with specific CUDA version
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# For CPU-only installation
pip install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

#### Problem: CUDA compatibility issues
```bash
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with correct CUDA version
# Find your CUDA version and install matching PyTorch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu<version>
```

### System Dependencies

#### Problem: Missing system libraries
```bash
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install mesa-libGL glib2
```

#### Problem: OpenCV installation fails
```bash
ModuleNotFoundError: No module named 'cv2'
```

**Solution**:
```bash
# Install OpenCV dependencies
sudo apt install libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev

# Install OpenCV
pip install opencv-python-headless
```

## Data Loading Problems

### API Connection Issues

#### Problem: Polygon API connection fails
```bash
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.polygon.io', port=443): Max retries exceeded
```

**Solution**:
```bash
# Check API key
echo $POLYGON_API_KEY

# Test API connection
curl -H "Authorization: Bearer $POLYGON_API_KEY" "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02"

# Check network connectivity
ping api.polygon.io

# Verify API key in configuration
python -c "
from src.utils.config_loader import load_config
config = load_config()
print('API Key:', config.get('secrets', {}).get('polygon_api_key', 'Not found'))
"
```

#### Problem: Rate limiting errors
```bash
HTTPError: 429 Client Error: Too Many Requests
```

**Solution**:
```yaml
# Update configuration to handle rate limiting
data:
  polygon:
    rate_limit:
      requests_per_minute: 5
      retry_attempts: 3
      retry_delay: 1.0
    batch_size: 50
    delay_between_batches: 1.0
```

### Data Format Issues

#### Problem: Invalid data format
```bash
ValueError: could not convert string to float: 'null'
```

**Solution**:
```python
# Add data validation in data loader
def clean_data(df):
    # Replace null strings with NaN
    df = df.replace('null', np.nan)
    
    # Convert to numeric, coercing errors to NaN
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna(subset=numeric_cols)
    
    return df

# Use in data loading
data = clean_data(data)
```

#### Problem: Missing data columns
```bash
KeyError: 'vwap'
```

**Solution**:
```python
# Check available columns
print(data.columns)

# Add missing columns with default values
def add_missing_columns(df):
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'vwap']
    
    for col in required_columns:
        if col not in df.columns:
            if col == 'vwap':
                # Calculate VWAP if missing
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            else:
                # Use default value
                df[col] = 0.0
    
    return df

# Use in data loading
data = add_missing_columns(data)
```

### Memory Issues

#### Problem: Out of memory when loading large datasets
```bash
MemoryError: Unable to allocate array with shape (1000000, 50) and data type float64
```

**Solution**:
```python
# Load data in chunks
def load_data_in_chunks(tickers, start_date, end_date, chunk_size='1M'):
    all_data = []
    
    # Generate date chunks
    dates = pd.date_range(start_date, end_date, freq=chunk_size)
    
    for i in range(len(dates) - 1):
        chunk_start = dates[i]
        chunk_end = dates[i + 1] - pd.Timedelta(days=1)
        
        print(f"Loading chunk {i+1}/{len(dates)-1}: {chunk_start} to {chunk_end}")
        
        # Load chunk
        chunk_data = data_loader.load_data(
            tickers=tickers,
            start_date=chunk_start.strftime('%Y-%m-%d'),
            end_date=chunk_end.strftime('%Y-%m-%d')
        )
        
        all_data.append(chunk_data)
    
    # Combine chunks
    combined_data = pd.concat(all_data, axis=0)
    
    return combined_data

# Use chunked loading
data = load_data_in_chunks(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    chunk_size='3M'
)
```

## Training Issues

### Environment Initialization

#### Problem: Environment creation fails
```bash
ValueError: Observation space mismatch
```

**Solution**:
```python
# Check observation space
print(f"Expected observation space: {env.observation_space}")
print(f"Actual observation shape: {observation.shape}")

# Fix observation space
def fix_observation_space(env, observation):
    # Reshape observation if needed
    if len(observation.shape) == 1:
        observation = observation.reshape(1, -1)
    
    # Clip values to observation space bounds
    observation = np.clip(
        observation,
        env.observation_space.low,
        env.observation_space.high
    )
    
    return observation

# Use in training
observation = fix_observation_space(env, observation)
```

#### Problem: Action space mismatch
```bash
ValueError: Action space mismatch
```

**Solution**:
```python
# Check action space
print(f"Expected action space: {env.action_space}")
print(f"Action shape: {action.shape}")

# Fix action space
def fix_action_space(env, action):
    # Ensure action is within bounds
    action = np.clip(action, env.action_space.low, env.action_space.high)
    
    # Reshape if needed
    if len(action.shape) == 1 and len(env.action_space.shape) > 1:
        action = action.reshape(env.action_space.shape)
    
    return action

# Use in training
action = fix_action_space(env, action)
```

### Model Training

#### Problem: Training loss becomes NaN
```bash
RuntimeWarning: invalid value encountered in multiply
```

**Solution**:
```python
# Add gradient clipping
config['training']['max_grad_norm'] = 0.5

# Reduce learning rate
config['training']['learning_rate'] = 0