# Polygon.io Connection Guide

This guide explains how to connect to Polygon.io for market data retrieval from another trading repository.

## Prerequisites

- Python 3.8+
- Polygon.io API key (free or paid)
- Required packages: `requests`, `pandas`, `tenacity`, `pydantic-settings`

## 1. Dependencies Installation

Install the required Python packages:

```bash
pip install requests pandas tenacity pydantic-settings
```

## 2. API Key Setup

### Get Polygon.io API Key
1. Sign up at [polygon.io](https://polygon.io)
2. Get your API key from the dashboard
3. Choose a plan based on your needs:
   - **Free tier**: 5 API calls/minute, 2 years of historical data
   - **Paid plans**: Higher limits, real-time data, more history

### Environment Configuration

Create a `.env` file in your project root:

```env
# Polygon.io API Configuration
POLYGON_API_KEY=your_api_key_here

# Optional: Other settings if needed
GCS_BUCKET=your_bucket_name
GCP_PROJECT=your_gcp_project
```

## 3. Module Setup

### Option A: Copy Module Files

Copy these files from the trade_system_modules repository to your project:

```
your_project/
├── src/
│   └── trade_system_modules/
│       ├── config/
│       │   └── settings.py
│       ├── data/
│       │   └── polygon_adapter.py
│       └── schemas/
│           └── bars.py
├── .env
└── your_data_script.py
```

### Option B: Install as Package

From the trade_system_modules directory:

```bash
pip install -e .
```

## 4. Basic Usage Example

```python
#!/usr/bin/env python3
"""
Basic Polygon.io data retrieval example
"""

import sys
import os
from pathlib import Path

# Add module path (if using Option A)
sys.path.append(str(Path(__file__).parent / 'src'))

from trade_system_modules.data.polygon_adapter import get_agg_minute
from trade_system_modules.config.settings import settings

def main():
    """Main data retrieval function."""
    try:
        print("=== Polygon.io Data Retrieval Example ===\n")

        # 1. Configuration check
        print("1. Configuration Check")
        if not settings.polygon_api_key:
            print("❌ POLYGON_API_KEY not found in environment variables")
            print("Please set your API key in the .env file")
            return

        print(f"✅ Polygon API Key configured: {settings.polygon_api_key[:8]}...")
        print()

        # 2. Get market data
        print("2. Retrieving Market Data")
        symbol = "AAPL"
        start_date = "2023-01-01"
        end_date = "2023-01-02"

        print(f"Fetching {symbol} minute data from {start_date} to {end_date}")

        data = get_agg_minute(symbol, start_date, end_date)

        print(f"✅ Retrieved {len(data)} bars")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Columns: {list(data.columns)}")
        print()

        # 3. Data inspection
        print("3. Data Preview")
        if not data.empty:
            print("First 5 rows:")
            print(data.head().to_string())
            print()

            print("Data statistics:")
            print(f"  Start time: {data['ts'].min()}")
            print(f"  End time: {data['ts'].max()}")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(f"  Total volume: {data['volume'].sum():,.0f}")
            print(f"  Total trades: {data['trades'].sum():,.0f}")
        else:
            print("No data returned - check symbol and date range")
        print()

        # 4. Error handling example
        print("4. Error Handling Example")
        try:
            invalid_data = get_agg_minute("INVALID_SYMBOL", "2023-01-01", "2023-01-02")
            print(f"Invalid symbol returned {len(invalid_data)} rows")
        except Exception as e:
            print(f"Expected error for invalid symbol: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your API key is valid and you have internet connection")

if __name__ == "__main__":
    main()
```

## 5. Advanced Usage Examples

### Multiple Symbols
```python
from trade_system_modules.data.polygon_adapter import get_agg_minute
import pandas as pd

def get_multiple_symbols(symbols, start_date, end_date):
    """Get data for multiple symbols."""
    data_dict = {}

    for symbol in symbols:
        try:
            data = get_agg_minute(symbol, start_date, end_date)
            data_dict[symbol] = data
            print(f"✅ {symbol}: {len(data)} bars")
        except Exception as e:
            print(f"❌ {symbol}: {e}")
            data_dict[symbol] = pd.DataFrame()

    return data_dict

# Usage
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
start_date = "2023-01-01"
end_date = "2023-01-02"

data = get_multiple_symbols(symbols, start_date, end_date)
```

### Date Range Processing
```python
import pandas as pd
from datetime import datetime, timedelta

def get_date_range_data(symbol, start_date, end_date, chunk_days=30):
    """Get data for large date ranges in chunks."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_data = []

    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_days), end)

        print(f"Fetching {symbol} from {current_start.date()} to {current_end.date()}")

        try:
            chunk_data = get_agg_minute(
                symbol,
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d")
            )
            all_data.append(chunk_data)
            print(f"  Retrieved {len(chunk_data)} bars")
        except Exception as e:
            print(f"  Error: {e}")

        current_start = current_end

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('ts').reset_index(drop=True)
        return combined_data
    else:
        return pd.DataFrame()

# Usage
large_dataset = get_date_range_data("AAPL", "2023-01-01", "2023-12-31")
print(f"Total bars retrieved: {len(large_dataset)}")
```

### Data Validation and Processing
```python
from trade_system_modules.schemas.bars import ensure_bar_schema

def process_and_validate_data(symbol, start_date, end_date):
    """Get and validate market data."""
    try:
        # Get raw data
        raw_data = get_agg_minute(symbol, start_date, end_date)

        if raw_data.empty:
            print(f"No data available for {symbol}")
            return pd.DataFrame()

        # Ensure schema compliance
        clean_data = ensure_bar_schema(raw_data)

        # Additional validation
        if clean_data['low'].max() > clean_data['high'].max():
            raise ValueError("Data integrity issue: low > high")

        if (clean_data['volume'] < 0).any():
            raise ValueError("Negative volume detected")

        print(f"✅ {symbol}: {len(clean_data)} validated bars")
        return clean_data

    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
        return pd.DataFrame()

# Usage
validated_data = process_and_validate_data("AAPL", "2023-01-01", "2023-01-02")
```

## 6. Rate Limiting and Best Practices

### Understanding Rate Limits
```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Custom retry decorator for rate limiting
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=lambda exc: hasattr(exc, 'response') and exc.response.status_code == 429
)
def rate_limited_get_agg_minute(symbol, start, end):
    """Get data with custom rate limiting."""
    return get_agg_minute(symbol, start, end)

# Usage
try:
    data = rate_limited_get_agg_minute("AAPL", "2023-01-01", "2023-01-02")
    print(f"Successfully retrieved {len(data)} bars")
except Exception as e:
    print(f"Failed after retries: {e}")
```

### Batch Processing with Rate Limiting
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

def get_batch_with_rate_limit(symbols, start_date, end_date, requests_per_minute=4):
    """Get data for multiple symbols with rate limiting."""
    data_dict = {}
    delay = 60 / requests_per_minute  # Delay between requests

    for i, symbol in enumerate(symbols):
        try:
            print(f"Fetching {symbol} ({i+1}/{len(symbols)})")

            data = get_agg_minute(symbol, start_date, end_date)
            data_dict[symbol] = data

            print(f"✅ {symbol}: {len(data)} bars")

            # Rate limiting delay (skip for last request)
            if i < len(symbols) - 1:
                print(f"Waiting {delay:.1f}s for rate limit...")
                time.sleep(delay)

        except Exception as e:
            print(f"❌ {symbol}: {e}")
            data_dict[symbol] = pd.DataFrame()

    return data_dict

# Usage
symbols = ["AAPL", "MSFT", "GOOGL"]
data = get_batch_with_rate_limit(symbols, "2023-01-01", "2023-01-02")
```

## 7. Data Storage and Caching

### Save to CSV
```python
def save_to_csv(data, symbol, filename=None):
    """Save data to CSV file."""
    if filename is None:
        filename = f"{symbol}_data.csv"

    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return filename

# Usage
data = get_agg_minute("AAPL", "2023-01-01", "2023-01-02")
save_to_csv(data, "AAPL")
```

### Simple Caching
```python
import pickle
from pathlib import Path
import hashlib

def get_cached_data(symbol, start_date, end_date, cache_dir=".cache"):
    """Get data with file-based caching."""
    # Create cache key
    cache_key = hashlib.md5(
        f"{symbol}_{start_date}_{end_date}".encode()
    ).hexdigest()

    cache_file = Path(cache_dir) / f"{cache_key}.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    # Check cache
    if cache_file.exists():
        print(f"Loading cached data for {symbol}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Fetch new data
    print(f"Fetching fresh data for {symbol}")
    data = get_agg_minute(symbol, start_date, end_date)

    # Cache the data
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

    return data

# Usage
cached_data = get_cached_data("AAPL", "2023-01-01", "2023-01-02")
```

## 8. Error Handling and Troubleshooting

### Common Errors and Solutions

```python
def robust_data_fetch(symbol, start_date, end_date):
    """Fetch data with comprehensive error handling."""

    try:
        data = get_agg_minute(symbol, start_date, end_date)

        # Check for empty response
        if data.empty:
            print(f"⚠️  No data returned for {symbol}")
            return pd.DataFrame()

        # Validate data integrity
        if len(data) == 0:
            print(f"⚠️  Empty dataset for {symbol}")
            return pd.DataFrame()

        print(f"✅ Successfully retrieved {len(data)} bars for {symbol}")
        return data

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("❌ Invalid API key")
        elif e.response.status_code == 403:
            print("❌ API key not authorized for this endpoint")
        elif e.response.status_code == 429:
            print("❌ Rate limit exceeded")
        else:
            print(f"❌ HTTP error {e.response.status_code}: {e}")

    except requests.exceptions.ConnectionError:
        print("❌ Network connection error")

    except requests.exceptions.Timeout:
        print("❌ Request timeout")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    return pd.DataFrame()

# Usage
data = robust_data_fetch("AAPL", "2023-01-01", "2023-01-02")
```

### API Key Validation
```python
def validate_api_key():
    """Test API key validity."""
    try:
        # Try to fetch a small amount of data
        test_data = get_agg_minute("AAPL", "2023-01-01", "2023-01-01")

        if not test_data.empty:
            print("✅ API key is valid")
            return True
        else:
            print("⚠️  API key may be valid but no data returned")
            return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("❌ Invalid API key")
            return False
        else:
            print(f"❌ API error: {e}")
            return False

    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

# Usage
if not validate_api_key():
    print("Please check your POLYGON_API_KEY in the .env file")
```

## 9. Performance Optimization

### Async Data Fetching
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_get_agg_minute(symbol, start_date, end_date):
    """Async wrapper for get_agg_minute."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        data = await loop.run_in_executor(
            executor, get_agg_minute, symbol, start_date, end_date
        )
    return data

async def get_multiple_symbols_async(symbols, start_date, end_date):
    """Fetch multiple symbols concurrently."""
    tasks = [
        async_get_agg_minute(symbol, start_date, end_date)
        for symbol in symbols
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    data_dict = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            print(f"❌ {symbol}: {result}")
            data_dict[symbol] = pd.DataFrame()
        else:
            print(f"✅ {symbol}: {len(result)} bars")
            data_dict[symbol] = result

    return data_dict

# Usage
symbols = ["AAPL", "MSFT", "GOOGL"]
data = asyncio.run(get_multiple_symbols_async(symbols, "2023-01-01", "2023-01-02"))
```

## 10. Integration with Other Systems

### Combine with IBKR Data
```python
from trade_system_modules.data.polygon_adapter import get_agg_minute
from trade_system_modules.execution.ibkr_exec import IBExec

def compare_data_sources(symbol, start_date, end_date):
    """Compare data from Polygon and IBKR."""

    # Get Polygon data
    try:
        polygon_data = get_agg_minute(symbol, start_date, end_date)
        print(f"Polygon: {len(polygon_data)} bars")
    except Exception as e:
        print(f"Polygon error: {e}")
        polygon_data = pd.DataFrame()

    # Get IBKR data (if available)
    try:
        ib_exec = IBExec()
        # Note: IBKR data retrieval would need additional implementation
        print("IBKR connection established")
        ib_exec.ib.disconnect()
    except Exception as e:
        print(f"IBKR error: {e}")

    return {
        "polygon": polygon_data,
        "ibkr": pd.DataFrame()  # Placeholder
    }

# Usage
comparison = compare_data_sources("AAPL", "2023-01-01", "2023-01-02")
```

## 11. Configuration Options

### Environment Variables
```env
# Polygon.io
POLYGON_API_KEY=your_api_key_here

# Optional: Proxy settings
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=http://proxy.company.com:8080

# Optional: Request timeout
REQUEST_TIMEOUT=30
```

### Runtime Configuration
```python
import os

# Custom timeout
os.environ['REQUEST_TIMEOUT'] = '60'

# Custom retry settings
custom_retry_config = {
    'stop': stop_after_attempt(3),
    'wait': wait_exponential(multiplier=2, min=4, max=30)
}
```

## 12. Monitoring and Logging

### Request Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def logged_get_agg_minute(symbol, start_date, end_date):
    """Get data with comprehensive logging."""

    logger.info(f"Starting data fetch for {symbol} from {start_date} to {end_date}")

    try:
        data = get_agg_minute(symbol, start_date, end_date)

        logger.info(f"Successfully retrieved {len(data)} bars for {symbol}")
        logger.debug(f"Data shape: {data.shape}")
        logger.debug(f"Date range: {data['ts'].min()} to {data['ts'].max()}")

        return data

    except Exception as e:
        logger.error(f"Failed to get data for {symbol}: {e}")
        raise

# Usage
data = logged_get_agg_minute("AAPL", "2023-01-01", "2023-01-02")
```

## 13. File Structure Reference

```
your_project/
├── .env                           # Environment variables
├── requirements.txt               # Python dependencies
├── src/
│   └── trade_system_modules/      # Copied module files
│       ├── config/
│       │   └── settings.py        # Configuration management
│       ├── data/
│       │   └── polygon_adapter.py # Polygon data adapter
│       └── schemas/
│           └── bars.py            # Data schema validation
├── scripts/
│   ├── fetch_data.py              # Data fetching scripts
│   └── validate_api.py            # API validation script
├── data/                          # Data storage directory
│   ├── raw/                       # Raw data files
│   └── processed/                 # Processed data files
└── cache/                         # Cache directory
    └── *.pkl                      # Cached data files
```

## 14. Best Practices

1. **API Key Security**: Never commit API keys to version control
2. **Rate Limiting**: Respect API limits to avoid being blocked
3. **Error Handling**: Implement comprehensive error handling
4. **Caching**: Cache frequently accessed data to reduce API calls
5. **Data Validation**: Always validate data integrity
6. **Logging**: Log important events for debugging
7. **Testing**: Test with various market conditions
8. **Monitoring**: Monitor API usage and performance

## 15. Troubleshooting

### Common Issues

**"Invalid API Key"**
- Check your API key in the .env file
- Ensure the key is active and not expired
- Verify the key has the correct permissions

**"Rate Limit Exceeded"**
- Implement rate limiting in your code
- Use the built-in retry mechanism
- Consider upgrading your Polygon.io plan

**"No Data Returned"**
- Check symbol spelling
- Verify date range is valid
- Ensure the symbol has data for the requested period

**"Connection Timeout"**
- Check your internet connection
- Increase timeout settings
- Try again later

### Debug Mode
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed request/response information
data = get_agg_minute("AAPL", "2023-01-01", "2023-01-02")
```

## Support

For issues with Polygon.io:
- Check [Polygon.io Documentation](https://polygon.io/docs)
- Review API status at [status.polygon.io](https://status.polygon.io)
- Contact Polygon.io support for account-specific issues

For issues with the adapter code:
- Check the error messages and logs
- Verify your Python environment
- Ensure all dependencies are installed correctly