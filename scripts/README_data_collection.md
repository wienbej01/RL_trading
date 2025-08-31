# Polygon Data Collection for RL Trading System

This directory contains scripts for collecting and validating historical market data from Polygon API for use in the RL trading system.

## Prerequisites

1. **Polygon API Key**: Sign up at [polygon.io](https://polygon.io) for a free API key
2. **Environment Variable**: Set `POLYGON_API_KEY` in your environment
3. **Python Dependencies**: Install required packages:
   ```bash
   pip install aiohttp backoff tqdm pandas pyarrow
   ```

## Data Collection Script

### `collect_polygon_data.py`

Collects historical OHLCV and quotes data from Polygon API with rate limiting compliance.

#### Usage Examples

**Pilot Portfolio (Recommended for initial testing):**
```bash
python scripts/collect_polygon_data.py --preset pilot --start-date 2024-01-01 --end-date 2024-06-30
```

**Extended Portfolio (Comprehensive testing):**
```bash
python scripts/collect_polygon_data.py --preset extended --start-date 2024-01-01 --end-date 2024-06-30
```

**Custom Symbols:**
```bash
python scripts/collect_polygon_data.py --symbols SPY,QQQ,AAPL --start-date 2024-01-01 --end-date 2024-06-30
```

**Include Quotes Data:**
```bash
python scripts/collect_polygon_data.py --preset pilot --start-date 2024-01-01 --end-date 2024-06-30 --include-quotes
```

#### Command Line Options

- `--symbols`: Comma-separated list of stock symbols
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)
- `--preset`: Use predefined portfolio ('pilot' or 'extended')
- `--include-quotes`: Include bid/ask quotes data (additional API calls)
- `--api-key`: Polygon API key (or set POLYGON_API_KEY env var)
- `--data-dir`: Custom data directory path

#### Pilot Portfolio

High-volume, liquid stocks suitable for options trading:
- **SPY**: SPDR S&P 500 ETF (Most liquid ETF)
- **QQQ**: Invesco QQQ ETF (Tech-heavy)
- **AAPL**: Apple Inc. (High volume, stable)
- **MSFT**: Microsoft Corp. (High volume, stable)
- **TSLA**: Tesla Inc. (High volatility)
- **NVDA**: NVIDIA Corp. (High volatility, tech)
- **AMD**: Advanced Micro Devices (High volatility, tech)
- **GOOGL**: Alphabet Inc. (High volume, stable)

#### Extended Portfolio

Includes additional stocks for comprehensive testing:
- All pilot stocks plus AMZN, META, NFLX, BA, XOM, JPM, V

## Data Validation Script

### `validate_collected_data.py`

Validates collected data quality and tests RL environment compatibility.

#### Usage Examples

**Validate all collected symbols:**
```bash
python scripts/validate_collected_data.py
```

**Validate specific symbols:**
```bash
python scripts/validate_collected_data.py --symbols SPY,QQQ
```

**Test RL environment:**
```bash
python scripts/validate_collected_data.py --test-rl --test-symbol SPY
```

**Custom data directory:**
```bash
python scripts/validate_collected_data.py --data-dir /path/to/data
```

## Data Storage Structure

Data is stored in partitioned Parquet format for optimal performance:

```
data/polygon/historical/
├── symbol=SPY/
│   ├── year=2024/
│   │   ├── month=01/
│   │   │   ├── day=01/
│   │   │   │   └── data.parquet
│   │   │   └── day=02/
│   │   │       └── data.parquet
│   │   └── month=02/
│   │       └── ...
│   └── year=2025/
│       └── ...
├── symbol=QQQ/
│   └── ...
└── collection_summary.json
```

## Rate Limiting

The scripts respect Polygon's free tier limits:
- **5 API calls per minute**
- **Automatic rate limiting** with 12-second delays between calls
- **Retry logic** with exponential backoff for failed requests
- **Progress tracking** to monitor collection status

## Data Quality Checks

The validation script performs comprehensive quality checks:

### OHLCV Data Validation
- ✅ Required columns present (open, high, low, close, volume)
- ✅ DatetimeIndex format
- ✅ No missing values in critical columns
- ✅ Price consistency (high >= low, etc.)
- ✅ Volume validation (non-negative)
- ✅ Timestamp continuity

### RL Environment Compatibility
- ✅ Data format compatibility
- ✅ Feature extraction validation
- ✅ Environment initialization
- ✅ Basic simulation testing

## Expected Data Volume

### Pilot Portfolio (3 months, OHLCV only)
- **Time Period**: 90 trading days
- **Trading Hours**: ~6.5 hours/day = 390 minutes
- **Total Records**: ~35,000 per symbol
- **Storage Size**: ~50-100MB per symbol
- **API Calls**: 8 symbols × 3 months = 24 calls

### Extended Portfolio (6 months, OHLCV + Quotes)
- **Time Period**: 180 trading days
- **Total Records**: ~500,000 per symbol (OHLCV)
- **Quotes Data**: ~2-5 million records per symbol
- **Storage Size**: ~1-2GB per symbol
- **API Calls**: 15 symbols × 6 months × 2 = 180 calls

## Next Steps After Data Collection

1. **Validate Data Quality**:
   ```bash
   python scripts/validate_collected_data.py --test-rl
   ```

2. **Run RL Training**:
   ```bash
   python src/rl/train.py --config configs/settings.yaml
   ```

3. **Backtest Performance**:
   ```bash
   python src/evaluation/backtest_evaluator.py --symbols SPY,QQQ
   ```

## Troubleshooting

### Common Issues

**Rate Limit Errors**:
- Script automatically handles rate limiting
- Reduce `--symbols` count if issues persist
- Use shorter date ranges

**Missing Data**:
- Some symbols may have limited historical data
- Check Polygon API documentation for symbol availability
- Try different date ranges

**Storage Issues**:
- Ensure sufficient disk space (1-2GB per symbol for extended data)
- Data is compressed efficiently with Parquet format

**API Key Issues**:
- Verify `POLYGON_API_KEY` environment variable
- Check API key validity on Polygon dashboard
- Free tier has request limits

### Performance Optimization

- **Parallel Processing**: Script processes symbols sequentially to respect rate limits
- **Chunking**: Monthly data chunks minimize API calls
- **Caching**: Collected data is cached locally to avoid re-downloading
- **Compression**: Parquet format provides excellent compression ratios

## Support

For issues with data collection:
1. Check the `collection_summary.json` file for detailed error logs
2. Verify API key and network connectivity
3. Review Polygon API status and documentation
4. Check available disk space and permissions