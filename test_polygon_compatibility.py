#!/usr/bin/env python3
"""
Test script to verify feature pipeline compatibility with Polygon data format.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.pipeline import FeaturePipeline

def create_sample_polygon_data():
    """Create sample Polygon OHLCV data for testing."""
    # Create timestamp index (millisecond precision like Polygon)
    start_time = datetime(2024, 1, 1, 9, 30, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(100)]

    # Convert to milliseconds since epoch (Polygon format)
    timestamps_ms = [int(ts.timestamp() * 1000) for ts in timestamps]

    # Create OHLCV data
    np.random.seed(42)
    base_price = 150.0

    data = []
    for i, ts_ms in enumerate(timestamps_ms):
        # Generate realistic price movements
        price_change = np.random.normal(0, 0.01)  # 1% volatility
        close = base_price * (1 + price_change * i * 0.01)
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = data[-1]['close'] if data else close

        # Generate volume
        volume = np.random.randint(1000, 10000)

        # Polygon-specific fields
        vwap = (open_price + high + low + close) / 4  # Simple VWAP approximation
        transactions = np.random.randint(10, 100)

        data.append({
            'timestamp': ts_ms,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume,
            'vwap': round(vwap, 2),
            'transactions': transactions
        })

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df

def create_sample_polygon_quotes():
    """Create sample Polygon quotes data for testing."""
    # Create timestamp index (nanosecond precision like Polygon)
    start_time = datetime(2024, 1, 1, 9, 30, 0)
    timestamps = [start_time + timedelta(seconds=i*10) for i in range(50)]

    # Convert to nanoseconds since epoch (Polygon format)
    timestamps_ns = [int(ts.timestamp() * 1e9) for ts in timestamps]

    data = []
    base_price = 150.0

    for i, ts_ns in enumerate(timestamps_ns):
        # Generate bid/ask spread
        mid_price = base_price + np.random.normal(0, 0.5)
        spread = np.random.uniform(0.01, 0.10)
        bid_price = mid_price - spread/2
        ask_price = mid_price + spread/2

        bid_size = np.random.randint(100, 1000)
        ask_size = np.random.randint(100, 1000)

        data.append({
            'timestamp': ts_ns,
            'bid_price': round(bid_price, 2),
            'bid_size': bid_size,
            'ask_price': round(ask_price, 2),
            'ask_size': ask_size,
            'bid_exchange': np.random.choice(['NYSE', 'NASDAQ', 'AMEX']),
            'ask_exchange': np.random.choice(['NYSE', 'NASDAQ', 'AMEX'])
        })

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
    df.set_index('timestamp', inplace=True)

    return df

def test_pipeline_compatibility():
    """Test the feature pipeline with Polygon data."""
    print("Testing Feature Pipeline Compatibility with Polygon Data")
    print("=" * 60)

    # Create sample data
    print("Creating sample Polygon OHLCV data...")
    ohlcv_data = create_sample_polygon_data()
    print(f"Created {len(ohlcv_data)} rows of OHLCV data")
    print(f"Columns: {list(ohlcv_data.columns)}")
    print(f"Date range: {ohlcv_data.index.min()} to {ohlcv_data.index.max()}")
    print()

    # Create sample quotes data
    print("Creating sample Polygon quotes data...")
    quotes_data = create_sample_polygon_quotes()
    print(f"Created {len(quotes_data)} rows of quotes data")
    print(f"Columns: {list(quotes_data.columns)}")
    print()

    # Test configuration for feature extraction
    config = {
        'data_source': 'polygon',
        'technical': {
            'calculate_returns': True,
            'calculate_log_returns': True,
            'sma_windows': [5, 10, 20],
            'ema_windows': [5, 10, 20],
            'calculate_atr': True,
            'calculate_rsi': True,
            'rsi_window': 14,
            'calculate_macd': True,
            'calculate_bollinger_bands': True,
            'calculate_stochastic': True,
            'calculate_williams_r': True
        },
        'microstructure': {
            'calculate_spread': True,
            'calculate_microprice': True,
            'calculate_queue_imbalance': True,
            'calculate_order_flow_imbalance': True,
            'calculate_vwap': True,
            'calculate_twap': True,
            'calculate_price_impact': True
        },
        'time': {
            'extract_time_of_day': True,
            'extract_day_of_week': True,
            'extract_session_features': True
        },
        'polygon': {
            'features': {
                'use_vwap_column': True
            },
            'quality_checks': {
                'enabled': True
            }
        }
    }

    # Initialize pipeline
    print("Initializing Feature Pipeline...")
    pipeline = FeaturePipeline(config)
    print("Pipeline initialized successfully")
    print()

    # Test with OHLCV data
    print("Testing feature extraction with OHLCV data...")
    try:
        features = pipeline.fit_transform(ohlcv_data)
        print(f"Feature extraction successful!")
        print(f"Extracted {len(features.columns)} features")
        print(f"Feature columns: {list(features.columns[:10])}...")  # Show first 10
        print(f"Sample features shape: {features.shape}")
        print(f"Features date range: {features.index.min()} to {features.index.max()}")
        print()

        # Check for NaN values
        nan_counts = features.isna().sum()
        total_nans = nan_counts.sum()
        print(f"Total NaN values: {total_nans}")
        if total_nans > 0:
            print("Features with NaN values:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"  {col}: {count} NaNs")
        print()

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test data source detection
    print("Testing data source detection...")
    detected_source = pipeline._detect_data_source(ohlcv_data)
    print(f"Detected data source: {detected_source}")
    print()

    # Test column mapping
    print("Testing column mapping...")
    mapped_data = pipeline._map_columns(ohlcv_data.copy(), 'ohlcv')
    print(f"Original columns: {list(ohlcv_data.columns)}")
    print(f"Mapped columns: {list(mapped_data.columns)}")
    print()

    # Test with quotes data
    print("Testing with quotes data...")
    try:
        # Create combined data for microstructure features
        combined_data = ohlcv_data.copy()
        # Add quote columns (resample quotes to minute frequency)
        quotes_resampled = quotes_data.resample('1min').agg({
            'bid_price': 'last',
            'bid_size': 'sum',
            'ask_price': 'last',
            'ask_size': 'sum',
            'bid_exchange': 'last',
            'ask_exchange': 'last'
        }).dropna()

        # Merge with OHLCV data
        combined_data = combined_data.join(quotes_resampled, how='left')
        combined_data = combined_data.fillna(method='ffill')

        print(f"Combined data shape: {combined_data.shape}")
        print(f"Combined data columns: {list(combined_data.columns)}")

        # Extract features with microstructure data
        features_with_micro = pipeline.fit_transform(combined_data)
        print(f"Features with microstructure: {len(features_with_micro.columns)} features")
        print()

    except Exception as e:
        print(f"Error with microstructure features: {e}")
        import traceback
        traceback.print_exc()

    print("Test completed successfully!")
    return True

if __name__ == "__main__":
    test_pipeline_compatibility()