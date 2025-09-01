#!/usr/bin/env python3
"""
Test script to verify RL environment compatibility with Polygon data format.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.sim.env_intraday_rl import IntradayRLEnv
from src.features.pipeline import FeaturePipeline
from src.utils.config_loader import Settings

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

def test_environment_with_polygon_data():
    """Test the RL environment with Polygon data."""
    print("Testing RL Environment with Polygon Data")
    print("=" * 50)

    # Create sample Polygon data
    print("Creating sample Polygon OHLCV data...")
    ohlcv_data = create_sample_polygon_data()
    print(f"Created {len(ohlcv_data)} rows of OHLCV data")
    print(f"Columns: {list(ohlcv_data.columns)}")
    print(f"Date range: {ohlcv_data.index.min()} to {ohlcv_data.index.max()}")
    print()

    # Create feature pipeline configuration
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

    # Initialize feature pipeline
    print("Initializing Feature Pipeline...")
    pipeline = FeaturePipeline(config)
    print("Pipeline initialized successfully")
    print()

    # Extract features
    print("Extracting features from Polygon data...")
    features = pipeline.fit_transform(ohlcv_data)
    print(f"Feature extraction successful!")
    print(f"Extracted {len(features.columns)} features")
    print(f"Features shape: {features.shape}")
    print()

    # Initialize RL environment
    print("Initializing RL Environment...")
    try:
        # Load settings from config file
        settings = Settings.from_paths('configs/settings.yaml')

        env = IntradayRLEnv(
            ohlcv=ohlcv_data,
            features=features,
            cash=100000.0,
            exec_params={'transaction_cost': 2.5, 'slippage': 0.01},
            point_value=5.0,
            config=settings._config
        )
        print("Environment initialized successfully!")
        print(f"Data source detected: {getattr(env, 'data_source', 'unknown')}")
        print()

        # Test environment reset
        print("Testing environment reset...")
        obs, info = env.reset()
        print(f"Reset successful! Observation shape: {obs.shape}")
        print(f"Initial observation: {obs[:5]}...")  # Show first 5 values
        print()

        # Test a few steps
        print("Testing environment steps...")
        for step in range(5):
            action = np.random.randint(0, 3)  # Random action
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {step + 1}: Action={action}, Reward={reward:.4f}, Done={done}")
            if done:
                break
        print()

        # Test environment methods
        print("Testing environment methods...")
        equity_curve = env.get_equity_curve()
        print(f"Equity curve length: {len(equity_curve)}")
        print(f"Final equity: ${equity_curve.iloc[-1]:.2f}")

        metrics = env.get_performance_metrics()
        print(f"Performance metrics: {list(metrics.keys())}")
        print(f"Total return: {metrics.get('total_return', 'N/A'):.4f}")
        print()

        print("Environment test completed successfully!")
        return True

    except Exception as e:
        print(f"Error during environment testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_environment_with_polygon_data()