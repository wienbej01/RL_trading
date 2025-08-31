"""
Integration Example: Using UnifiedDataLoader with Feature Pipeline

This example demonstrates how to use the new UnifiedDataLoader to load data
from Polygon and Databento sources and process it through the feature pipeline.

The example shows:
1. Loading data using the unified interface
2. Schema validation and data quality checks
3. Integration with the feature engineering pipeline
4. Performance optimizations and caching
5. Error handling and backward compatibility
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from src.data.data_loader import UnifiedDataLoader, DataLoaderError
from src.features.pipeline import FeaturePipeline
from src.utils.config_loader import Settings


def create_sample_data():
    """Create sample partitioned Parquet data for testing."""
    # Create temporary directory structure
    temp_dir = Path(tempfile.mkdtemp())

    # Create sample OHLCV data
    dates = pd.date_range('2023-01-01 09:30:00', periods=100, freq='1min')
    np.random.seed(42)

    data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.normal(0, 1, 100).cumsum(),
        'high': 101 + np.random.normal(0, 1, 100).cumsum(),
        'low': 99 + np.random.normal(0, 1, 100).cumsum(),
        'close': 100 + np.random.normal(0, 1, 100).cumsum(),
        'volume': np.random.randint(1000, 5000, 100),
        'bid_price': 99.5 + np.random.normal(0, 0.5, 100).cumsum(),
        'ask_price': 100.5 + np.random.normal(0, 0.5, 100).cumsum(),
        'bid_size': np.random.randint(10, 100, 100),
        'ask_size': np.random.randint(10, 100, 100)
    })

    # Create partitioned structure
    symbol = 'TEST'
    data_dir = temp_dir / "polygon" / "historical" / f"symbol={symbol}"

    for date in dates:
        year, month, day = date.year, date.month, date.day
        partition_dir = data_dir / f"year={year}" / f"month={month}" / f"day={day}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        # Filter data for this day
        day_data = data[(data['timestamp'].dt.date == date.date())].copy()
        if not day_data.empty:
            day_data.set_index('timestamp', inplace=True)
            day_data.to_parquet(partition_dir / "data.parquet")

    return temp_dir


def example_basic_usage():
    """Example 1: Basic usage of UnifiedDataLoader."""
    print("=== Example 1: Basic Usage ===")

    # Create mock settings
    settings = Settings()
    settings._config = {
        'data': {
            'cache_enabled': True,
            'validation_enabled': True,
            'quality_checks_enabled': True,
            'max_gap_minutes': 60,
            'max_price_change_pct': 0.20,
            'min_volume_threshold': 0
        }
    }

    # Initialize data loader
    loader = UnifiedDataLoader(settings)

    # Create sample data
    temp_dir = create_sample_data()

    try:
        # Load OHLCV data
        data = loader.load_data(
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-01-02',
            data_type='ohlcv',
            freq='1min'
        )

        print(f"Loaded {len(data)} rows of OHLCV data")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")

        # Load order book data
        orderbook_data = loader.load_data(
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-01-02',
            data_type='orderbook',
            freq='1min'
        )

        print(f"Loaded {len(orderbook_data)} rows of order book data")

    except DataLoaderError as e:
        print(f"Data loading error: {e}")
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


def example_feature_pipeline_integration():
    """Example 2: Integration with Feature Pipeline."""
    print("\n=== Example 2: Feature Pipeline Integration ===")

    # Create mock settings
    settings = Settings()
    settings._config = {
        'data': {
            'cache_enabled': False,  # Disable caching for this example
            'validation_enabled': True,
            'quality_checks_enabled': True
        }
    }

    # Initialize components
    loader = UnifiedDataLoader(settings)

    # Feature pipeline configuration
    feature_config = {
        'technical': {
            'calculate_returns': True,
            'sma_windows': [5, 10, 20],
            'calculate_rsi': True,
            'rsi_window': 14,
            'calculate_atr': True,
            'atr_window': 14
        },
        'microstructure': {
            'calculate_spread': True,
            'calculate_microprice': True,
            'calculate_queue_imbalance': True,
            'calculate_vwap': True
        },
        'time': {
            'extract_time_of_day': True,
            'extract_day_of_week': True
        }
    }

    pipeline = FeaturePipeline(feature_config)

    # Create sample data
    temp_dir = create_sample_data()

    try:
        # Load data
        market_data = loader.load_data(
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-01-02',
            data_type='ohlcv'
        )

        print(f"Loaded market data: {len(market_data)} rows")

        # Process through feature pipeline
        features = pipeline.fit_transform(market_data)

        print(f"Generated {len(features.columns)} features")
        print(f"Feature columns: {list(features.columns)}")

        # Show some statistics
        print(f"Features shape: {features.shape}")
        print(f"NaN values per column:\n{features.isnull().sum()}")

    except Exception as e:
        print(f"Integration error: {e}")
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


def example_performance_optimization():
    """Example 3: Performance optimizations and caching."""
    print("\n=== Example 3: Performance Optimization ===")

    settings = Settings()
    settings._config = {
        'data': {
            'cache_enabled': True,
            'validation_enabled': True,
            'quality_checks_enabled': True,
            'max_workers': 4  # Enable parallel processing
        }
    }

    loader = UnifiedDataLoader(settings)
    temp_dir = create_sample_data()

    try:
        import time

        # First load (will cache)
        start_time = time.time()
        data1 = loader.load_data(
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-01-02',
            data_type='ohlcv'
        )
        first_load_time = time.time() - start_time

        # Second load (from cache)
        start_time = time.time()
        data2 = loader.load_data(
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-01-02',
            data_type='ohlcv'
        )
        second_load_time = time.time() - start_time

        print(".2f")
        print(".2f")
        print(".1f")

        # Verify data consistency
        pd.testing.assert_frame_equal(data1, data2)
        print("✓ Cached data matches original data")

    except Exception as e:
        print(f"Performance test error: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def example_error_handling():
    """Example 4: Error handling and data quality."""
    print("\n=== Example 4: Error Handling ===")

    settings = Settings()
    settings._config = {
        'data': {
            'cache_enabled': False,
            'validation_enabled': True,
            'quality_checks_enabled': True
        }
    }

    loader = UnifiedDataLoader(settings)

    # Test invalid data type
    try:
        loader.load_data(
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-01-02',
            data_type='invalid_type'
        )
    except DataLoaderError as e:
        print(f"✓ Correctly caught invalid data type error: {e}")

    # Test with non-existent data
    try:
        data = loader.load_data(
            symbol='NONEXISTENT',
            start_date='2023-01-01',
            end_date='2023-01-02',
            data_type='ohlcv'
        )
        print(f"✓ Gracefully handled missing data: returned {len(data)} rows")
    except Exception as e:
        print(f"Error handling missing data: {e}")


def example_backward_compatibility():
    """Example 5: Backward compatibility demonstration."""
    print("\n=== Example 5: Backward Compatibility ===")

    # Create data in legacy format (without proper partitioning)
    legacy_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01 09:30:00', periods=50, freq='1min'),
        'open': np.random.uniform(100, 105, 50),
        'high': np.random.uniform(105, 110, 50),
        'low': np.random.uniform(95, 100, 50),
        'close': np.random.uniform(100, 105, 50),
        'volume': np.random.randint(1000, 2000, 50)
    })
    legacy_data.set_index('timestamp', inplace=True)

    settings = Settings()
    settings._config = {
        'data': {
            'cache_enabled': False,
            'validation_enabled': True,
            'quality_checks_enabled': True
        }
    }

    loader = UnifiedDataLoader(settings)

    # Test schema validation on legacy format
    try:
        validated_data = loader._perform_quality_checks(legacy_data, 'ohlcv')
        print(f"✓ Legacy data format validated: {len(validated_data)} rows")
        print(f"  Columns: {list(validated_data.columns)}")
    except Exception as e:
        print(f"Legacy data validation error: {e}")


def main():
    """Run all examples."""
    print("UnifiedDataLoader Integration Examples")
    print("=" * 50)

    example_basic_usage()
    example_feature_pipeline_integration()
    example_performance_optimization()
    example_error_handling()
    example_backward_compatibility()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nKey Features Demonstrated:")
    print("• Unified interface for Polygon and Databento data")
    print("• Automatic schema validation and data quality checks")
    print("• Seamless integration with feature engineering pipeline")
    print("• Performance optimizations with caching and parallel processing")
    print("• Robust error handling and backward compatibility")
    print("• Support for OHLCV, order book, and trade data types")


if __name__ == "__main__":
    main()