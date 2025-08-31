#!/usr/bin/env python3
"""
Validate Collected Polygon Data with RL Environment

This script validates the collected Polygon data by testing it with the RL environment.
It loads the collected data, runs basic validation checks, and performs a short
backtest simulation to ensure everything works correctly.

Usage:
    python scripts/validate_collected_data.py --data-dir data/polygon/historical --symbols SPY
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import UnifiedDataLoader
from src.features.pipeline import FeaturePipeline
from src.sim.env_intraday_rl import IntradayRLEnv
from src.utils.config_loader import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_and_validate_data(data_dir: Path, symbols: List[str]) -> Dict[str, Any]:
    """
    Load and validate collected data.

    Args:
        data_dir: Data directory path
        symbols: List of symbols to validate

    Returns:
        Validation results
    """
    results = {
        'total_symbols': len(symbols),
        'valid_symbols': 0,
        'total_records': 0,
        'data_quality': {},
        'symbol_details': {}
    }

    # Initialize data loader with config
    settings = Settings.from_paths('configs/settings.yaml')
    loader = UnifiedDataLoader(settings, data_source='polygon')

    for symbol in tqdm(symbols, desc="Validating symbols"):
        symbol_results = {
            'records': 0,
            'date_range': None,
            'data_quality': {},
            'errors': []
        }

        try:
            # Get data info
            info = loader.get_data_info(symbol)
            if info['total_files'] == 0:
                symbol_results['errors'].append("No data files found")
                continue

            # Load sample data
            start_date = info['date_range']['start'] if info['date_range'] else None
            end_date = info['date_range']['end'] if info['date_range'] else None

            if start_date and end_date:
                # Load a week's worth of data for validation
                sample_start = max(start_date, end_date - pd.Timedelta(days=7))
                sample_end = end_date

                df = loader.load_data(
                    symbol=symbol,
                    start_date=sample_start.strftime('%Y-%m-%d'),
                    end_date=sample_end.strftime('%Y-%m-%d'),
                    data_type='ohlcv',
                    freq='1min'
                )

                if not df.empty:
                    symbol_results['records'] = len(df)
                    symbol_results['date_range'] = (df.index.min(), df.index.max())

                    # Data quality checks
                    symbol_results['data_quality'] = {
                        'missing_values': df.isnull().sum().sum(),
                        'duplicate_timestamps': df.index.duplicated().sum(),
                        'negative_prices': (df[['open', 'high', 'low', 'close']] < 0).sum().sum(),
                        'zero_volume': (df['volume'] == 0).sum(),
                        'price_anomalies': check_price_anomalies(df)
                    }

                    results['total_records'] += len(df)
                    results['valid_symbols'] += 1
                else:
                    symbol_results['errors'].append("No data loaded")
            else:
                symbol_results['errors'].append("No date range available")

        except Exception as e:
            symbol_results['errors'].append(str(e))
            logger.error(f"Error validating {symbol}: {e}")

        results['symbol_details'][symbol] = symbol_results

    return results


def check_price_anomalies(df: pd.DataFrame) -> int:
    """Check for price anomalies in the data."""
    anomalies = 0

    if 'close' in df.columns:
        # Check for extreme price changes (>50% in a minute)
        returns = df['close'].pct_change(fill_method=None)
        anomalies += (returns.abs() > 0.5).sum()

        # Check for OHLC consistency
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        anomalies += invalid_ohlc.sum()

    return int(anomalies)


def test_rl_environment(data_dir: Path, symbol: str) -> Dict[str, Any]:
    """
    Test the RL environment with collected data.

    Args:
        data_dir: Data directory path
        symbol: Symbol to test

    Returns:
        Test results
    """
    results = {
        'environment_test': 'failed',
        'steps_tested': 0,
        'avg_reward': 0.0,
        'errors': []
    }

    try:
        # Initialize data loader with config
        settings = Settings.from_paths('configs/settings.yaml')
        loader = UnifiedDataLoader(settings, data_source='polygon')

        # Load data
        df = loader.load_data(
            symbol=symbol,
            start_date='2024-01-01',
            end_date='2024-12-31',
            data_type='ohlcv',
            freq='1min'
        )

        if df.empty:
            results['errors'].append("No data available for testing")
            return results

        # Take a sample for testing (first 1000 rows)
        test_df = df.head(1000)

        # Initialize feature pipeline
        config = {
            'data_source': 'polygon',
            'technical': {
                'calculate_returns': True,
                'sma_windows': [5, 10, 20],
                'calculate_atr': True,
                'calculate_rsi': True,
                'rsi_window': 14
            }
        }

        pipeline = FeaturePipeline(config)
        features = pipeline.fit_transform(test_df)

        # Initialize RL environment
        env = IntradayRLEnv(
            ohlcv=test_df,
            features=features,
            cash=100000.0,
            point_value=5.0
        )

        # Test environment
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        # Run 100 steps with random actions
        for _ in range(100):
            action = np.random.randint(0, 3)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        results['environment_test'] = 'passed'
        results['steps_tested'] = steps
        results['avg_reward'] = total_reward / steps if steps > 0 else 0.0

        logger.info(f"RL environment test passed: {steps} steps, avg reward: {results['avg_reward']:.4f}")

    except Exception as e:
        results['errors'].append(str(e))
        logger.error(f"RL environment test failed: {e}")

    return results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate collected Polygon data")
    parser.add_argument('--data-dir', type=str, default='data/polygon/historical',
                       help='Data directory path')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to validate')
    parser.add_argument('--test-rl', action='store_true', help='Test RL environment with data')
    parser.add_argument('--test-symbol', type=str, help='Symbol to use for RL testing')

    args = parser.parse_args()

    # Determine symbols to validate
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        # Auto-detect symbols from data directory
        data_dir = Path(args.data_dir)
        if data_dir.exists():
            symbols = []
            for item in data_dir.glob("symbol=*"):
                if item.is_dir():
                    symbol = item.name.split('=')[1]
                    symbols.append(symbol)
        else:
            print(f"Data directory {data_dir} does not exist")
            return

    if not symbols:
        print("No symbols found to validate")
        return

    print(f"Validating data for {len(symbols)} symbols: {symbols}")
    print("="*60)

    # Validate data
    validation_results = load_and_validate_data(Path(args.data_dir), symbols)

    # Print validation summary
    print("\nVALIDATION SUMMARY")
    print("="*60)
    print(f"Symbols validated: {validation_results['valid_symbols']}/{validation_results['total_symbols']}")
    print(f"Total records: {validation_results['total_records']:,}")

    for symbol, details in validation_results['symbol_details'].items():
        print(f"\n{symbol}:")
        if details['records'] > 0:
            print(f"  Records: {details['records']:,}")
            if details['date_range']:
                start, end = details['date_range']
                print(f"  Date range: {start} to {end}")
            print(f"  Data quality: {details['data_quality']}")
        else:
            print(f"  Errors: {details['errors']}")

    # Test RL environment if requested
    if args.test_rl:
        test_symbol = args.test_symbol or symbols[0]
        print(f"\nTesting RL environment with {test_symbol}...")
        rl_results = test_rl_environment(Path(args.data_dir), test_symbol)

        print("\nRL ENVIRONMENT TEST")
        print("="*60)
        print(f"Test result: {rl_results['environment_test']}")
        if rl_results['environment_test'] == 'passed':
            print(f"Steps tested: {rl_results['steps_tested']}")
            print(f"Average reward: {rl_results['avg_reward']:.4f}")
        else:
            print(f"Errors: {rl_results['errors']}")

    print("\nValidation completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()