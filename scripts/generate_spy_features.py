#!/usr/bin/env python3
"""
Generate features for SPY data using the feature engineering pipeline.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.pipeline import FeaturePipeline
from src.utils.config_loader import Settings

def generate_spy_features(data_path: str, output_path: str, config_path: str = "configs/settings.yaml"):
    """
    Generate features for SPY data.

    Args:
        data_path: Path to SPY data parquet file
        output_path: Path to save features
        config_path: Path to configuration file
    """
    # Load configuration
    settings = Settings.from_paths(config_path)

    # Load data
    print(f"Loading data from {data_path}")
    data = pd.read_parquet(data_path)
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Create feature pipeline configuration
    feature_config = {
        'data_source': 'polygon',
        'technical': {
            'calculate_returns': True,
            'calculate_log_returns': True,
            'sma_windows': [5, 10, 20, 50],
            'ema_windows': [5, 10, 20, 50],
            'calculate_atr': True,
            'calculate_rsi': True,
            'rsi_window': 14,
            'calculate_macd': True,
            'macd_config': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'calculate_bollinger_bands': True,
            'bollinger_config': {
                'window': 20,
                'num_std': 2
            },
            'calculate_stochastic': True,
            'stochastic_config': {
                'k_period': 14,
                'd_period': 3
            },
            'calculate_williams_r': True,
            'williams_window': 14
        },
        'microstructure': {
            'calculate_spread': True,
            'calculate_microprice': False,  # No bid/ask data
            'calculate_queue_imbalance': False,  # No bid/ask data
            'calculate_order_flow_imbalance': False,  # No bid/ask data
            'calculate_vwap': True,
            'calculate_twap': True,
            'twap_window': 5,
            'calculate_price_impact': True
        },
        'time': {
            'extract_time_of_day': True,
            'extract_day_of_week': True,
            'extract_session_features': True
        },
        'normalization': {
            'method': 'standardize',
            'fit_on_train': True
        },
        'polygon': {
            'features': {
                'use_vwap_column': False  # SPY data may not have VWAP
            },
            'quality_checks': {
                'enabled': True
            }
        }
    }

    # Create feature pipeline
    pipeline = FeaturePipeline(feature_config)

    # Generate features
    print("Generating features...")
    features = pipeline.fit_transform(data)

    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save features
    features.to_parquet(output_path)
    print(f"Features saved to {output_path}")

    return features

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate features for SPY data")
    parser.add_argument("--data", required=True, help="Path to SPY data parquet file")
    parser.add_argument("--output", required=True, help="Path to save features")
    parser.add_argument("--config", default="configs/settings.yaml", help="Path to configuration file")

    args = parser.parse_args()

    generate_spy_features(args.data, args.output, args.config)