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
            'williams_window': 14,
            'vol_of_vol': {'enabled': True, 'window': 14, 'vol_window': 14},
            'sma_slope': {'enabled': True, 'window': 20, 'slope_window': 5},
            'obv': {'enabled': True}
        },
        'microstructure': {
            'calculate_spread': True,
            'calculate_microprice': False,  # No bid/ask data
            'calculate_queue_imbalance': False,  # No bid/ask data
            'calculate_order_flow_imbalance': False,  # No bid/ask data
            'calculate_vwap': True,
            'calculate_twap': True,
            'twap_window': 5,
            'calculate_price_impact': True,
            'fvg': {'enabled': True}
        },
        'time': {
            'extract_time_of_day': True,
            'extract_day_of_week': True,
            'extract_session_features': True
        },
        'vix': {'enabled': True, 'path': 'data/external/vix.parquet'},
        'normalization': {
            'method': 'standardize',
            'fit_on_train': True
        },
        # Prefer automated correlation pruning over brittle manual lists
        'feature_selection': { 'correlation_threshold': 0.95 },
        'polygon': {
            'features': {
                'use_vwap_column': False  # SPY data may not have VWAP
            },
            'quality_checks': {
                'enabled': True
            }
        }
    }

    # SMT config from settings if available; default to SPY/QQQ raw files
    try:
        smt_enabled = True
        spy_path = 'data/raw/SPY_1min.parquet'
        qqq_path = 'data/raw/QQQ_1min.parquet'
        from pathlib import Path as _P
        if not (_P(spy_path).exists() and _P(qqq_path).exists()):
            smt_enabled = False
            print("SMT disabled: SPY/QQQ raw parquets not found. Place them under data/raw/.")
        feature_config['smt'] = {
            'enabled': smt_enabled,
            'momentum_span': 5,
            'paths': {'SPY': spy_path, 'QQQ': qqq_path}
        }
    except Exception:
        pass

    # Create feature pipeline
    pipeline = FeaturePipeline(feature_config)

    # Generate features
    print("Generating features...")
    features = pipeline.fit_transform(data)

    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    # Brief feature summary
    cats = {
        'VIX base': any(c.startswith('vix_') for c in features.columns),
        'VIX term-structure': any(c in ('vix_9d_ratio','vix_1m_3m_ratio') for c in features.columns),
        'SMT vs SPY': 'smt_vs_spy' in features.columns,
        'SMT SPY-QQQ': 'smt_spy_qqq' in features.columns,
        'ICT (OR/PDM/disp/FVG)': any(k in features.columns for k in ['dist_orh_bp','dist_orl_bp','dist_pdm_mid','disp_bar','fvg','fvg_density']),
        'VPA (climax/churn/etc)': any(k in features.columns for k in ['climax_vol','churn','churn_z','imbalance_persist','direction_ema','intrabar_vol'])
    }
    print("Feature summary:")
    for k,v in cats.items():
        print(f"  - {k}: {'yes' if v else 'no'}")

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
