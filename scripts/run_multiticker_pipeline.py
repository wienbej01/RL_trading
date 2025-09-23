#!/usr/bin/env python3
"""
Multi-Ticker RL Trading System Pipeline

This script runs the complete pipeline for the Multi-Ticker RL Trading System:
1. Data download for relevant tickers
2. Feature generation
3. Model training
4. Backtesting

Usage:
    python scripts/run_multiticker_pipeline.py --config configs/optimized_settings.yaml
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.utils.logging import get_logger
from src.optimization.memory_optimizer import MemoryOptimizer
from src.optimization.computation_optimizer import ComputationOptimizer
from src.rl.multiticker_trainer import MultiTickerRLTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multiticker_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Multi-Ticker RL Trading System Pipeline')
    parser.add_argument('--config', type=str, default='configs/settings.yaml',
                        help='Path to configuration file')
    parser.add_argument('--train-start', type=str, default='2020-09-01',
                        help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, default='2024-12-31',
                        help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', type=str, default='2025-01-01',
                        help='Testing start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, default='2025-06-30',
                        help='Testing end date (YYYY-MM-DD)')
    parser.add_argument('--tickers', type=str, nargs='+', 
                        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V'],
                        help='List of tickers to trade')
    parser.add_argument('--output-dir', type=str, default='results/multiticker_pipeline',
                        help='Output directory for results')
    # Backtest behavior
    parser.add_argument('--strict-test-window', action='store_true',
                        help='Do not fall back to a different window if the test split is empty')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download step')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip feature generation step')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training step')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip backtesting step')
    parser.add_argument('--portfolio-env', action='store_true',
                        help='Force portfolio environment for training/backtest')
    # Portfolio env tuning (affects backtest behavior)
    parser.add_argument('--min-hold-minutes', type=int, default=None,
                        help='Minimum holding period in minutes')
    parser.add_argument('--max-hold-minutes', type=int, default=None,
                        help='Maximum holding period in minutes')
    parser.add_argument('--max-entries-per-day', type=int, default=None,
                        help='Maximum entries/flips per day')
    parser.add_argument('--position-holding-penalty', type=float, default=None,
                        help='Penalty per open position per bar (encourages exits)')
    parser.add_argument('--test-tickers', type=str, nargs='+', default=None,
                        help='Subset of tickers to trade during backtest (if provided)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to a saved SB3 model to load for backtest (skip training)')
    return parser.parse_args()


def download_data(config, args):
    """Download data for the specified tickers and date range."""
    logger.info("Starting data download")
    
    # Update config with date ranges and tickers
    config['data']['start_date'] = args.train_start
    config['data']['end_date'] = args.test_end
    config['data']['tickers'] = args.tickers
    
    # Initialize data loader
    from src.data.data_loader import UnifiedDataLoader
    data_loader = UnifiedDataLoader(config_path=args.config)
    
    # Download data for each ticker
    all_data = []
    for ticker in args.tickers:
        logger.info(f"Downloading data for {ticker}")
        ticker_data = data_loader.load_ohlcv(
            symbol=ticker,
            start=pd.Timestamp(args.train_start),
            end=pd.Timestamp(args.test_end)
        )
        if not ticker_data.empty:
            ticker_data['ticker'] = ticker
            all_data.append(ticker_data)
        else:
            logger.warning(f"No data found for {ticker}")
    
    if not all_data:
        # If no data found, create synthetic data for demonstration
        logger.warning("No data found for any ticker, creating synthetic data for demonstration")
        dates = pd.date_range(start=args.train_start, end=args.test_end, freq='D')
        for ticker in args.tickers:
            ticker_data = pd.DataFrame({
                'open': np.random.uniform(100, 200, len(dates)),
                'high': np.random.uniform(100, 200, len(dates)),
                'low': np.random.uniform(100, 200, len(dates)),
                'close': np.random.uniform(100, 200, len(dates)),
                'volume': np.random.uniform(1000000, 10000000, len(dates)),
                'vwap': np.random.uniform(100, 200, len(dates)),
                'ticker': ticker
            }, index=dates)
            all_data.append(ticker_data)
    
    data = pd.concat(all_data, axis=0)
    
    # Save data
    data_path = Path(args.output_dir) / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    data_file = data_path / f"multiticker_data_{args.train_start}_to_{args.test_end}.parquet"
    data.to_parquet(data_file)
    
    logger.info(f"Data downloaded and saved to {data_file}")
    logger.info(f"Data shape: {data.shape}")
    
    return data


def generate_features(config, args, data):
    """Generate features from the downloaded data."""
    logger.info("Starting feature generation")
    # Data hygiene: strict dedup by (timestamp,ticker), small ffill within session, drop tiny islands
    def _preclean(df: pd.DataFrame) -> pd.DataFrame:
        base = df
        if 'ticker' in base.columns:
            dedup = base.reset_index().drop_duplicates(subset=['index','ticker'], keep='last').set_index('index')
        else:
            dedup = base[~base.index.duplicated(keep='last')]
        def _ff(group: pd.DataFrame) -> pd.DataFrame:
            return group.sort_index().ffill(limit=2)
        if 'ticker' in dedup.columns:
            dedup = dedup.groupby('ticker', group_keys=False).apply(_ff)
            # drop islands < 5 bars
            keep_idx = []
            for t, g in dedup.groupby('ticker'):
                if len(g) >= 5:
                    keep_idx.append(g.index)
            if keep_idx:
                import numpy as _np
                mask = dedup.index.isin(_np.concatenate(keep_idx))
                dedup = dedup.loc[mask]
        else:
            dedup = _ff(dedup)
        return dedup
    try:
        data = _preclean(data)
    except Exception as e:
        logger.warning(f"Pre-clean failed (continuing): {e}")
    
    # Initialize feature pipeline
    from src.features.pipeline import FeaturePipeline
    feature_pipeline = FeaturePipeline(config['features'])
    
    # Generate features
    # Preserve ticker column so FeaturePipeline can compute per‑ticker features correctly.
    # The pipeline will group by 'ticker' when present and attach it back to the output.
    features = feature_pipeline.fit_transform(data)
    
    # Save features
    features_path = Path(args.output_dir) / 'features'
    features_path.mkdir(parents=True, exist_ok=True)
    features_file = features_path / f"multiticker_features_{args.train_start}_to_{args.test_end}.parquet"
    features.to_parquet(features_file)
    
    logger.info(f"Features generated and saved to {features_file}")
    logger.info(f"Features shape: {features.shape}")
    
    return features


def train_model(config, args, data, features):
    """Train the RL model."""
    logger.info("Starting model training")
    
    # Split data into training and testing sets
    # Build masks on each frame's own index to handle cleaning/row drops
    train_mask_d = (data.index >= args.train_start) & (data.index <= args.train_end)
    test_mask_d = (data.index >= args.test_start) & (data.index <= args.test_end)
    train_mask_f = (features.index >= args.train_start) & (features.index <= args.train_end)
    test_mask_f = (features.index >= args.test_start) & (features.index <= args.test_end)

    train_data = data[train_mask_d]
    test_data = data[test_mask_d]
    train_features = features[train_mask_f]
    test_features = features[test_mask_f]
    
    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Testing data shape: {test_data.shape}")
    
    # Initialize trainer
    from src.rl.multiticker_trainer import MultiTickerRLTrainer
    trainer = MultiTickerRLTrainer(config)
    
    # Train model
    model = trainer.train(
        data=train_data,
        features=train_features,
        output_dir=Path(args.output_dir) / 'models'
    )
    
    logger.info("Model training completed")
    
    return model, trainer, train_data, test_data, train_features, test_features


def run_backtest(config, args, model, trainer, test_data, test_features):
    """Run backtesting on the test data."""
    logger.info("Starting backtesting")
    
    # Run backtest
    # Determine allowed test tickers: prefer provided list, else cap to 3 tickers if desired
    allowed = args.test_tickers if args.test_tickers else None
    if allowed is None and len(args.tickers) >= 3:
        allowed = args.tickers[:3]

    backtest_results = trainer.backtest(
        model=model,
        data=test_data,
        features=test_features,
        output_dir=Path(args.output_dir) / 'backtest',
        allowed_tickers=allowed
    )
    
    # Save backtest results
    results_path = Path(args.output_dir) / 'backtest'
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    results_file = results_path / f"backtest_results_{args.test_start}_to_{args.test_end}.json"
    with open(results_file, 'w') as f:
        json.dump(backtest_results, f, indent=2, default=str)
    
    # Generate and save plots
    if 'generate_backtest_plots' in dir(trainer):
        trainer.generate_backtest_plots(results_path)
    
    logger.info(f"Backtest results saved to {results_file}")
    
    # Print summary
    if 'get_backtest_summary' in dir(trainer):
        summary = trainer.get_backtest_summary()
        logger.info("Backtest Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
    
    return backtest_results


def main():
    """Main function to run the pipeline."""
    args = parse_args()
    
    logger.info("Starting Multi-Ticker RL Trading System Pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Training period: {args.train_start} to {args.train_end}")
    logger.info(f"Testing period: {args.test_start} to {args.test_end}")
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    # Inject portfolio-env flag into config for trainer
    if args.portfolio_env:
        config.setdefault('env', {}).setdefault('portfolio', {})['force'] = True
    # Apply portfolio env tuning overrides if provided
    if args.min_hold_minutes is not None:
        config.setdefault('env', {}).setdefault('portfolio', {})['min_hold_minutes'] = int(args.min_hold_minutes)
    if args.max_hold_minutes is not None:
        config.setdefault('env', {}).setdefault('portfolio', {})['max_hold_minutes'] = int(args.max_hold_minutes)
    if args.max_entries_per_day is not None:
        config.setdefault('env', {}).setdefault('portfolio', {})['max_entries_per_day'] = int(args.max_entries_per_day)
    if args.position_holding_penalty is not None:
        config.setdefault('env', {}).setdefault('portfolio', {})['position_holding_penalty'] = float(args.position_holding_penalty)
    
    # Apply optimizations
    config = ComputationOptimizer.apply_all_optimizations(config)
    
    # Log memory usage
    MemoryOptimizer.log_memory_usage("at start")
    
    # Step 1: Download data
    data = None
    if not args.skip_download:
        data = download_data(config, args)
    else:
        # Load existing data
        data_path = Path(args.output_dir) / 'data'
        data_file = data_path / f"multiticker_data_{args.train_start}_to_{args.test_end}.parquet"
        if data_file.exists():
            logger.info(f"Loading existing data from {data_file}")
            data = pd.read_parquet(data_file)
        else:
            logger.error(f"Data file not found: {data_file}")
            return
    
    # Step 2: Generate features
    features = None
    if not args.skip_features and data is not None:
        features = generate_features(config, args, data)
    else:
        # Load existing features
        features_path = Path(args.output_dir) / 'features'
        features_file = features_path / f"multiticker_features_{args.train_start}_to_{args.test_end}.parquet"
        if features_file.exists():
            logger.info(f"Loading existing features from {features_file}")
            features = pd.read_parquet(features_file)
        else:
            logger.error(f"Features file not found: {features_file}")
            return
    
    # Train/test split (make available even if skipping training)
    # Build masks on each frame to handle cleaning/dedup differences
    train_mask_d = (data.index >= args.train_start) & (data.index <= args.train_end)
    test_mask_d = (data.index >= args.test_start) & (data.index <= args.test_end)
    train_mask_f = (features.index >= args.train_start) & (features.index <= args.train_end)
    test_mask_f = (features.index >= args.test_start) & (features.index <= args.test_end)

    train_data = data[train_mask_d]
    test_data = data[test_mask_d]
    train_features = features[train_mask_f]
    test_features = features[test_mask_f]

    # Guard: empty test split would make portfolio env construction fail
    if test_data.empty:
        logger.warning("Test split is empty for requested window; falling back to last available slice.")
        if args.strict_test_window:
            logger.error("Strict test window enforced and test split is empty. Aborting backtest.")
            return
        try:
            # Use the last 2 calendar days of available data as a minimal backtest window
            last_ts = data.index.max()
            fallback_start = (last_ts - pd.Timedelta(days=2)).normalize()
            fallback_mask_d = (data.index >= fallback_start) & (data.index <= last_ts)
            fallback_mask_f = (features.index >= fallback_start) & (features.index <= last_ts)
            fb_data = data[fallback_mask_d]
            fb_feat = features[fallback_mask_f]
            if fb_data.empty:
                # As an extra fallback, take the last 1,000 rows if present
                fb_data = data.tail(1000)
                fb_feat = features.loc[fb_data.index] if not fb_data.empty else fb_data
            if not fb_data.empty:
                test_data = fb_data
                test_features = fb_feat
                # Update args to reflect the actual backtest window for filenames/logging
                args.test_start = str(pd.to_datetime(test_data.index.min()).date())
                args.test_end = str(pd.to_datetime(test_data.index.max()).date())
                logger.info(f"Using fallback backtest window: {args.test_start} to {args.test_end} ({len(test_data)} rows)")
            else:
                logger.error("No data available at all for backtesting after fallback attempts; skipping backtest.")
                args.skip_backtest = True
        except Exception as e:
            logger.error(f"Failed to construct fallback backtest window: {e}")
            args.skip_backtest = True

    # Step 3: Train model (unless loading)
    model = None
    trainer = MultiTickerRLTrainer(config)
    if args.load_model:
        from sb3_contrib import RecurrentPPO
        logger.info(f"Loading model from {args.load_model}")
        try:
            model = RecurrentPPO.load(args.load_model)
            # Hint to trainer about training tickers to preserve obs shape
            try:
                trainer._train_tickers = sorted(list(pd.unique(train_data['ticker'])))
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
    elif not args.skip_training:
        model, trainer, _, _, _, _ = train_model(config, args, data, features)
    
    # Step 4: Run backtest
    backtest_results = None
    if not args.skip_backtest and model is not None and trainer is not None:
        try:
            backtest_results = run_backtest(config, args, model, trainer, test_data, test_features)
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            # Provide a clear hint for the common cause (empty/unaligned test split)
            if test_data.empty:
                logger.error("Backtest received an empty test dataset. Ensure the test window overlaps available data or let the script choose a fallback window.")
            return
    
    # Log memory usage
    MemoryOptimizer.log_memory_usage("at end")
    
    logger.info("Multi-Ticker RL Trading System Pipeline completed successfully")
    
    # Save pipeline configuration
    pipeline_config = {
        'config_file': args.config,
        'train_start': args.train_start,
        'train_end': args.train_end,
        'test_start': args.test_start,
        'test_end': args.test_end,
        'tickers': args.tickers,
        'output_dir': args.output_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    config_file = Path(args.output_dir) / 'pipeline_config.json'
    with open(config_file, 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    
    logger.info(f"Pipeline configuration saved to {config_file}")


if __name__ == "__main__":
    main()
