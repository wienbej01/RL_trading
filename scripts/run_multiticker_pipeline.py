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
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download step')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip feature generation step')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training step')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip backtesting step')
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
    
    # Initialize feature pipeline
    from src.features.pipeline import FeaturePipeline
    feature_pipeline = FeaturePipeline(config['features'])
    
    # Generate features
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
    train_mask = (data.index >= args.train_start) & (data.index <= args.train_end)
    test_mask = (data.index >= args.test_start) & (data.index <= args.test_end)
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    train_features = features[train_mask]
    test_features = features[test_mask]
    
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
    backtest_results = trainer.backtest(
        model=model,
        data=test_data,
        features=test_features,
        output_dir=Path(args.output_dir) / 'backtest'
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
    
    # Step 3: Train model
    model = None
    trainer = None
    train_data = None
    test_data = None
    train_features = None
    test_features = None
    if not args.skip_training and data is not None and features is not None:
        model, trainer, train_data, test_data, train_features, test_features = train_model(config, args, data, features)
    
    # Step 4: Run backtest
    backtest_results = None
    if not args.skip_backtest and model is not None and trainer is not None:
        backtest_results = run_backtest(config, args, model, trainer, test_data, test_features)
    
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