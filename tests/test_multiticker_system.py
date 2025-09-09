
"""
Multi-ticker RL trading system test suite.

This module provides comprehensive tests for the multi-ticker RL trading system,
including unit tests, integration tests, and performance tests.
"""

import unittest
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import json
import os
from typing import Dict, List, Any, Optional, Tuple

from src.data.multiticker_data_loader import MultiTickerDataLoader
from src.features.multiticker_pipeline import MultiTickerFeaturePipeline
from src.sim.multiticker_env import MultiTickerIntradayRLEnv
from src.rl.multiticker_trainer import MultiTickerRLTrainer
from src.rl.multiticker_policy import MultiTickerPPOLSTMPolicy
from src.evaluation.multiticker_evaluator import MultiTickerEvaluator
from src.monitoring.multiticker_monitor import MultiTickerMonitor
from src.monitoring.alert_system import AlertSystem
from src.evaluation.backtest_reporter import BacktestReporter
from src.utils.config_loader import load_config


class TestMultiTickerDataLoader(unittest.TestCase):
    """Test cases for MultiTickerDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'data': {
                'tickers': ['AAPL', 'MSFT', 'GOOGL'],
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'data_source': 'polygon',
                'cache_dir': '/tmp/test_cache'
            },
            'paths': {
                'data_root': '/tmp/test_data',
                'cache_dir': '/tmp/test_cache'
            }
        }
        
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.config['paths']['data_root'] = self.temp_dir
        self.config['paths']['cache_dir'] = os.path.join(self.temp_dir, 'cache')
        os.makedirs(self.config['paths']['cache_dir'], exist_ok=True)
        
        # Create sample data
        self._create_sample_data()
        
        # Initialize data loader
        self.data_loader = MultiTickerDataLoader(self.config)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def _create_sample_data(self):
        """Create sample data for testing."""
        tickers = self.config['data']['tickers']
        start_date = pd.to_datetime(self.config['data']['start_date'])
        end_date = pd.to_datetime(self.config['data']['end_date'])
        
        # Create date range
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Create sample data for each ticker
        for ticker in tickers:
            # Create OHLCV data
            np.random.seed(hash(ticker) % 1000)  # Consistent random data per ticker
            
            close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
            open_prices = close_prices * (1 + np.random.randn(len(dates)) * 0.005)
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
            volumes = np.random.randint(1000000, 10000000, len(dates))
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
            
            # Save to CSV
            ticker_dir = os.path.join(self.temp_dir, 'polygon', 'historical', ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            df.to_csv(os.path.join(ticker_dir, f'{ticker}_daily.csv'), index=False)
            
    def test_load_ticker_list(self):
        """Test loading ticker list."""
        tickers = self.data_loader.load_ticker_list()
        self.assertEqual(tickers, self.config['data']['tickers'])
        
    def test_load_data_for_ticker(self):
        """Test loading data for a single ticker."""
        ticker = self.config['data']['tickers'][0]
        data = self.data_loader.load_data_for_ticker(ticker)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('timestamp', data.columns)
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns)
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        
    def test_load_data_for_all_tickers(self):
        """Test loading data for all tickers."""
        data = self.data_loader.load_data_for_all_tickers()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), len(self.config['data']['tickers']))
        
        for ticker in self.config['data']['tickers']:
            self.assertIn(ticker, data)
            self.assertIsInstance(data[ticker], pd.DataFrame)
            
    def test_load_data_for_date_range(self):
        """Test loading data for a specific date range."""
        start_date = '2023-06-01'
        end_date = '2023-06-30'
        
        data = self.data_loader.load_data_for_date_range(start_date, end_date)
        
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), len(self.config['data']['tickers']))
        
        for ticker in self.config['data']['tickers']:
            self.assertIn(ticker, data)
            self.assertIsInstance(data[ticker], pd.DataFrame)
            
            # Check date range
            min_date = data[ticker]['timestamp'].min()
            max_date = data[ticker]['timestamp'].max()
            
            self.assertGreaterEqual(min_date, pd.to_datetime(start_date))
            self.assertLessEqual(max_date, pd.to_datetime(end_date))
            
    def test_get_data_availability(self):
        """Test getting data availability."""
        availability = self.data_loader.get_data_availability()
        
        self.assertIsInstance(availability, dict)
        self.assertEqual(len(availability), len(self.config['data']['tickers']))
        
        for ticker in self.config['data']['tickers']:
            self.assertIn(ticker, availability)
            self.assertIn('start_date', availability[ticker])
            self.assertIn('end_date', availability[ticker])
            self.assertIn('data_points', availability[ticker])
            
    def test_cache_data(self):
        """Test caching data."""
        ticker = self.config['data']['tickers'][0]
        data = self.data_loader.load_data_for_ticker(ticker)
        
        # Cache data
        self.data_loader.cache_data(ticker, data)
        
        # Check if cache file exists
        cache_file = os.path.join(self.config['paths']['cache_dir'], f'{ticker}.pkl')
        self.assertTrue(os.path.exists(cache_file))
        
        # Load cached data
        cached_data = self.data_loader.load_cached_data(ticker)
        
        # Check if cached data matches original data
        pd.testing.assert_frame_equal(data, cached_data)


class TestMultiTickerFeaturePipeline(unittest.TestCase):
    """Test cases for MultiTickerFeaturePipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'features': {
                'technical': {
                    'sma_windows': [5, 10, 20],
                    'ema_windows': [5, 10, 20],
                    'calculate_rsi': True,
                    'rsi_window': 14,
                    'calculate_macd': True,
                    'calculate_bollinger_bands': True,
                    'calculate_atr': True,
                    'calculate_returns': True,
                    'calculate_log_returns': True
                },
                'microstructure': {
                    'calculate_spread': True,
                    'calculate_vwap': True,
                    'calculate_twap': True,
                    'calculate_price_impact': True
                },
                'time': {
                    'extract_time_of_day': True,
                    'extract_day_of_week': True,
                    'extract_session_features': True
                },
                'normalization': {
                    'method': 'standard',
                    'fit_per_ticker': True
                }
            }
        }
        
        # Create sample data
        self._create_sample_data()
        
        # Initialize feature pipeline
        self.feature_pipeline = MultiTickerFeaturePipeline(self.config)
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def _create_sample_data(self):
        """Create sample data for testing."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        
        self.sample_data = {}
        
        for ticker in tickers:
            # Create OHLCV data
            np.random.seed(hash(ticker) % 1000)  # Consistent random data per ticker
            
            close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
            open_prices = close_prices * (1 + np.random.randn(len(dates)) * 0.005)
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
            volumes = np.random.randint(1000000, 10000000, len(dates))
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
            
            self.sample_data[ticker] = df
            
    def test_fit_transform(self):
        """Test fitting and transforming data."""
        features = self.feature_pipeline.fit_transform(self.sample_data)
        
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), len(self.sample_data))
        
        for ticker in self.sample_data:
            self.assertIn(ticker, features)
            self.assertIsInstance(features[ticker], pd.DataFrame)
            
            # Check if features were added
            self.assertGreater(len(features[ticker].columns), len(self.sample_data[ticker].columns))
            
    def test_transform(self):
        """Test transforming data."""
        # First fit the pipeline
        self.feature_pipeline.fit(self.sample_data)
        
        # Then transform
        features = self.feature_pipeline.transform(self.sample_data)
        
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), len(self.sample_data))
        
        for ticker in self.sample_data:
            self.assertIn(ticker, features)
            self.assertIsInstance(features[ticker], pd.DataFrame)
            
    def test_get_feature_names(self):
        """Test getting feature names."""
        # First fit and transform
        self.feature_pipeline.fit(self.sample_data)
        features = self.feature_pipeline.transform(self.sample_data)
        
        # Get feature names
        feature_names = self.feature_pipeline.get_feature_names()
        
        self.assertIsInstance(feature_names, dict)
        self.assertEqual(len(feature_names), len(self.sample_data))
        
        for ticker in self.sample_data:
            self.assertIn(ticker, feature_names)
            self.assertIsInstance(feature_names[ticker], list)
            
            # Check if feature names match columns
            self.assertEqual(set(feature_names[ticker]), set(features[ticker].columns))
            
    def test_get_normalization_params(self):
        """Test getting normalization parameters."""
        # First fit the pipeline
        self.feature_pipeline.fit(self.sample_data)
        
        # Get normalization parameters
        norm_params = self.feature_pipeline.get_normalization_params()
        
        self.assertIsInstance(norm_params, dict)
        self.assertEqual(len(norm_params), len(self.sample_data))
        
        for ticker in self.sample_data:
            self.assertIn(ticker, norm_params)
            self.assertIn('mean', norm_params[ticker])
            self.assertIn('std', norm_params[ticker])
            
    def test_save_load_pipeline(self):
        """Test saving and loading pipeline."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # First fit the pipeline
            self.feature_pipeline.fit(self.sample_data)
            
            # Save pipeline
            pipeline_path = os.path.join(temp_dir, 'feature_pipeline.pkl')
            self.feature_pipeline.save_pipeline(pipeline_path)
            
            # Load pipeline
            loaded_pipeline = MultiTickerFeaturePipeline.load_pipeline(pipeline_path)
            
            # Transform with loaded pipeline
            features = loaded_pipeline.transform(self.sample_data)
            
            # Check if features are correct
            self.assertIsInstance(features, dict)
            self.assertEqual(len(features), len(self.sample_data))
            
        finally:
            shutil.rmtree(temp_dir)


class TestMultiTickerIntradayRLEnv(unittest.TestCase):
    """Test cases for MultiTickerIntradayRLEnv."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'environment': {
                'initial_cash': 100000,
                'max_positions': 5,
                'position_size': 0.2,
                'commission': 0.001,
                'slippage': 0.0005,
                'reward_type': 'hybrid2',
                'tickers': ['AAPL', 'MSFT', 'GOOGL'],
                'lookback_window': 10,
                'episode_length': 100,
                'trading_hours': {
                    'start': '09:30',
                    'end': '16:00'
                }
            },
            'risk': {
                'max_drawdown': 0.1,
                'max_position_size': 0.3,
                'stop_loss': 0.05,
                'take_profit': 0.1
            }
        }
        
        # Create sample data
        self._create_sample_data()
        
        # Initialize environment
        self.env = MultiTickerIntradayRLEnv(self.config, self.sample_data)
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def _create_sample_data(self):
        """Create sample data for testing."""
        tickers = self.config['environment']['tickers']
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='H')
        
        self.sample_data = {}
        
        for ticker in tickers:
            # Create OHLCV data
            np.random.seed(hash(ticker) % 1000)  # Consistent random data per ticker
            
            close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
            open_prices = close_prices * (1 + np.random.randn(len(dates)) * 0.002)
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.randn(len(dates))) * 0.005)
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.randn(len(dates))) * 0.005)
            volumes = np.random.randint(10000, 100000, len(dates))
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
            
            self.sample_data[ticker] = df
            
    def test_reset(self):
        """Test resetting the environment."""
        obs = self.env.reset()
        
        self.assertIsInstance(obs, dict)
        self.assertIn('features', obs)
        self.assertIn('portfolio', obs)
        self.assertIn('positions', obs)
        self.assertIn('cash', obs)
        self.assertIn('equity', obs)
        
    def test_step(self):
        """Test stepping through the environment."""
        # Reset environment
        self.env.reset()
        
        # Get action space
        action_space = self.env.action_space
        
        # Take a random action
        action = action_space.sample()
        obs, reward, done, info = self.env.step(action)
        
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
    def test_get_portfolio_value(self):
        """Test getting portfolio value."""
        # Reset environment
        self.env.reset()
        
        # Get portfolio value
        portfolio_value = self.env.get_portfolio_value()
        
        self.assertIsInstance(portfolio_value, float)
        self.assertGreater(portfolio_value, 0)
        
    def test_get_positions(self):
        """Test getting positions."""
        # Reset environment
        self.env.reset()
        
        # Get positions
        positions = self.env.get_positions()
        
        self.assertIsInstance(positions, dict)
        self.assertEqual(len(positions), len(self.config['environment']['tickers']))
        
        for ticker in self.config['environment']['tickers']:
            self.assertIn(ticker, positions)
            self.assertIsInstance(positions[ticker], float)
            
    def test_get_cash(self):
        """Test getting cash."""
        # Reset environment
        self.env.reset()
        
        # Get cash
        cash = self.env.get_cash()
        
        self.assertIsInstance(cash, float)
        self.assertGreaterEqual(cash, 0)
        
    def test_get_equity_curve(self):
        """Test getting equity curve."""
        # Reset environment
        self.env.reset()
        
        # Take a few steps
        for _ in range(10):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            if done:
                break
                
        # Get equity curve
        equity_curve = self.env.get_equity_curve()
        
        self.assertIsInstance(equity_curve, list)
        self.assertGreater(len(equity_curve), 0)
        
    def test_get_trade_history(self):
        """Test getting trade history."""
        # Reset environment
        self.env.reset()
        
        # Take a few steps
        for _ in range(10):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            if done:
                break
                
        # Get trade history
        trade_history = self.env.get_trade_history()
        
        self.assertIsInstance(trade_history, list)
        
    def test_render(self):
        """Test rendering the environment."""
        # Reset environment
        self.env.reset()
        
        # Take a few steps
        for _ in range(10):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            # Render
            self.env.render()
            
            if done:
                break


class TestMultiTickerRLTrainer(unittest.TestCase):
    """Test cases for MultiTickerRLTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'training': {
                'total_timesteps': 1000,
                'learning_rate': 0.0003,
                'batch_size': 64,
                'n_steps': 2048,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'target_kl': 0.01,
                'tensorboard_log': '/tmp/tensorboard',
                'save_path': '/tmp/models'
            },
            'environment': {
                'initial_cash': 100000,
                'max_positions': 5,
                'position_size': 0.2,
                'commission': 0.001,
                'slippage': 0.0005,
                'reward_type': 'hybrid2',
                'tickers': ['AAPL', 'MSFT', 'GOOGL'],
                'lookback_window': 10,
                'episode_length': 100
            },
            'paths': {
                'models_dir': '/tmp/models',
                'logs_dir': '/tmp/logs'
            }
        }
        
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.config['paths']['models_dir'] = os.path.join(self.temp_dir, 'models')
        self.config['paths']['logs_dir'] = os.path.join(self.temp_dir, 'logs')
        os.makedirs(self.config['paths']['models_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        
        # Create sample data
        self._create_sample_data()
        
        # Initialize trainer
        self.trainer = MultiTickerRLTrainer(self.config)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def _create_sample_data(self):
        """Create sample data for testing."""
        tickers = self.config['environment']['tickers']
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='H')
        
        self.sample_data = {}
        
        for ticker in tickers:
            # Create OHLCV data
            np.random.seed(hash(ticker) % 1000)  # Consistent random data per ticker
            
            close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
            open_prices = close_prices * (1 + np.random.randn(len(dates)) * 0.002)
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.randn(len(dates))) * 0.005)
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.randn(len(dates))) * 0.005)
            volumes = np.random.randint(10000, 100000, len(dates))
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
            
            self.sample_data[ticker] = df
            
    def test_create_env(self):
        """Test creating environment."""
        env = self.trainer.create_env(self.sample_data)
        
        self.assertIsInstance(env, MultiTickerIntradayRLEnv)
        
    def test_create_model(self):
        """Test creating model."""
        # Create environment
        env = self.trainer.create_env(self.sample_data)
        
        # Create model
        model = self.trainer.create_model(env)
        
        self.assertIsInstance(model, MultiTickerPPOLSTMPolicy)
        
    def test_train(self):
        """Test training model."""
        # Train model
        model = self.trainer.train(self.sample_data)
        
        self.assertIsInstance(model, MultiTickerPPOLSTMPolicy)
        
    def test_save_load_model(self):
        """Test saving and loading model."""
        # Train model
        model = self.trainer.train(self.sample_data)
        
        # Save model
        model_path = os.path.join(self.config['paths']['models_dir'], 'test_model')
        self.trainer.save_model(model, model_path)
        
        # Load model
        loaded_model = self.trainer.load_model(model_path)
        
        self.assertIsInstance(loaded_model, MultiTickerPPOLSTMPolicy)
        
    def test_evaluate_model(self):
        """Test evaluating model."""
        # Train model
        model = self.trainer.train(self.sample_data)
        
        # Evaluate model
        eval_results = self.trainer.evaluate_model(model, self.sample_data)
        
        self.assertIsInstance(eval_results, dict)
        self.assertIn('total_return', eval_results)
        self.assertIn('sharpe_ratio', eval_results)
        self.assertIn('max_drawdown', eval_results)
        self.assertIn('win_rate', eval_results)


class TestMultiTickerEvaluator(unittest.TestCase):
    """Test cases for MultiTickerEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'evaluation': {
                'metrics': [
                    'total_return',
                    'annual_return',
                    'sharpe_ratio',
                    'sortino_ratio',
                    'calmar_ratio',
                    'max_drawdown',
                    'win_rate',
                    'profit_factor'
                ],
                'benchmark': 'SPY',
                'risk_free_rate': 0.02,
                'confidence_level': 0.95
            }
        }
        
        # Create sample data
        self._create_sample_data()
        
        # Initialize evaluator
        self.evaluator = MultiTickerEvaluator(self.config)
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def _create_sample_data(self):
        """Create sample data for testing."""
        # Create equity curve
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.randn(len(dates)) * 0.02
        equity = [100000]
        
        for ret in returns:
            equity.append(equity[-1] * (1 + ret))
            
        self.equity_curve = equity[1:]
        self.timestamps = dates
        
        # Create trade history
        self.trade_history = []
        
        for i in range(100):
            entry_time = dates[i * 2]
            exit_time = dates[i * 2 + 1]
            entry_price = 100 + np.random.randn() * 5
            exit_price = entry_price * (1 + np.random.randn() * 0.02)
            quantity = np.random.randint(10, 100)
            direction = np.random.choice(['long', 'short'])
            pnl = (exit_price - entry_price) * quantity if direction == 'long' else (entry_price - exit_price) * quantity
            
            self.trade_history.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'direction': direction,
                'pnl': pnl,
                'ticker': np.random.choice(['AAPL', 'MSFT', 'GOOGL'])
            })
            
        # Create ticker metrics
        self.ticker_metrics = {
            'AAPL': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'win_rate': 0.55,
                'num_trades': 35
            },
            'MSFT': {
                'total_return': 0.12,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.10,
                'win_rate': 0.52,
                'num_trades': 33
            },
            'GOOGL': {
                'total_return': 0.18,
                'sharpe_ratio': 1.4,
                'max_drawdown': 0.07,
                'win_rate': 0.58,
                'num_trades': 32
            }
        }
        
        # Create regime metrics
        self.regime_metrics = {
            'bull': {
                'total_return': 0.25,
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.05,
                'win_rate': 0.65,
                'num_trades': 40
            },
            'bear': {
                'total_return': -0.05,
                'sharpe_ratio': -0.5,
                'max_drawdown': 0.15,
                'win_rate': 0.35,
                'num_trades': 30
            },
            'sideways': {
                'total_return': 0.02,
                'sharpe_ratio': 0.3,
                'max_drawdown': 0.08,
                'win_rate': 0.48,
                'num_trades': 30
            }
        }
        
    def test_evaluate(self):
        """Test evaluating backtest results."""
        results = self.evaluator.evaluate(
            self.equity_curve,
            self.timestamps,
            self.trade_history,
            self.ticker_metrics,
            self.regime_metrics
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('total_return', results)
        self.assertIn('annual_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('win_rate', results)
        
    def test_calculate_metrics(self):
        """Test calculating metrics."""
        metrics = self.evaluator.calculate_metrics(
            self.equity_curve,
            self.timestamps,
            self.trade_history
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_return', metrics)
        self.assertIn('annual_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        
    def test_calculate_ticker_metrics(self):
        """Test calculating ticker metrics."""
        ticker_metrics = self.evaluator.calculate_ticker_metrics(self.trade_history)
        
        self.assertIsInstance(ticker_metrics, dict)
        
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            if ticker in ticker_metrics:
                self.assertIn('total_return', ticker_metrics[ticker])
                self.assertIn('sharpe_ratio', ticker_metrics[ticker])
                self.assertIn('max_drawdown', ticker_metrics[ticker])
                self.assertIn('win_rate', ticker_metrics[ticker])
                
    def test_calculate_regime_metrics(self):
        """Test calculating regime metrics."""
        # Create regime data
        regime_data = []
        
        for i in range(len(self.timestamps)):
            regime_data.append({
                'timestamp': self.timestamps[i],
                'regime': np.random.choice(['bull', 'bear', 'sideways'])
            })
            
        regime_metrics = self.evaluator.calculate_regime_metrics(
            self.equity_curve,
            self.timestamps,
            self.trade_history,
            regime_data
        )
        
        self.assertIsInstance(regime_metrics, dict)
        
        for regime in ['bull', 'bear', 'sideways']:
            if regime in regime_metrics:
                self.assertIn('total_return', regime_metrics[regime])
                self.assertIn('sharpe_ratio', regime_metrics[regime])
                self.assertIn('max_drawdown', regime_metrics[regime])
                self.assertIn('win_rate', regime_metrics[regime])
                
    def test_generate_report(self):
        """Test generating report."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Generate report
            report_path = os.path.join(temp_dir, 'evaluation_report.html')
            self.evaluator.generate_report(
                self.equity_curve,
                self.timestamps,
                self.trade_history,
                self.ticker_metrics,
                self.regime_metrics,
                report_path
            )
            
            # Check if report file exists
            self.assertTrue(os.path.exists(report_path))
            
        finally:
            shutil.rmtree(temp_dir)


class TestMultiTickerMonitor(unittest.TestCase):
    """Test cases for MultiTickerMonitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'monitoring': {
                'metrics': [
                    'total_return',
                    'sharpe_ratio',
                    'max_drawdown',
                    'win_rate'
                ],
                'update_interval': 60,
                'dashboard_port': 8080,
                'alert_thresholds': {
                    'drawdown': 0.1,
                    'sharpe_ratio': 0.5,
                    'win_rate': 0.4
                }
            }
        }
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize monitor
        self.monitor = MultiTickerMonitor(self.config)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_update_metrics(self):
        """Test updating metrics."""
        metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'win_rate': 0.55
        }
        
        self.monitor.update_metrics(metrics)
        
        # Check if metrics were updated
        self.assertEqual(self.monitor.metrics['total_return'], 0.15)
        self.assertEqual(self.monitor.metrics['sharpe_ratio'], 1.2)
        self.assertEqual(self.monitor.metrics['max_drawdown'], 0.08)
        self.assertEqual(self.monitor.metrics['win_rate'], 0.55)
        
    def test_check_alerts(self):
        """Test checking alerts."""
        # Update metrics that should trigger alerts
        metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 0.3,  # Below threshold
            'max_drawdown': 0.12,  # Above threshold
            'win_rate': 0.35  # Below threshold
        }
        
        self.monitor.update_metrics(metrics)
        
        # Check alerts
        alerts = self.monitor.check_alerts()
        
        self.assertIsInstance(alerts, list)
        self.assertGreater(len(alerts), 0)
        
    def test_get_metrics_history(self):
        """Test getting metrics history."""
        # Update metrics multiple times
        for i in range(10):
            metrics = {
                'total_return': 0.01 * i,
                'sharpe_ratio': 0.1 * i,
                'max_drawdown': 0.01 * i,
                'win_rate': 0.4 + 0.01 * i
            }
            
            self.monitor.update_metrics(metrics)
            
        # Get metrics history
        history = self.monitor.get_metrics_history()
        
        self.assertIsInstance(history, dict)
        self.assertIn('total_return', history)
        self.assertIn('sharpe_ratio', history)
        self.assertIn('max_drawdown', history)
        self.assertIn('win_rate', history)
        
        for metric_name, metric_values in history.items():
            self.assertIsInstance(metric_values, list)
            self.assertEqual(len(metric_values), 10)
            
    def test_save_load_metrics(self):
        """Test saving and loading metrics."""
        # Update metrics
        metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'win_rate': 0.55
        }
        
        self.monitor.update_metrics(metrics)
        
        # Save metrics
        metrics_path = os.path.join(self.temp_dir, 'metrics.json')
        self.monitor.save_metrics(metrics_path)
        
        # Load metrics
        loaded_monitor = MultiTickerMonitor(self.config)
        loaded_monitor.load_metrics(metrics_path)
        
        # Check if metrics were loaded correctly
        self.assertEqual(loaded_monitor.metrics['total_return'], 0.15)
        self.assertEqual(loaded_monitor.metrics['sharpe_ratio'], 1.2)
        self.assertEqual(loaded_monitor.metrics['max_drawdown'], 0.08)
        self.assertEqual(loaded_monitor.metrics['win_rate'], 0.55)


class TestAlertSystem(unittest.TestCase):
    """Test cases for AlertSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'alert_thresholds': {
                'drawdown': 0.1,
                'sharpe_ratio': 0.5,
                'win_rate': 0.4,
                'volatility': 0.3
            },
            'notification_channels': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': 'test@gmail.com',
                    'password': 'password',
                    'to_emails': ['alert@example.com']
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': 'https://hooks.slack.com/services/test'
                }
            },
            'suppression_rules': {
                'drawdown': {
                    'suppress_for': 3600  # 1 hour
                }
            },
            'aggregation_rules': {
                'system_error': {
                    'aggregate_for': 300  # 5 minutes
                }
            }
        }
        
        # Initialize alert system
        self.alert_system = AlertSystem(self.config)
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_add_alert(self):
        """Test adding alert."""
        alert = {
            'type': 'drawdown',
            'severity': 'high',
            'message': 'Drawdown exceeded threshold',
            'value': 0.15,
            'threshold': 0.1
        }
        
        self.alert_system.add_alert(alert)
        
        # Check if alert was added
        self.assertEqual(len(self.alert_system.alerts), 1)
        self.assertEqual(self.alert_system.alerts[0]['type'], 'drawdown')
        
    def test_check_performance_alerts(self):
        """Test checking performance alerts."""
        metrics = {
            'max_drawdown': 0.15,  # Above threshold
            'sharpe_ratio': 0.3,  # Below threshold
            'win_rate': 0.35,  # Below threshold
            'annual_volatility': 0.35  # Above threshold
        }
        
        alerts = self.alert_system.check_performance_alerts(metrics)
        
        self.assertIsInstance(alerts, list)
        self.assertGreater(len(alerts), 0)
        
        # Check if alerts were added to system
        self.assertGreater(len(self.alert_system.alerts), 0)
        
    def test_check_trade_alerts(self):
        """Test checking trade alerts."""
        trade = {
            'entry_time': '2023-01-01 09:30:00',
            'exit_time': '2023-01-01 16:00:00',
            'ticker': 'AAPL',
            'pnl': 100
        }
        
        alerts = self.alert_system.check_trade_alerts(trade)
        
        self.assertIsInstance(alerts, list)
        
    def test_check_regime_alert(self):
        """Test checking regime alert."""
        alert = self.alert_system.check_regime_alert('bull', 'bear')
        
        self.assertIsInstance(alert, dict)
        self.assertEqual(alert['type'], 'regime_change')
        self.assertEqual(alert['from_regime'], 'bull')
        self.assertEqual(alert['to_regime'], 'bear')
        
    def test_add_system_error_alert(self):
        """Test adding system error alert."""
        alert = self.alert_system.add_system_error_alert('Test error message')
        
        self.assertIsInstance(alert, dict)
        self.assertEqual(alert['type'], 'system_error')
        self.assertEqual(alert['message'], 'System error: Test error message')
        
    def test_add_data_quality_alert(self):
        """Test adding data quality alert."""
        alert = self.alert_system.add_data_quality_alert('Missing data', 'AAPL')
        
        self.assertIsInstance(alert, dict)
        self.assertEqual(alert['type'], 'data_quality')
        self.assertEqual(alert['ticker'], 'AAPL')
        
    def test_add_model_performance_alert(self):
        """Test adding model performance alert."""
        alert = self.alert_system.add_model_performance_alert('Poor prediction', 'AAPL')
        
        self.assertIsInstance(alert, dict)
        self.assertEqual(alert['type'], 'model_performance')
        self.assertEqual(alert['ticker'], 'AAPL')
        
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Add some alerts
        for i in range(5):
            alert = {
                'type': f'test_alert_{i}',
                'severity': 'medium',
