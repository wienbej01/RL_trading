"""
Performance benchmarking framework for multi-ticker RL trading system.

This module provides a comprehensive performance benchmarking framework for the multi-ticker RL trading system,
including comparison with baseline strategies, statistical significance testing, and performance attribution.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from src.data.multiticker_data_loader import MultiTickerDataLoader
from src.features.multiticker_pipeline import MultiTickerFeaturePipeline
from src.sim.multiticker_env import MultiTickerIntradayRLEnv
from src.rl.multiticker_trainer import MultiTickerRLTrainer
from src.rl.multiticker_policy import MultiTickerPPOLSTMPolicy
from src.evaluation.multiticker_evaluator import MultiTickerEvaluator
from src.monitoring.multiticker_monitor import MultiTickerMonitor
from src.utils.config_loader import load_config


class BenchmarkStrategy:
    """
    Benchmark strategy for comparison.
    
    This class represents a benchmark strategy that can be compared
    against the RL trading system.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        strategy_type: str,
        strategy_params: Dict[str, Any]
    ):
        """
        Initialize benchmark strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            strategy_type: Type of strategy
            strategy_params: Strategy parameters
        """
        self.name = name
        self.description = description
        self.strategy_type = strategy_type
        self.strategy_params = strategy_params
        self.results = {}
        
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run the benchmark strategy.
        
        Args:
            data: Market data for all tickers
            
        Returns:
            Strategy results
        """
        print(f"Running benchmark strategy: {self.name}")
        
        if self.strategy_type == "buy_and_hold":
            return self._run_buy_and_hold(data)
        elif self.strategy_type == "moving_average_crossover":
            return self._run_moving_average_crossover(data)
        elif self.strategy_type == "mean_reversion":
            return self._run_mean_reversion(data)
        elif self.strategy_type == "momentum":
            return self._run_momentum(data)
        elif self.strategy_type == "volatility_targeting":
            return self._run_volatility_targeting(data)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
            
    def _run_buy_and_hold(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run buy and hold strategy.
        
        Args:
            data: Market data for all tickers
            
        Returns:
            Strategy results
        """
        results = {
            'equity_curve': [],
            'returns': [],
            'positions': {},
            'trades': [],
            'metrics': {}
        }
        
        # Initialize positions
        initial_capital = self.strategy_params.get('initial_capital', 100000)
        position_size = self.strategy_params.get('position_size', 0.1)
        
        # Equal weight portfolio
        num_tickers = len(data)
        weight_per_ticker = position_size / num_tickers if num_tickers > 0 else 0
        
        # Calculate initial positions
        for ticker, ticker_data in data.items():
            initial_price = ticker_data['close'].iloc[0]
            shares = (initial_capital * weight_per_ticker) / initial_price
            results['positions'][ticker] = shares
        
        # Simulate buy and hold
        equity = initial_capital
        
        # Get all timestamps
        all_timestamps = set()
        for ticker_data in data.values():
            all_timestamps.update(ticker_data.index)
        all_timestamps = sorted(all_timestamps)
        
        for timestamp in all_timestamps:
            # Calculate portfolio value at timestamp
            portfolio_value = 0
            
            for ticker, shares in results['positions'].items():
                if ticker in data and timestamp in data[ticker].index:
                    price = data[ticker].loc[timestamp, 'close']
                    portfolio_value += shares * price
            
            # Store equity
            results['equity_curve'].append({
                'timestamp': timestamp,
                'equity': portfolio_value
            })
            
            # Calculate return
            if len(results['equity_curve']) > 1:
                prev_equity = results['equity_curve'][-2]['equity']
                ret = (portfolio_value - prev_equity) / prev_equity
                results['returns'].append(ret)
            else:
                results['returns'].append(0.0)
        
        # Calculate metrics
        returns = np.array(results['returns'])
        equity_values = [eq['equity'] for eq in results['equity_curve']]
        
        results['metrics'] = self._calculate_metrics(returns, equity_values)
        
        return results
        
    def _run_moving_average_crossover(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run moving average crossover strategy.
        
        Args:
            data: Market data for all tickers
            
        Returns:
            Strategy results
        """
        results = {
            'equity_curve': [],
            'returns': [],
            'positions': {},
            'trades': [],
            'metrics': {}
        }
        
        # Get parameters
        initial_capital = self.strategy_params.get('initial_capital', 100000)
        position_size = self.strategy_params.get('position_size', 0.1)
        short_window = self.strategy_params.get('short_window', 10)
        long_window = self.strategy_params.get('long_window', 30)
        
        # Initialize positions and cash
        positions = {}
        cash = initial_capital
        
        # Get all timestamps
        all_timestamps = set()
        for ticker_data in data.values():
            all_timestamps.update(ticker_data.index)
        all_timestamps = sorted(all_timestamps)
        
        # Calculate moving averages for each ticker
        mas = {}
        for ticker, ticker_data in data.items():
            ticker_data = ticker_data.copy()
            ticker_data['ma_short'] = ticker_data['close'].rolling(window=short_window).mean()
            ticker_data['ma_long'] = ticker_data['close'].rolling(window=long_window).mean()
            mas[ticker] = ticker_data
        
        # Simulate strategy
        prev_signals = {}
        
        for timestamp in all_timestamps:
            # Calculate signals for each ticker
            signals = {}
            
            for ticker, ticker_data in mas.items():
                if timestamp in ticker_data.index:
                    ma_short = ticker_data.loc[timestamp, 'ma_short']
                    ma_long = ticker_data.loc[timestamp, 'ma_long']
                    
                    if pd.notna(ma_short) and pd.notna(ma_long):
                        # Buy signal: short MA crosses above long MA
                        if ma_short > ma_long:
                            signals[ticker] = 1
                        # Sell signal: short MA crosses below long MA
                        else:
                            signals[ticker] = -1
                    else:
                        signals[ticker] = 0
                else:
                    signals[ticker] = 0
            
            # Execute trades based on signal changes
            for ticker, signal in signals.items():
                prev_signal = prev_signals.get(ticker, 0)
                
                if signal != prev_signal:
                    # Signal changed, execute trade
                    if signal == 1 and prev_signal != 1:
                        # Buy signal
                        if ticker in data and timestamp in data[ticker].index:
                            price = data[ticker].loc[timestamp, 'close']
                            shares_to_buy = (cash * position_size) / price
                            
                            if shares_to_buy > 0:
                                positions[ticker] = positions.get(ticker, 0) + shares_to_buy
                                cash -= shares_to_buy * price
                                
                                results['trades'].append({
                                    'timestamp': timestamp,
                                    'ticker': ticker,
                                    'action': 'buy',
                                    'shares': shares_to_buy,
                                    'price': price
                                })
                    
                    elif signal == -1 and prev_signal != -1:
                        # Sell signal
                        if ticker in positions and positions[ticker] > 0:
                            if ticker in data and timestamp in data[ticker].index:
                                price = data[ticker].loc[timestamp, 'close']
                                shares_to_sell = positions[ticker]
                                
                                cash += shares_to_sell * price
                                positions[ticker] = 0
                                
                                results['trades'].append({
                                    'timestamp': timestamp,
                                    'ticker': ticker,
                                    'action': 'sell',
                                    'shares': shares_to_sell,
                                    'price': price
                                })
                
                prev_signals[ticker] = signal
            
            # Calculate portfolio value
            portfolio_value = cash
            
            for ticker, shares in positions.items():
                if ticker in data and timestamp in data[ticker].index:
                    price = data[ticker].loc[timestamp, 'close']
                    portfolio_value += shares * price
            
            # Store equity
            results['equity_curve'].append({
                'timestamp': timestamp,
                'equity': portfolio_value
            })
            
            # Calculate return
            if len(results['equity_curve']) > 1:
                prev_equity = results['equity_curve'][-2]['equity']
                ret = (portfolio_value - prev_equity) / prev_equity
                results['returns'].append(ret)
            else:
                results['returns'].append(0.0)
        
        # Store final positions
        results['positions'] = positions
        
        # Calculate metrics
        returns = np.array(results['returns'])
        equity_values = [eq['equity'] for eq in results['equity_curve']]
        
        results['metrics'] = self._calculate_metrics(returns, equity_values)
        
        return results
        
    def _run_mean_reversion(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run mean reversion strategy.
        
        Args:
            data: Market data for all tickers
            
        Returns:
            Strategy results
        """
        results = {
            'equity_curve': [],
            'returns': [],
            'positions': {},
            'trades': [],
            'metrics': {}
        }
        
        # Get parameters
        initial_capital = self.strategy_params.get('initial_capital', 100000)
        position_size = self.strategy_params.get('position_size', 0.1)
        lookback_window = self.strategy_params.get('lookback_window', 20)
        entry_threshold = self.strategy_params.get('entry_threshold', 2.0)
        exit_threshold = self.strategy_params.get('exit_threshold', 0.5)
        
        # Initialize positions and cash
        positions = {}
        cash = initial_capital
        
        # Get all timestamps
        all_timestamps = set()
        for ticker_data in data.values():
            all_timestamps.update(ticker_data.index)
        all_timestamps = sorted(all_timestamps)
        
        # Calculate z-scores for each ticker
        z_scores = {}
        for ticker, ticker_data in data.items():
            ticker_data = ticker_data.copy()
            ticker_data['mean'] = ticker_data['close'].rolling(window=lookback_window).mean()
            ticker_data['std'] = ticker_data['close'].rolling(window=lookback_window).std()
            ticker_data['z_score'] = (ticker_data['close'] - ticker_data['mean']) / ticker_data['std']
            z_scores[ticker] = ticker_data
        
        # Simulate strategy
        prev_positions = {}
        
        for timestamp in all_timestamps:
            # Calculate z-scores for each ticker
            current_z_scores = {}
            
            for ticker, ticker_data in z_scores.items():
                if timestamp in ticker_data.index:
                    z_score = ticker_data.loc[timestamp, 'z_score']
                    if pd.notna(z_score):
                        current_z_scores[ticker] = z_score
            
            # Execute trades based on z-scores
            for ticker, z_score in current_z_scores.items():
                prev_position = prev_positions.get(ticker, 0)
                
                # Entry signals
                if z_score < -entry_threshold and prev_position >= 0:
                    # Buy signal (price is below mean)
                    if ticker in data and timestamp in data[ticker].index:
                        price = data[ticker].loc[timestamp, 'close']
                        shares_to_buy = (cash * position_size) / price
                        
                        if shares_to_buy > 0:
                            positions[ticker] = positions.get(ticker, 0) + shares_to_buy
                            cash -= shares_to_buy * price
                            
                            results['trades'].append({
                                'timestamp': timestamp,
                                'ticker': ticker,
                                'action': 'buy',
                                'shares': shares_to_buy,
                                'price': price
                            })
                
                elif z_score > entry_threshold and prev_position <= 0:
                    # Sell signal (price is above mean)
                    if ticker in positions and positions[ticker] > 0:
                        if ticker in data and timestamp in data[ticker].index:
                            price = data[ticker].loc[timestamp, 'close']
                            shares_to_sell = positions[ticker]
                            
                            cash += shares_to_sell * price
                            positions[ticker] = 0
                            
                            results['trades'].append({
                                'timestamp': timestamp,
                                'ticker': ticker,
                                'action': 'sell',
                                'shares': shares_to_sell,
                                'price': price
                            })
                
                # Exit signals
                elif abs(z_score) < exit_threshold:
                    # Close position if z-score is close to zero
                    if ticker in positions and positions[ticker] > 0:
                        if ticker in data and timestamp in data[ticker].index:
                            price = data[ticker].loc[timestamp, 'close']
                            shares_to_sell = positions[ticker]
                            
                            cash += shares_to_sell * price
                            positions[ticker] = 0
                            
                            results['trades'].append({
                                'timestamp': timestamp,
                                'ticker': ticker,
                                'action': 'sell',
                                'shares': shares_to_sell,
                                'price': price
                            })
                
                prev_positions[ticker] = positions.get(ticker, 0)
            
            # Calculate portfolio value
            portfolio_value = cash
            
            for ticker, shares in positions.items():
                if ticker in data and timestamp in data[ticker].index:
                    price = data[ticker].loc[timestamp, 'close']
                    portfolio_value += shares * price
            
            # Store equity
            results['equity_curve'].append({
                'timestamp': timestamp,
                'equity': portfolio_value
            })
            
            # Calculate return
            if len(results['equity_curve']) > 1:
                prev_equity = results['equity_curve'][-2]['equity']
                ret = (portfolio_value - prev_equity) / prev_equity
                results['returns'].append(ret)
            else:
                results['returns'].append(0.0)
        
        # Store final positions
        results['positions'] = positions
        
        # Calculate metrics
        returns = np.array(results['returns'])
        equity_values = [eq['equity'] for eq in results['equity_curve']]
        
        results['metrics'] = self._calculate_metrics(returns, equity_values)
        
        return results
        
    def _run_momentum(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run momentum strategy.
        
        Args:
            data: Market data for all tickers
            
        Returns:
            Strategy results
        """
        results = {
            'equity_curve': [],
            'returns': [],
            'positions': {},
            'trades': [],
            'metrics': {}
        }
        
        # Get parameters
        initial_capital = self.strategy_params.get('initial_capital', 100000)
        position_size = self.strategy_params.get('position_size', 0.1)
        lookback_window = self.strategy_params.get('lookback_window', 20)
        top_n = self.strategy_params.get('top_n', 3)
        
        # Initialize positions and cash
        positions = {}
        cash = initial_capital
        
        # Get all timestamps
        all_timestamps = set()
        for ticker_data in data.values():
            all_timestamps.update(ticker_data.index)
        all_timestamps = sorted(all_timestamps)
        
        # Calculate momentum for each ticker
        momentum = {}
        for ticker, ticker_data in data.items():
            ticker_data = ticker_data.copy()
            ticker_data['momentum'] = ticker_data['close'].pct_change(lookback_window)
            momentum[ticker] = ticker_data
        
        # Simulate strategy
        prev_positions = {}
        
        for timestamp in all_timestamps:
            # Calculate momentum for each ticker
            current_momentum = {}
            
            for ticker, ticker_data in momentum.items():
                if timestamp in ticker_data.index:
                    mom = ticker_data.loc[timestamp, 'momentum']
                    if pd.notna(mom):
                        current_momentum[ticker] = mom
            
            # Sort tickers by momentum
            sorted_tickers = sorted(current_momentum.items(), key=lambda x: x[1], reverse=True)
            
            # Select top N tickers
            top_tickers = [ticker for ticker, _ in sorted_tickers[:top_n]]
            
            # Execute trades
            for ticker in current_momentum.keys():
                prev_position = prev_positions.get(ticker, 0)
                
                if ticker in top_tickers and prev_position <= 0:
                    # Buy top momentum tickers
                    if ticker in data and timestamp in data[ticker].index:
                        price = data[ticker].loc[timestamp, 'close']
                        shares_to_buy = (cash * position_size / top_n) / price
                        
                        if shares_to_buy > 0:
                            positions[ticker] = positions.get(ticker, 0) + shares_to_buy
                            cash -= shares_to_buy * price
                            
                            results['trades'].append({
                                'timestamp': timestamp,
                                'ticker': ticker,
                                'action': 'buy',
                                'shares': shares_to_buy,
                                'price': price
                            })
                
                elif ticker not in top_tickers and prev_position > 0:
                    # Sell tickers not in top N
                    if ticker in positions and positions[ticker] > 0:
                        if ticker in data and timestamp in data[ticker].index:
                            price = data[ticker].loc[timestamp, 'close']
                            shares_to_sell = positions[ticker]
                            
                            cash += shares_to_sell * price
                            positions[ticker] = 0
                            
                            results['trades'].append({
                                'timestamp': timestamp,
                                'ticker': ticker,
                                'action': 'sell',
                                'shares': shares_to_sell,
                                'price': price
                            })
                
                prev_positions[ticker] = positions.get(ticker, 0)
            
            # Calculate portfolio value
            portfolio_value = cash
            
            for ticker, shares in positions.items():
                if ticker in data and timestamp in data[ticker].index:
                    price = data[ticker].loc[timestamp, 'close']
                    portfolio_value += shares * price
            
            # Store equity
            results['equity_curve'].append({
                'timestamp': timestamp,
                'equity': portfolio_value
            })
            
            # Calculate return
            if len(results['equity_curve']) > 1:
                prev_equity = results['equity_curve'][-2]['equity']
                ret = (portfolio_value - prev_equity) / prev_equity
                results['returns'].append(ret)
            else:
                results['returns'].append(0.0)
        
        # Store final positions
        results['positions'] = positions
        
        # Calculate metrics
        returns = np.array(results['returns'])
        equity_values = [eq['equity'] for eq in results['equity_curve']]
        
        results['metrics'] = self._calculate_metrics(returns, equity_values)
        
        return results
        
    def _run_volatility_targeting(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run volatility targeting strategy.
        
        Args:
            data: Market data for all tickers
            
        Returns:
            Strategy results
        """
        results = {
            'equity_curve': [],
            'returns': [],
            'positions': {},
            'trades': [],
            'metrics': {}
        }
        
        # Get parameters
        initial_capital = self.strategy_params.get('initial_capital', 100000)
        target_volatility = self.strategy_params.get('target_volatility', 0.15)
        lookback_window = self.strategy_params.get('lookback_window', 20)
        rebalance_frequency = self.strategy_params.get('rebalance_frequency', 5)
        
        # Initialize positions and cash
        positions = {}
        cash = initial_capital
        
        # Get all timestamps
        all_timestamps = set()
        for ticker_data in data.values():
            all_timestamps.update(ticker_data.index)
        all_timestamps = sorted(all_timestamps)
        
        # Calculate volatility for each ticker
        volatility = {}
        for ticker, ticker_data in data.items():
            ticker_data = ticker_data.copy()
            ticker_data['volatility'] = ticker_data['close'].pct_change().rolling(window=lookback_window).std() * np.sqrt(252)
            volatility[ticker] = ticker_data
        
        # Simulate strategy
        prev_positions = {}
        rebalance_counter = 0
        
        for timestamp in all_timestamps:
            # Calculate volatility for each ticker
            current_volatility = {}
            
            for ticker, ticker_data in volatility.items():
                if timestamp in ticker_data.index:
                    vol = ticker_data.loc[timestamp, 'volatility']
                    if pd.notna(vol) and vol > 0:
                        current_volatility[ticker] = vol
            
            # Rebalance portfolio
            if rebalance_counter % rebalance_frequency == 0 and current_volatility:
                # Calculate inverse volatility weights
                inv_vol = {ticker: 1/vol for ticker, vol in current_volatility.items()}
                total_inv_vol = sum(inv_vol.values())
                
                if total_inv_vol > 0:
                    # Normalize weights
                    weights = {ticker: inv_vol[ticker] / total_inv_vol for ticker in inv_vol.keys()}
                    
                    # Scale weights to achieve target volatility
                    portfolio_volatility = np.sqrt(sum(weights[ticker]**2 * current_volatility[ticker]**2 
                                                      for ticker in weights.keys()))
                    
                    if portfolio_volatility > 0:
                        scale_factor = target_volatility / portfolio_volatility
                        weights = {ticker: weight * scale_factor for ticker, weight in weights.items()}
                        
                        # Execute trades to achieve target weights
                        for ticker, target_weight in weights.items():
                            if ticker in data and timestamp in data[ticker].index:
                                price = data[ticker].loc[timestamp, 'close']
                                portfolio_value = cash + sum(positions.get(t, 0) * 
                                                            data[t].loc[timestamp, 'close'] 
                                                            for t in positions.keys() 
                                                            if t in data and timestamp in data[t].index)
                                
                                target_value = portfolio_value * target_weight
                                current_value = positions.get(ticker, 0) * price
                                
                                if target_value > current_value:
                                    # Buy
                                    value_to_buy = target_value - current_value
                                    shares_to_buy = value_to_buy / price
                                    
                                    if shares_to_buy > 0 and cash >= shares_to_buy * price:
                                        positions[ticker] = positions.get(ticker, 0) + shares_to_buy
                                        cash -= shares_to_buy * price
                                        
                                        results['trades'].append({
                                            'timestamp': timestamp,
                                            'ticker': ticker,
                                            'action': 'buy',
                                            'shares': shares_to_buy,
                                            'price': price
                                        })
                                
                                elif target_value < current_value:
                                    # Sell
                                    shares_to_sell = (current_value - target_value) / price
                                    shares_to_sell = min(shares_to_sell, positions.get(ticker, 0))
                                    
                                    if shares_to_sell > 0:
                                        positions[ticker] = positions.get(ticker, 0) - shares_to_sell
                                        cash += shares_to_sell * price
                                        
                                        results['trades'].append({
                                            'timestamp': timestamp,
                                            'ticker': ticker,
                                            'action': 'sell',
                                            'shares': shares_to_sell,
                                            'price': price
                                        })
            
            rebalance_counter += 1
            
            # Calculate portfolio value
            portfolio_value = cash
            
            for ticker, shares in positions.items():
                if ticker in data and timestamp in data[ticker].index:
                    price = data[ticker].loc[timestamp, 'close']
                    portfolio_value += shares * price
            
            # Store equity
            results['equity_curve'].append({
                'timestamp': timestamp,
                'equity': portfolio_value
            })
            
            # Calculate return
            if len(results['equity_curve']) > 1:
                prev_equity = results['equity_curve'][-2]['equity']
                ret = (portfolio_value - prev_equity) / prev_equity
                results['returns'].append(ret)
            else:
                results['returns'].append(0.0)
        
        # Store final positions
        results['positions'] = positions
        
        # Calculate metrics
        returns = np.array(results['returns'])
        equity_values = [eq['equity'] for eq in results['equity_curve']]
        
        results['metrics'] = self._calculate_metrics(returns, equity_values)
        
        return results
        
    def _calculate_metrics(self, returns: np.ndarray, equity_values: List[float]) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            returns: Array of returns
            equity_values: List of equity values
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        if len(returns) == 0:
            return metrics
        
        # Total return
        total_return = (equity_values[-1] / equity_values[0]) - 1 if len(equity_values) > 1 else 0.0
        metrics['total_return'] = total_return
        
        # Annualized return
        if len(returns) > 0:
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            metrics['annualized_return'] = annualized_return
        
        # Volatility
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)
            metrics['annualized_volatility'] = volatility
        
        # Sharpe ratio
        if 'annualized_return' in metrics and 'annualized_volatility' in metrics:
            if metrics['annualized_volatility'] > 0:
                sharpe_ratio = metrics['annualized_return'] / metrics['annualized_volatility']
                metrics['sharpe_ratio'] = sharpe_ratio
        
        # Maximum drawdown
        if len(equity_values) > 0:
            cumulative_max = np.maximum.accumulate(equity_values)
            drawdown = (equity_values - cumulative_max) / cumulative_max
            max_drawdown = np.min(drawdown)
            metrics['max_drawdown'] = max_drawdown
        
        # Win rate
        if len(returns) > 0:
            win_rate = np.mean(returns > 0)
            metrics['win_rate'] = win_rate
        
        # Profit factor
        if len(returns) > 0:
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) > 0:
                profit_factor = np.sum(positive_returns) / abs(np.sum(negative_returns))
                metrics['profit_factor'] = profit_factor
        
        return metrics
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert benchmark strategy to dictionary.
        
        Returns:
            Benchmark strategy as dictionary
        """
        return {
            'name': self.name,
            'description': self.description,
            'strategy_type': self.strategy_type,
            'strategy_params': self.strategy_params,
            'results': self.results
        }


class PerformanceBenchmarkingFramework:
    """
    Performance benchmarking framework for multi-ticker RL trading system.
    
    This class provides a comprehensive framework for benchmarking the performance
    of the multi-ticker RL trading system against various baseline strategies,
    including statistical significance testing and performance attribution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance benchmarking framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.benchmark_strategies = []
        self.rl_results = {}
        self.benchmark_results = {}
        self.comparison = {}
        self.statistical_tests = {}
        
        # Framework settings
        self.output_dir = config.get('benchmarking', {}).get('output_dir', '/tmp/benchmarking_results')
        self.significance_level = config.get('benchmarking', {}).get('significance_level', 0.05)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def add_benchmark_strategy(self, strategy: BenchmarkStrategy):
        """
        Add a benchmark strategy.
        
        Args:
            strategy: Benchmark strategy to add
        """
        self.benchmark_strategies.append(strategy)
        
    def add_default_benchmark_strategies(self) -> List[BenchmarkStrategy]:
        """
        Add default benchmark strategies.
        
        Returns:
            List of added benchmark strategies
        """
        added_strategies = []
        
        # Strategy 1: Buy and Hold
        strategy = BenchmarkStrategy(
            name="buy_and_hold",
            description="Buy and hold strategy with equal weighting",
            strategy_type="buy_and_hold",
            strategy_params={
                'initial_capital': 100000,
                'position_size': 1.0
            }
        )
        self.add_benchmark_strategy(strategy)
        added_strategies.append(strategy)
        
        # Strategy 2: Moving Average Crossover
        strategy = BenchmarkStrategy(
            name="moving_average_crossover",
            description="Moving average crossover strategy",
            strategy_type="moving_average_crossover",
            strategy_params={
                'initial_capital': 100000,
                'position_size': 0.2,
                'short_window': 10,
                'long_window': 30
            }
        )
        self.add_benchmark_strategy(strategy)
        added_strategies.append(strategy)
        
        # Strategy 3: Mean Reversion
        strategy = BenchmarkStrategy(
            name="mean_reversion",
            description="Mean reversion strategy based on z-score",
            strategy_type="mean_reversion",
            strategy_params={
                'initial_capital': 100000,
                'position_size': 0.1,
                'lookback_window': 20,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5
            }
        )
        self.add_benchmark_strategy(strategy)
        added_strategies.append(strategy)
        
        # Strategy 4: Momentum
        strategy = BenchmarkStrategy(
            name="momentum",
            description="Momentum strategy selecting top N performers",
            strategy_type="momentum",
            strategy_params={
                'initial_capital': 100000,
                'position_size': 0.3,
                'lookback_window': 20,
                'top_n': 3
            }
        )
        self.add_benchmark_strategy(strategy)
        added_strategies.append(strategy)
        
        # Strategy 5: Volatility Targeting
        strategy = BenchmarkStrategy(
            name="volatility_targeting",
            description="Volatility targeting strategy with inverse volatility weighting",
            strategy_type="volatility_targeting",
            strategy_params={
                'initial_capital': 100000,
                'target_volatility': 0.15,
                'lookback_window': 20,
                'rebalance_frequency': 5
            }
        )
        self.add_benchmark_strategy(strategy)
        added_strategies.append(strategy)
        
        return added_strategies
        
    def run_benchmarks(
        self,
        data: Dict[str, pd.DataFrame],
        rl_model: MultiTickerPPOLSTMPolicy,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Run benchmarking analysis.
        
        Args:
            data: Market data for all tickers
            rl_model: Trained RL model
            parallel: Whether to run benchmarks in parallel
            
        Returns:
            Benchmarking results
        """
        print(f"Running benchmarking analysis with {len(self.benchmark_strategies)} benchmark strategies...")
        
        # Run RL model
        print("Running RL model...")
        self.rl_results = self._run_rl_model(data, rl_model)
        
        if parallel and len(self.benchmark_strategies) > 1:
            # Run benchmark strategies in parallel
            self._run_benchmarks_parallel(data)
        else:
            # Run benchmark strategies sequentially
            self._run_benchmarks_sequential(data)
        
        # Compare results
        print("Comparing results...")
        self.comparison = self._compare_results()
        
        # Run statistical tests
        print("Running statistical tests...")
        self.statistical_tests = self._run_statistical_tests()
        
        # Save results
        self._save_results()
        
        return {
            'rl_results': self.rl_results,
            'benchmark_results': self.benchmark_results,
            'comparison': self.comparison,
            'statistical_tests': self.statistical_tests
        }
        
    def _run_rl_model(
        self,
        data: Dict[str, pd.DataFrame],
        rl_model: MultiTickerPPOLSTMPolicy
    ) -> Dict[str, Any]:
        """
        Run RL model.
        
        Args:
            data: Market data for all tickers
            rl_model: Trained RL model
            
        Returns:
            RL model results
        """
        # Initialize evaluator
        evaluator = MultiTickerEvaluator(self.config.get('evaluation', {}))
        
        # Evaluate model
        results = evaluator.evaluate_model(rl_model, data)
        
        return results
        
    def _run_benchmarks_sequential(self, data: Dict[str, pd.DataFrame]):
        """
        Run benchmark strategies sequentially.
        
        Args:
            data: Market data for all tickers
        """
        for strategy in self.benchmark_strategies:
            try:
                result = strategy.run(data)
                self.benchmark_results[strategy.name] = result
                print(f"Completed benchmark strategy: {strategy.name}")
            except Exception as e:
                print(f"Error running benchmark strategy {strategy.name}: {str(e)}")
                self.benchmark_results[strategy.name] = {'error': str(e)}
                
    def _run_benchmarks_parallel(self, data: Dict[str, pd.DataFrame]):
        """
        Run benchmark strategies in parallel.
        
        Args:
            data: Market data for all tickers
        """
        with ProcessPoolExecutor(max_workers=4) as executor:
            # Submit all strategies
            future_to_strategy = {
                executor.submit(strategy.run, data): strategy
                for strategy in self.benchmark_strategies
            }
            
            # Collect results
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    self.benchmark_results[strategy.name] = result
                    print(f"Completed benchmark strategy: {strategy.name}")
                except Exception as e:
                    print(f"Error running benchmark strategy {strategy.name}: {str(e)}")
                    self.benchmark_results[strategy.name] = {'error': str(e)}
                    
    def _compare_results(self) -> Dict[str, Any]:
        """
        Compare RL model results with benchmark strategies.
        
        Returns:
            Comparison results
        """
        comparison = {
            'metrics_comparison': {},
            'ranking': {},
            'outperformance': {}
        }
        
        # Get key metrics
        key_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        # Collect all results
        all_results = {'RL Model': self.rl_results}
        all_results.update(self.benchmark_results)
        
        # Compare metrics
        for metric in key_metrics:
            comparison['metrics_comparison'][metric] = {}
            
            for strategy_name, results in all_results.items():
                if 'error' not in results and 'metrics' in results and metric in results['metrics']:
                    comparison['metrics_comparison'][metric][strategy_name] = results['metrics'][metric]
        
        # Rank strategies by each metric
        for metric in key_metrics:
            if metric in comparison['metrics_comparison']:
                metric_values = comparison['metrics_comparison'][metric]
                
                # For metrics where higher is better
                if metric in ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 'profit_factor']:
                    ranking = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                # For metrics where lower is better
                elif metric in ['max_drawdown']:
                    ranking = sorted(metric_values.items(), key=lambda x: x[1])
                else:
                    # Default: assume higher is better
                    ranking = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                
                comparison['ranking'][metric] = ranking
        
        # Calculate outperformance
        rl_metrics = self.rl_results.get('metrics', {})
        
        for strategy_name, results in self.benchmark_results.items():
            if 'error' not in results and 'metrics' in results:
                strategy_metrics = results['metrics']
                
                comparison['outperformance'][strategy_name] = {}
                
                for metric in key_metrics:
                    if metric in rl_metrics and metric in strategy_metrics:
                        rl_value = rl_metrics[metric]
                        strategy_value = strategy_metrics[metric]
                        
                        if strategy_value != 0:
                            # For metrics where higher is better
                            if metric in ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 'profit_factor']:
                                outperformance = (rl_value - strategy_value) / abs(strategy_value)
                            # For metrics where lower is better
                            elif metric in ['max_drawdown']:
                                outperformance = (strategy_value - rl_value) / abs(strategy_value)
                            else:
                                # Default: assume higher is better
                                outperformance = (rl_value - strategy_value) / abs(strategy_value)
                            
                            comparison['outperformance'][strategy_name][metric] = outperformance
        
        return comparison
        
    def _run_statistical_tests(self) -> Dict[str, Any]:
        """
        Run statistical tests on performance differences.
        
        Returns:
            Statistical test results
        """
        statistical_tests = {
            't_tests': {},
            'mann_whitney_u': {},
            'bootstrap': {}
        }
        
        # Get returns
        rl_returns = self.rl_results.get('returns', [])
        
        for strategy_name, results in self.benchmark_results.items():
            if 'error' not in results and 'returns' in results:
                strategy_returns = results['returns']
                
                if len(rl_returns) > 1 and len(strategy_returns) > 1:
                    # T-test
                    t_stat, p_value = stats.ttest_ind(rl_returns, strategy_returns, equal_var=False)
                    statistical_tests['t_tests'][strategy_name] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < self.significance_level
                    }
                    
                    # Mann-Whitney U test
                    u_stat, p_value = stats.mannwhitneyu(rl_returns, strategy_returns, alternative='two-sided')
                    statistical_tests['mann_whitney_u'][strategy_name] = {
                        'u_statistic': u_stat,
                        'p_value': p_value,
                        'significant': p_value < self.significance_level
                    }
                    
                    # Bootstrap test
                    bootstrap_result = self._bootstrap_test(rl_returns, strategy_returns)
                    statistical_tests['bootstrap'][strategy_name] = bootstrap_result
        
        return statistical_tests
        
    def _bootstrap_test(
        self,
        returns1: List[float],
        returns2: List[float],
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Run bootstrap test for performance difference.
        
        Args:
            returns1: First set of returns
            returns2: Second set of returns
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap test results
        """
        # Calculate observed difference
        observed_diff = np.mean(returns1) - np.mean(returns2)
        
        # Combine returns
        combined_returns = returns1 + returns2
        n1 = len(returns1)
        n2 = len(returns2)
        n = len(combined_returns)
        
        # Bootstrap
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            resampled = np.random.choice(combined_returns, size=n, replace=True)
            
            # Split into two samples
            sample1 = resampled[:n1]
            sample2 = resampled[n1:]
            
            # Calculate difference
            diff = np.mean(sample1) - np.mean(sample2)
            bootstrap_diffs.append(diff)
        
        # Calculate p-value
        if observed_diff > 0:
            p_value = np.mean([diff >= observed_diff for diff in bootstrap_diffs])
        else:
            p_value = np.mean([diff <= observed_diff for diff in bootstrap_diffs])
        
        # Two-sided p-value
        p_value = min(p_value * 2, 1.0)
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'bootstrap_distribution': bootstrap_diffs
        }
        
    def _save_results(self):
        """Save benchmarking results."""
        # Save individual results
        results_path = os.path.join(self.output_dir, "benchmarking_results.json")
        
        # Convert to JSON-serializable format
        json_results = self._convert_to_json_serializable({
            'rl_results': self.rl_results,
            'benchmark_results': self.benchmark_results,
            'comparison': self.comparison,
            'statistical_tests': self.statistical_tests
        })
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Generate and save report
        report_path = os.path.join(self.output_dir, "benchmarking_report.html")
        self._generate_report(report_path)
        
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)
            
    def _generate_report(self, report_path: str):
        """
        Generate benchmarking report.
        
        Args:
            report_path: Path to save report
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmarking Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .positive { color: green; }
                .negative { color: red; }
                .significant { font-weight: bold; }
                .rank-1 { background-color: #gold; }
                .rank-2 { background-color: #silver; }
                .rank-3 { background-color: #cd7f32; }
            </style>
        </head>
        <body>
            <h1>Performance Benchmarking Report</h1>
            
            <h2>Summary</h2>
            <p>This report compares the performance of the RL trading model against various benchmark strategies.</p>
            
            <h2>Metrics Comparison</h2>
            {metrics_tables}
            
            <h2>Statistical Tests</h2>
            {statistical_tests_table}
            
            <h2>Rankings</h2>
            {rankings_table}
        </body>
        </html>
        """
        
        # Generate metrics tables
        metrics_tables = ""
        key_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        for metric in key_metrics:
            if metric in self.comparison['metrics_comparison']:
                metrics_tables += f"<h3>{metric.replace('_', ' ').title()}</h3>"
                metrics_tables += "<table>"
                metrics_tables += "<tr><th>Strategy</th><th>Value</th></tr>"
                
                for strategy_name, value in self.comparison['metrics_comparison'][metric].items():
                    metrics_tables += f"<tr><td>{strategy_name}</td><td>{value:.4f}</td></tr>"
                
                metrics_tables += "</table>"
        
        # Generate statistical tests table
        statistical_tests_table = "<table>"
        statistical_tests_table += "<tr><th>Benchmark Strategy</th><th>T-Test P-Value</th><th>Mann-Whitney U P-Value</th><th>Bootstrap P-Value</th></tr>"
        
        for strategy_name in self.benchmark_results.keys():
            if strategy_name in self.statistical_tests['t_tests']:
                t_test = self.statistical_tests['t_tests'][strategy_name]
                mann_whitney = self.statistical_tests['mann_whitney_u'][strategy_name]
                bootstrap = self.statistical_tests['bootstrap'][strategy_name]
                
                t_significant = "significant" if t_test['significant'] else "not significant"
                mw_significant = "significant" if mann_whitney['significant'] else "not significant"
                bs_significant = "significant" if bootstrap['significant'] else "not significant"
                
                statistical_tests_table += f"""
                    <tr>
                        <td>{strategy_name}</td>
                        <td class="{'significant' if t_test['significant'] else ''}">{t_test['p_value']:.4f} ({t_significant})</td>
                        <td class="{'significant' if mann_whitney['significant'] else ''}">{mann_whitney['p_value']:.4f} ({mw_significant})</td>
                        <td class="{'significant' if bootstrap['significant'] else ''}">{bootstrap['p_value']:.4f} ({bs_significant})</td>
                    </tr>
                """
        
        statistical_tests_table += "</table>"
        
        # Generate rankings table
        rankings_table = "<table>"
        rankings_table += "<tr><th>Rank</th><th>Total Return</th><th>Sharpe Ratio</th><th>Max Drawdown</th></tr>"
        
        # Get top 3 for each metric
        for rank in range(1, 4):
            rankings_table += f"<tr><td>{rank}</td>"
            
            for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
                if metric in self.comparison['ranking'] and len(self.comparison['ranking'][metric]) >= rank:
                    strategy_name = self.comparison['ranking'][metric][rank-1][0]
                    rankings_table += f"<td class='rank-{rank}'>{strategy_name}</td>"
                else:
                    rankings_table += "<td></td>"
            
            rankings_table += "</tr>"
        
        rankings_table += "</table>"
        
        # Format HTML
        html = html.format(
            metrics_tables=metrics_tables,
            statistical_tests_table=statistical_tests_table,
            rankings_table=rankings_table
        )
        
        # Save HTML report
        with open(report_path, 'w') as f:
            f.write(html)
            
    def get_best_benchmark(self, metric: str = 'sharpe_ratio') -> Optional[str]:
        """
        Get the best benchmark strategy for a given metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Name of the best benchmark strategy or None
        """
        if metric in self.comparison['ranking'] and self.comparison['ranking'][metric]:
            return self.comparison['ranking'][metric][0][0]
        
        return None
        
    def get_rl_rank(self, metric: str = 'sharpe_ratio') -> Optional[int]:
        """
        Get the rank of the RL model for a given metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Rank of the RL model or None
        """
        if metric in self.comparison['ranking']:
            for i, (strategy_name, _) in enumerate(self.comparison['ranking'][metric]):
                if strategy_name == 'RL Model':
                    return i + 1
        
        return None
        
    def is_significantly_better(self, strategy_name: str) -> Optional[bool]:
        """
        Check if RL model is significantly better than a benchmark strategy.
        
        Args:
            strategy_name: Name of the benchmark strategy
            
        Returns:
            True if significantly better, False if not, None if cannot determine
        """
        if strategy_name in self.statistical_tests['t_tests']:
            return self.statistical_tests['t_tests'][strategy_name]['significant']
        
        return None
        
    def get_outperformance_summary(self) -> Dict[str, float]:
        """
        Get summary of RL model outperformance over benchmarks.
        
        Returns:
            Summary of outperformance
        """
        summary = {}
        
        for strategy_name, outperformance in self.comparison['outperformance'].items():
            # Calculate average outperformance across metrics
            avg_outperformance = np.mean(list(outperformance.values()))
            summary[strategy_name] = avg_outperformance
        
        return summary


def create_benchmarking_suite(config: Dict[str, Any]) -> PerformanceBenchmarkingFramework:
    """
    Create a comprehensive benchmarking suite.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Performance benchmarking framework with comprehensive benchmark suite
    """
    # Initialize framework
    framework = PerformanceBenchmarkingFramework(config)
    
    # Add default benchmark strategies
    framework.add_default_benchmark_strategies()
    
    return framework