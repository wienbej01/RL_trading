
"""
Multi-ticker performance tracking for RL trading system.

This module provides comprehensive performance tracking capabilities for
multi-ticker RL trading strategies, including portfolio metrics, risk analysis,
and performance attribution.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..utils.logging import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceMetrics:
    """
    Container for performance metrics.
    """
    
    def __init__(self):
        """Initialize performance metrics container."""
        # Return metrics
        self.total_return = 0.0
        self.annual_return = 0.0
        self.monthly_returns = []
        self.daily_returns = []
        
        # Risk metrics
        self.annual_volatility = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.var_95 = 0.0
        self.var_99 = 0.0
        self.cvar_95 = 0.0
        self.cvar_99 = 0.0
        
        # Trade metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        self.avg_trade_duration = 0.0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.recovery_factor = 0.0
        self.payoff_ratio = 0.0
        
        # Long/short metrics
        self.num_long_trades = 0
        self.num_short_trades = 0
        self.avg_pnl_long = 0.0
        self.avg_pnl_short = 0.0
        self.avg_duration_min = 0.0
        self.avg_duration_long_min = 0.0
        self.avg_duration_short_min = 0.0
        self.long_win_rate = 0.0
        self.short_win_rate = 0.0
        self.trades_per_day = 0.0
        
        # Cost metrics
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_transaction_costs = 0.0
        
        # Equity metrics
        self.final_equity = 0.0
        self.peak_equity = 0.0
        
        # Ticker-specific metrics
        self.ticker_metrics = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary.
        
        Returns:
            Dictionary representation of metrics
        """
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'annual_volatility': self.annual_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_trade_duration': self.avg_trade_duration,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'recovery_factor': self.recovery_factor,
            'payoff_ratio': self.payoff_ratio,
            'num_long_trades': self.num_long_trades,
            'num_short_trades': self.num_short_trades,
            'avg_pnl_long': self.avg_pnl_long,
            'avg_pnl_short': self.avg_pnl_short,
            'avg_duration_min': self.avg_duration_min,
            'avg_duration_long_min': self.avg_duration_long_min,
            'avg_duration_short_min': self.avg_duration_short_min,
            'long_win_rate': self.long_win_rate,
            'short_win_rate': self.short_win_rate,
            'trades_per_day': self.trades_per_day,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_transaction_costs': self.total_transaction_costs,
            'final_equity': self.final_equity,
            'peak_equity': self.peak_equity,
            'ticker_metrics': self.ticker_metrics
        }


class TickerPerformance:
    """
    Container for ticker-specific performance metrics.
    """
    
    def __init__(self, ticker: str):
        """
        Initialize ticker performance container.
        
        Args:
            ticker: Ticker symbol
        """
        self.ticker = ticker
        self.total_return = 0.0
        self.annual_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        self.num_long_trades = 0
        self.num_short_trades = 0
        self.avg_pnl_long = 0.0
        self.avg_pnl_short = 0.0
        self.long_win_rate = 0.0
        self.short_win_rate = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_transaction_costs = 0.0
        self.equity_curve = []
        self.trade_history = []
        self.position_history = []
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ticker metrics to dictionary.
        
        Returns:
            Dictionary representation of ticker metrics
        """
        return {
            'ticker': self.ticker,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'num_long_trades': self.num_long_trades,
            'num_short_trades': self.num_short_trades,
            'avg_pnl_long': self.avg_pnl_long,
            'avg_pnl_short': self.avg_pnl_short,
            'long_win_rate': self.long_win_rate,
            'short_win_rate': self.short_win_rate,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_transaction_costs': self.total_transaction_costs
        }


class PerformanceTracker:
    """
    Multi-ticker performance tracker for RL trading system.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        output_dir: str = "performance_results",
        risk_free_rate: float = 0.02
    ):
        """
        Initialize performance tracker.
        
        Args:
            initial_capital: Initial capital for the portfolio
            output_dir: Directory to save performance results
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.risk_free_rate = risk_free_rate
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Ticker-specific performance
        self.ticker_performance = {}
        
        # Equity curve
        self.equity_curve = []
        self.timestamps = []
        
        # Trade history
        self.trade_history = []
        
        # Position history
        self.position_history = []
        
        # Daily returns
        self.daily_returns = []
        
        # Drawdown history
        self.drawdown_history = []
        
        # Regime-specific performance
        self.regime_performance = {}
        
        # Initialize with starting capital
        self.add_equity_point(datetime.now(), self.initial_capital)
        
    def add_equity_point(self, timestamp: datetime, equity: float) -> None:
        """
        Add equity point to the tracker.
        
        Args:
            timestamp: Timestamp of the equity point
            equity: Equity value
        """
        self.timestamps.append(timestamp)
        self.equity_curve.append(equity)
        self.current_capital = equity
        
        # Calculate daily return if we have previous data
        if len(self.equity_curve) > 1:
            daily_return = (equity - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
            
        # Calculate drawdown
        peak_equity = max(self.equity_curve)
        drawdown = (peak_equity - equity) / peak_equity
        self.drawdown_history.append(drawdown)
        
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add trade to the tracker.
        
        Args:
            trade: Trade dictionary with trade details
        """
        self.trade_history.append(trade)
        
        # Update ticker-specific performance
        ticker = trade.get('ticker')
        if ticker not in self.ticker_performance:
            self.ticker_performance[ticker] = TickerPerformance(ticker)
            
        self.ticker_performance[ticker].trade_history.append(trade)
        
    def add_position(self, position: Dict[str, Any]) -> None:
        """
        Add position to the tracker.
        
        Args:
            position: Position dictionary with position details
        """
        self.position_history.append(position)
        
        # Update ticker-specific performance
        ticker = position.get('ticker')
        if ticker not in self.ticker_performance:
            self.ticker_performance[ticker] = TickerPerformance(ticker)
            
        self.ticker_performance[ticker].position_history.append(position)
        
    def add_regime_performance(self, regime: str, metrics: Dict[str, float]) -> None:
        """
        Add regime-specific performance metrics.
        
        Args:
            regime: Regime identifier
            metrics: Performance metrics for the regime
        """
        if regime not in self.regime_performance:
            self.regime_performance[regime] = []
            
        self.regime_performance[regime].append(metrics)
        
    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            PerformanceMetrics object with calculated metrics
        """
        if len(self.equity_curve) < 2:
            return self.metrics
            
        # Convert to pandas Series for easier calculations
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        
        # Calculate return metrics
        self.metrics.total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Calculate annualized return
        days_elapsed = (self.timestamps[-1] - self.timestamps[0]).days
        if days_elapsed > 0:
            years_elapsed = days_elapsed / 365.25
            self.metrics.annual_return = (1 + self.metrics.total_return) ** (1 / years_elapsed) - 1
        else:
            self.metrics.annual_return = 0.0
            
        # Calculate daily returns
        returns = equity_series.pct_change().dropna()
        self.metrics.daily_returns = returns.tolist()
        
        # Calculate monthly returns
        monthly_returns = equity_series.resample('M').last().pct_change().dropna()
        self.metrics.monthly_returns = monthly_returns.tolist()
        
        # Calculate risk metrics
        if len(returns) > 0:
            self.metrics.annual_volatility = returns.std() * np.sqrt(252)
            
            # Calculate Sharpe ratio
            if self.metrics.annual_volatility > 0:
                excess_return = self.metrics.annual_return - self.risk_free_rate
                self.metrics.sharpe_ratio = excess_return / self.metrics.annual_volatility
            else:
                self.metrics.sharpe_ratio = 0.0
                
            # Calculate Sortino ratio (using downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                if downside_deviation > 0:
                    excess_return = self.metrics.annual_return - self.risk_free_rate
                    self.metrics.sortino_ratio = excess_return / downside_deviation
                else:
                    self.metrics.sortino_ratio = 0.0
            else:
                self.metrics.sortino_ratio = 0.0
                
            # Calculate Calmar ratio
            if self.metrics.max_drawdown > 0:
                self.metrics.calmar_ratio = self.metrics.annual_return / self.metrics.max_drawdown
            else:
                self.metrics.calmar_ratio = 0.0
                
            # Calculate VaR and CVaR
            self.metrics.var_95 = returns.quantile(0.05)
            self.metrics.var_99 = returns.quantile(0.01)
            self.metrics.cvar_95 = returns[returns <= self.metrics.var_95].mean()
            self.metrics.cvar_99 = returns[returns <= self.metrics.var_99].mean()
            
        # Calculate drawdown metrics
        self.metrics.max_drawdown = max(self.drawdown_history)
        self.metrics.current_drawdown = self.drawdown_history[-1]
        
        # Calculate equity metrics
        self.metrics.final_equity = self.current_capital
        self.metrics.peak_equity = max(self.equity_curve)
        
        # Calculate trade metrics
        self._calculate_trade_metrics()
        
        # Calculate ticker-specific metrics
        self._calculate_ticker_metrics()
        
        return self.metrics
        
    def _calculate_trade_metrics(self) -> None:
        """Calculate trade-based metrics."""
        if not self.trade_history:
            return
            
        # Convert to DataFrame for easier calculations
        trades_df = pd.DataFrame(self.trade_history)
        
        # Basic trade metrics
        self.metrics.total_trades = len(trades_df)
        
        if self.metrics.total_trades > 0:
            # Calculate P&L for each trade
            trades_df['pnl'] = trades_df['exit_price'] * trades_df['quantity'] - trades_df['entry_price'] * trades_df['quantity']
            trades_df['pnl'] -= trades_df.get('commission', 0) + trades_df.get('slippage', 0)
            
            # Winning and losing trades
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            self.metrics.winning_trades = len(winning_trades)
            self.metrics.losing_trades = len(losing_trades)
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
            
            # Average win and loss
            if len(winning_trades) > 0:
                self.metrics.avg_win = winning_trades['pnl'].mean()
            else:
                self.metrics.avg_win = 0.0
                
            if len(losing_trades) > 0:
                self.metrics.avg_loss = losing_trades['pnl'].mean()
            else:
                self.metrics.avg_loss = 0.0
                
            # Profit factor
            if self.metrics.avg_loss != 0:
                self.metrics.profit_factor = abs(self.metrics.avg_win / self.metrics.avg_loss) * (self.metrics.winning_trades / self.metrics.losing_trades)
            else:
                self.metrics.profit_factor = float('inf') if self.metrics.avg_win > 0 else 0.0
                
            # Largest win and loss
            self.metrics.largest_win = trades_df['pnl'].max()
            self.metrics.largest_loss = trades_df['pnl'].min()
            
            # Long/short metrics
            long_trades = trades_df[trades_df['direction'] == 'LONG']
            short_trades = trades_df[trades_df['direction'] == 'SHORT']
            
            self.metrics.num_long_trades = len(long_trades)
            self.metrics.num_short_trades = len(short_trades)
            
            if len(long_trades) > 0:
                self.metrics.avg_pnl_long = long_trades['pnl'].mean()
                long_winning_trades = long_trades[long_trades['pnl'] > 0]
                self.metrics.long_win_rate = len(long_winning_trades) / len(long_trades)
            else:
                self.metrics.avg_pnl_long = 0.0
                self.metrics.long_win_rate = 0.0
                
            if len(short_trades) > 0:
                self.metrics.avg_pnl_short = short_trades['pnl'].mean()
                short_winning_trades = short_trades[short_trades['pnl'] > 0]
                self.metrics.short_win_rate = len(short_winning_trades) / len(short_trades)
            else:
                self.metrics.avg_pnl_short = 0.0
                self.metrics.short_win_rate = 0.0
                
            # Trade duration
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                trades_df['duration'] = (pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])).dt.total_seconds() / 60  # in minutes
                self.metrics.avg_trade_duration = trades_df['duration'].mean()
                
                # Long/short duration
                if len(long_trades) > 0:
                    self.metrics.avg_duration_long_min = long_trades['duration'].mean()
                else:
                    self.metrics.avg_duration_long_min = 0.0
                    
                if len(short_trades) > 0:
                    self.metrics.avg_duration_short_min = short_trades['duration'].mean()
                else:
                    self.metrics.avg_duration_short_min = 0.0
                    
            # Consecutive wins/losses
            trades_df['win'] = trades_df['pnl'] > 0
            trades_df['loss'] = trades_df['pnl'] < 0
            
            # Calculate consecutive wins
            consecutive_wins = 0
            max_consecutive_wins = 0
            for win in trades_df['win']:
                if win:
                    consecutive_wins += 1
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_wins = 0
                    
            self.metrics.max_consecutive_wins = max_consecutive_wins
            
            # Calculate consecutive losses
            consecutive_losses = 0
            max_consecutive_losses = 0
            for loss in trades_df['loss']:
                if loss:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
                    
            self.metrics.max_consecutive_losses = max_consecutive_losses
            
            # Recovery factor
            if self.metrics.max_drawdown > 0:
                self.metrics.recovery_factor = self.metrics.total_return / self.metrics.max_drawdown
            else:
                self.metrics.recovery_factor = 0.0
                
            # Payoff ratio
            if self.metrics.avg_loss != 0:
                self.metrics.payoff_ratio = abs(self.metrics.avg_win / self.metrics.avg_loss)
            else:
                self.metrics.payoff_ratio = float('inf') if self.metrics.avg_win > 0 else 0.0
                
            # Trades per day
            if self.timestamps:
                days_elapsed = (self.timestamps[-1] - self.timestamps[0]).days
                if days_elapsed > 0:
                    self.metrics.trades_per_day = self.metrics.total_trades / days_elapsed
                    
            # Cost metrics
            self.metrics.total_commission = trades_df.get('commission', 0).sum()
            self.metrics.total_slippage = trades_df.get('slippage', 0).sum()
            self.metrics.total_transaction_costs = self.metrics.total_commission + self.metrics.total_slippage
            
    def _calculate_ticker_metrics(self) -> None:
        """Calculate ticker-specific metrics."""
        for ticker, ticker_perf in self.ticker_performance.items():
            # Get trades for this ticker
            ticker_trades = [t for t in self.trade_history if t.get('ticker') == ticker]
            
            if not ticker_trades:
                continue
                
            # Convert to DataFrame
            trades_df = pd.DataFrame(ticker_trades)
            
            # Calculate P&L for each trade
            trades_df['pnl'] = trades_df['exit_price'] * trades_df['quantity'] - trades_df['entry_price'] * trades_df['quantity']
            trades_df['pnl'] -= trades_df.get('commission', 0) + trades_df.get('slippage', 0)
            
            # Basic metrics
            ticker_perf.total_trades = len(trades_df)
            
            # Winning and losing trades
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            ticker_perf.winning_trades = len(winning_trades)
            ticker_perf.losing_trades = len(losing_trades)
            ticker_perf.win_rate = ticker_perf.winning_trades / ticker_perf.total_trades if ticker_perf.total_trades > 0 else 0.0
            
            # Average win and loss
            if len(winning_trades) > 0:
                ticker_perf.avg_win = winning_trades['pnl'].mean()
            else:
                ticker_perf.avg_win = 0.0
                
            if len(losing_trades) > 0:
                ticker_perf.avg_loss = losing_trades['pnl'].mean()
            else:
                ticker_perf.avg_loss = 0.0
                
            # Profit factor
            if ticker_perf.avg_loss != 0:
                ticker_perf.profit_factor = abs(ticker_perf.avg_win / ticker_perf.avg_loss) * (ticker_perf.winning_trades / ticker_perf.losing_trades)
            else:
                ticker_perf.profit_factor = float('inf') if ticker_perf.avg_win > 0 else 0.0
                
            # Long/short metrics
            long_trades = trades_df[trades_df['direction'] == 'LONG']
            short_trades = trades_df[trades_df['direction'] == 'SHORT']
            
            ticker_perf.num_long_trades = len(long_trades)
            ticker_perf.num_short_trades = len(short_trades)
            
            if len(long_trades) > 0:
                ticker_perf.avg_pnl_long = long_trades['pnl'].mean()
                long_winning_trades = long_trades[long_trades['pnl'] > 0]
                ticker_perf.long_win_rate = len(long_winning_trades) / len(long_trades)
            else:
                ticker_perf.avg_pnl_long = 0.0
                ticker_perf.long_win_rate = 0.0
                
            if len(short_trades) > 0:
                ticker_perf.avg_pnl_short = short_trades['pnl'].mean()
                short_winning_trades = short_trades[short_trades['pnl'] > 0]
                ticker_perf.short_win_rate = len(short_winning_trades) / len(short_trades)
            else:
                ticker_perf.avg_pnl_short = 0.0
                ticker_perf.short_win_rate = 0.0
                
            # Cost metrics
            ticker_perf.total_commission = trades_df.get('commission', 0).sum()
            ticker_perf.total_slippage = trades_df.get('slippage', 0).sum()
            ticker_perf.total_transaction_costs = ticker_perf.total_commission + ticker_perf.total_slippage
            
            # Calculate total return for this ticker
            total_pnl = trades_df['pnl'].sum()
            # Estimate initial allocation (equal weight by default)
            initial_allocation = self.initial_capital / len(self.ticker_performance)
            ticker_perf.total_return = total_pnl / initial_allocation
            
            # Calculate annualized return
            if self.timestamps:
                days_elapsed = (self.timestamps[-1] - self.timestamps[0]).days
                if days_elapsed > 0:
                    years_elapsed = days_elapsed / 365.25
                    ticker_perf.annual_return = (1 + ticker_perf.total_return) ** (1 / years_elapsed) - 1
                else:
                    ticker_perf.annual_return = 0.0
                    
            # Calculate Sharpe ratio
            if len(trades_df) > 1:
                returns = trades_df['pnl'] / initial_allocation
                if returns.std() > 0:
                    ticker_perf.sharpe_ratio = (returns.mean() - self.risk_free_rate / 252) / returns.std() * np.sqrt(252)
                else:
                    ticker_perf.sharpe_ratio = 0.0
                    
            # Calculate max drawdown for this ticker
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                # Create equity curve for this ticker
                trades_df = trades_df.sort_values('entry_time')
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                trades_df['equity'] = initial_allocation + trades_df['cumulative_pnl']
                
                # Calculate drawdown
                trades_df['peak'] = trades_df['equity'].cummax()
                trades_df['drawdown'] = (trades_df['peak'] - trades_df['equity']) / trades_df['peak']
                ticker_perf.max_drawdown = trades_df['drawdown'].max()
                
            # Add to overall metrics
            self.metrics.ticker_metrics[ticker] = ticker_perf.to_dict()
            
    def save_results(self, filepath: str) -> None:
        """
        Save performance results to file.
        
        Args:
            filepath: Path to save results
        """
        # Calculate metrics before saving
        self.calculate_metrics()
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'drawdown_history': self.drawdown_history,
            'trade_history': self.trade_history,
            'position_history': self.position_history,
            'metrics': self.metrics.to_dict(),
            'ticker_performance': {ticker: perf.to_dict() for ticker, perf in self.ticker_performance.items()},
            'regime_performance': self.regime_performance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
            
        logger.info(f"Performance results saved to {filepath}")
        
    @classmethod
    def load_results(cls, filepath: str) -> 'PerformanceTracker':
        """
        Load performance results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            PerformanceTracker instance with loaded results
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
            
        # Create tracker
        tracker = cls(
            initial_capital=results['initial_capital'],
            output_dir=str(Path(filepath).parent)
        )
        
        # Load data
        tracker.current_capital = results['final_capital']
        tracker.timestamps = [datetime.fromisoformat(ts) for ts in results['timestamps']]
        tracker.equity_curve = results['equity_curve']
        tracker.daily_returns = results['daily_returns']
        tracker.drawdown_history = results['drawdown_history']
        tracker.trade_history = results['trade_history']
        tracker.position_history = results['position_history']
        
        # Load metrics
        metrics_dict = results['metrics']
        tracker.metrics.total_return = metrics_dict['total_return']
        tracker.metrics.annual_return = metrics_dict['annual_return']
        tracker.metrics.annual_volatility = metrics_dict['annual_volatility']
        tracker.metrics.sharpe_ratio = metrics_dict['sharpe_ratio']
        tracker.metrics.sortino_ratio = metrics_dict['sortino_ratio']
        tracker.metrics.calmar_ratio = metrics_dict['calmar_ratio']
        tracker.metrics.max_drawdown = metrics_dict['max_drawdown']
        tracker.metrics.current_drawdown = metrics_dict['current_drawdown']
        tracker.metrics.var_95 = metrics_dict['var_95']
        tracker.metrics.var_99 = metrics_dict['var_99']
        tracker.metrics.cvar_95 = metrics_dict['cvar_95']
        tracker.metrics.cvar_99 = metrics_dict['cvar_99']
        tracker.metrics.total_trades = metrics_dict['total_trades']
        tracker.metrics.winning_trades = metrics_dict['winning_trades']
        tracker.metrics.losing_trades = metrics_dict['losing_trades']
        tracker.metrics.win_rate = metrics_dict['win_rate']
        tracker.metrics.avg_win = metrics_dict['avg_win']
        tracker.metrics.avg_loss = metrics_dict['avg_loss']
        tracker.metrics.profit_factor = metrics_dict['profit_factor']
        tracker.metrics.largest_win = metrics_dict['largest_win']
        tracker.metrics.largest_loss = metrics_dict['largest_loss']
        tracker.metrics.avg_trade_duration = metrics_dict['avg_trade_duration']
        tracker.metrics.max_consecutive_wins = metrics_dict['max_consecutive_wins']
        tracker.metrics.max_consecutive_losses = metrics_dict['max_consecutive_losses']
        tracker.metrics.recovery_factor = metrics_dict['recovery_factor']
        tracker.metrics.payoff_ratio = metrics_dict['payoff_ratio']
        tracker.metrics.num_long_trades = metrics_dict['num_long_trades']
        tracker.metrics.num_short_trades = metrics_dict['num_short_trades']
        tracker.metrics.avg_pnl_long = metrics_dict['avg_pnl_long']
        tracker.metrics.avg_pnl_short = metrics_dict['avg_pnl_short']
        tracker.metrics.avg_duration_min = metrics_dict['avg_duration_min']
        tracker.metrics.avg_duration_long_min = metrics_dict['avg_duration_long_min']
        tracker.metrics.avg_duration_short_min = metrics_dict['avg_duration_short_min']
        tracker.metrics.long_win_rate = metrics_dict['long_win_rate']
        tracker.metrics.short_win_rate = metrics_dict['short_win_rate']
        tracker.metrics.trades_per_day = metrics_dict['trades_per_day']
        tracker.metrics.total_commission = metrics_dict['total_commission']
        tracker.metrics.total_slippage = metrics_dict['total_slippage']
        tracker.metrics.total_transaction_costs = metrics_dict['total_transaction_costs']
        tracker.metrics.final_equity = metrics_dict['final_equity']
        tracker.metrics.peak_equity = metrics_dict['peak_equity']
        tracker.metrics.ticker_metrics = metrics_dict['ticker_metrics']
        
        # Load ticker performance
        for ticker, ticker_perf_dict in results['ticker_performance'].items():
            ticker_perf = TickerPerformance(ticker)
            ticker_perf.total_return = ticker_perf_dict['total_return']
            ticker_perf.annual_return = ticker_perf_dict['annual_return']
            ticker_perf.sharpe_ratio = ticker_perf_dict['sharpe_ratio']
            ticker_perf.max_drawdown = ticker_perf_dict['max_drawdown']
            ticker_perf.total_trades = ticker_perf_dict['total_trades']
            ticker_perf.winning_trades = ticker_perf_dict['winning_trades']
            ticker_perf.win_rate = ticker_perf_dict['win_rate']
            ticker_perf.avg_win = ticker_perf_dict['avg_win']
            ticker_perf.avg_loss = ticker_perf_dict['avg_loss']
            ticker_perf.profit_factor = ticker_perf_dict['profit_factor']
            ticker_perf.num_long_trades = ticker_perf_dict['num_long_trades']
            ticker_perf.num_short_trades = ticker_perf_dict['num_short_trades']
            ticker_perf.avg_pnl_long = ticker_perf_dict['avg_pnl_long']
            ticker_perf.avg_pnl_short = ticker_perf_dict['avg_pnl_short']
            ticker_perf.long_win_rate = ticker_perf_dict['long_win_rate']
            ticker_perf.short_win_rate = ticker_perf_dict['short_win_rate']
            ticker_perf.total_commission = ticker_perf_dict['total_commission']
            ticker_perf.total_slippage = ticker_perf_dict['total_slippage']
            ticker_perf.total_transaction_costs = ticker_perf_dict['total_transaction_costs']
            
            tracker.ticker_performance[ticker] = ticker_perf
            
        # Load regime performance
        tracker.regime_performance = results['regime_performance']
        
        return tracker
        
    def generate_report(self, output_path: str) -> None:
        """
        Generate comprehensive performance report.
        
        Args:
            output_path: Path to save report
        """
        # Calculate metrics before generating report
        self.calculate_metrics()
        
        # Create report
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_return': self.metrics.total_return,
                'annual_return': self.metrics.annual_return,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'max_drawdown': self.metrics.max_drawdown,
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate
            },
            'detailed_metrics': self.metrics.to_dict(),
            'ticker_performance': {ticker: perf.to_dict() for ticker, perf in self.ticker_performance.items()},
            'regime_performance': self.regime_performance
        }
        
        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Performance report saved to {output_path}")
        
    def create_plots(self, output_dir: str) -> None:
        """
        Create performance plots.
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics before creating plots
        self.calculate_metrics()
        
        # Create equity curve plot
        self._plot_equity_curve(output_dir / "equity_curve.png")
        
        # Create drawdown plot
        self._plot_drawdown(output_dir / "drawdown.png")
        
        # Create returns distribution plot
        self._plot_returns_distribution(output_dir / "returns_distribution.png")
        
        # Create monthly returns heatmap
        self._plot_monthly_returns_heatmap(output_dir / "monthly_returns_heatmap.png")
        
        # Create rolling metrics plot
        self._plot_rolling_metrics(output_dir / "rolling_metrics.png")
        
        # Create ticker performance plot
        self._plot_ticker_performance(output_dir / "ticker_performance.png")
        
        # Create regime performance plot
        self._plot_regime_performance(output_dir / "regime_performance.png")
        
        logger.info(f"Performance plots saved to {output_dir}")
        
    def _plot_equity_curve(self, output_path: Path) -> None:
        """
        Plot equity curve.
        
        Args:
            output_path: Path to save plot
        """
        plt.figure(figsize=(12, 6))
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity_curve
        })
        equity_df.set_index('timestamp', inplace=True)
        
        # Plot equity curve
        plt.plot(equity_df.index, equity_df['equity'], label='Portfolio Equity', linewidth=2)
        
        # Add initial capital line
        plt.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Add peak equity line
        plt.axhline(y=self.metrics.peak_equity, color='green', linestyle='--', alpha=0.5, label='Peak Equity')
        
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
