"""
Performance metrics and risk calculations for the RL trading system.

This module provides comprehensive metrics including Differential Sharpe ratio,
drawdown analysis, CVaR calculation, and other trading performance measures.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for trading performance metrics."""
    
    # Basic returns metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Risk metrics
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Trade-specific metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_trades: int
    
    # Additional metrics
    information_ratio: float
    beta: float
    alpha: float
    tracking_error: float


class DifferentialSharpe:
    """
    Online estimator for Differential Sharpe Ratio (DSR).
    
    This class implements the online DSR calculation as described in Moody & Saffell (2001).
    It maintains running estimates of mean and variance to provide a stationary
    reward signal for reinforcement learning.
    """
    
    def __init__(self, alpha: float = 0.01, eps: float = 1e-8):
        """
        Initialize the Differential Sharpe calculator.
        
        Args:
            alpha: Exponential moving average smoothing factor
            eps: Small constant to avoid division by zero
        """
        self.alpha = alpha
        self.eps = eps
        self.m = 0.0   # Running mean
        self.v = 0.0   # Running variance proxy
        self.count = 0
        
    def update(self, r: float) -> float:
        """
        Update the DSR estimate with a new return.
        
        Args:
            r: New return value
            
        Returns:
            Differential Sharpe ratio contribution
        """
        self.count += 1
        
        # Exponential moving average updates
        delta = r - self.m
        self.m += self.alpha * delta
        self.v = (1 - self.alpha) * (self.v + self.alpha * delta * delta)
        
        # Standard deviation with small epsilon to avoid division by zero
        sd = np.sqrt(max(self.v, self.eps))
        
        # Differential Sharpe (scaled incremental contribution)
        dsr = (r - self.m) / (sd + self.eps)
        
        return dsr
    
    def reset(self) -> None:
        """Reset the calculator state."""
        self.m = 0.0
        self.v = 0.0
        self.count = 0


class DrawdownAnalyzer:
    """Analyzer for drawdown calculations and tracking."""
    
    def __init__(self):
        self.peak = -np.inf
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_periods = []
        
    def update(self, equity: float, timestamp: Optional[datetime] = None) -> float:
        """
        Update drawdown analysis with new equity value.
        
        Args:
            equity: Current equity value
            timestamp: Optional timestamp for the equity value
            
        Returns:
            Current drawdown percentage
        """
        if equity > self.peak:
            self.peak = equity
            
        self.current_drawdown = (self.peak - equity) / max(self.peak, 1e-6)
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        return self.current_drawdown
    
    def get_drawdown_stats(self, equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive drawdown statistics from equity curve.
        
        Args:
            equity_curve: Series of equity values with datetime index
            
        Returns:
            Dictionary of drawdown statistics
        """
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'drawdown_duration_avg': 0.0,
                'drawdown_duration_max': 0.0,
                'time_to_recovery_avg': 0.0
            }
        
        # Calculate drawdown series
        drawdowns = (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
        
        # Find drawdown periods
        is_drawdown = drawdowns > 0
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)
        
        drawdown_periods = []
        for start_idx in drawdown_starts[drawdown_starts].index:
            end_idx = drawdown_ends.loc[start_idx:].index[0] if not drawdown_ends.loc[start_idx:].empty else equity_curve.index[-1]
            duration = (end_idx - start_idx).total_seconds() / 86400  # Convert to days
            max_dd = drawdowns.loc[start_idx:end_idx].max()
            drawdown_periods.append({
                'start': start_idx,
                'end': end_idx,
                'duration': duration,
                'max_drawdown': max_dd
            })
        
        # Calculate statistics
        if drawdown_periods:
            durations = [p['duration'] for p in drawdown_periods]
            max_drawdowns = [p['max_drawdown'] for p in drawdown_periods]
            
            return {
                'max_drawdown': max(max_drawdowns),
                'avg_drawdown': np.mean(max_drawdowns),
                'drawdown_duration_avg': np.mean(durations),
                'drawdown_duration_max': max(durations),
                'time_to_recovery_avg': np.mean(durations)
            }
        else:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'drawdown_duration_avg': 0.0,
                'drawdown_duration_max': 0.0,
                'time_to_recovery_avg': 0.0
            }


class RiskMetricsCalculator:
    """Calculator for various risk metrics."""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level for CVaR calculation
            
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        
        var = RiskMetricsCalculator.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else var
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns: Series of returns
            max_drawdown: Maximum drawdown
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0 or max_drawdown == 0:
            return 0.0
        
        annualized_return = np.sqrt(252) * returns.mean()
        return annualized_return / abs(max_drawdown)


class TradeAnalyzer:
    """Analyzer for trade-specific metrics."""
    
    def __init__(self):
        self.trades = []
        
    def add_trade(self, entry_time: datetime, exit_time: datetime, 
                  entry_price: float, exit_price: float, 
                  direction: str, quantity: int, costs: float = 0.0) -> None:
        """
        Add a trade to the analyzer.
        
        Args:
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            direction: 'long' or 'short'
            quantity: Quantity traded
            costs: Transaction costs
        """
        if direction not in ['long', 'short']:
            raise ValueError("Direction must be 'long' or 'short'")
        
        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * quantity - costs
        else:  # short
            pnl = (entry_price - exit_price) * quantity - costs
        
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'quantity': quantity,
            'pnl': pnl,
            'duration': (exit_time - entry_time).total_seconds() / 60,  # minutes
            'return_pct': pnl / (entry_price * quantity) * 100
        }
        
        self.trades.append(trade)
    
    def get_trade_metrics(self) -> Dict[str, float]:
        """
        Calculate trade-specific metrics.
        
        Returns:
            Dictionary of trade metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_duration': 0.0
            }
        
        df = pd.DataFrame(self.trades)
        
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        total_gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'profit_factor': total_gross_profit / total_gross_loss if total_gross_loss > 0 else np.inf,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
            'avg_duration': df['duration'].mean()
        }


def calculate_performance_metrics(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    benchmark_returns: Optional[pd.Series] = None
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from equity curve.
    
    Args:
        equity_curve: Series of equity values with datetime index
        risk_free_rate: Risk-free rate (annualized)
        benchmark_returns: Optional benchmark returns for alpha/beta calculation
        
    Returns:
        PerformanceMetrics object with all calculated metrics
    """
    if len(equity_curve) < 2:
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
            calmar_ratio=0.0, var_95=0.0, var_99=0.0, cvar_95=0.0,
            cvar_99=0.0, win_rate=0.0, profit_factor=0.0, avg_win=0.0,
            avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
            total_trades=0, information_ratio=0.0, beta=0.0,
            alpha=0.0, tracking_error=0.0
        )
    
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = RiskMetricsCalculator.calculate_sharpe_ratio(returns, risk_free_rate)
    sortino_ratio = RiskMetricsCalculator.calculate_sortino_ratio(returns, risk_free_rate)
    
    # Drawdown analysis
    drawdown_analyzer = DrawdownAnalyzer()
    for equity in equity_curve:
        drawdown_analyzer.update(equity)
    max_drawdown = drawdown_analyzer.max_drawdown
    calmar_ratio = RiskMetricsCalculator.calculate_calmar_ratio(returns, max_drawdown)
    
    # Value at Risk and CVaR
    var_95 = RiskMetricsCalculator.calculate_var(returns, 0.95)
    var_99 = RiskMetricsCalculator.calculate_var(returns, 0.99)
    cvar_95 = RiskMetricsCalculator.calculate_cvar(returns, 0.95)
    cvar_99 = RiskMetricsCalculator.calculate_cvar(returns, 0.99)
    
    # Trade metrics (placeholder - would need trade data)
    trade_metrics = {
        'win_rate': 0.0, 'profit_factor': 0.0, 'avg_win': 0.0,
        'avg_loss': 0.0, 'largest_win': 0.0, 'largest_loss': 0.0,
        'total_trades': 0
    }
    
    # Information ratio (relative to benchmark)
    information_ratio = 0.0
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        excess_returns = returns - benchmark_returns
        if excess_returns.std() > 0:
            information_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    # Alpha and beta (relative to benchmark)
    beta = 0.0
    alpha = 0.0
    tracking_error = 0.0
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        if benchmark_returns.std() > 0:
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            beta = covariance / (benchmark_returns.var() + 1e-8)
            alpha = annualized_return - (risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate))
            tracking_error = returns.std() * np.sqrt(252)
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        win_rate=trade_metrics['win_rate'],
        profit_factor=trade_metrics['profit_factor'],
        avg_win=trade_metrics['avg_win'],
        avg_loss=trade_metrics['avg_loss'],
        largest_win=trade_metrics['largest_win'],
        largest_loss=trade_metrics['largest_loss'],
        total_trades=trade_metrics['total_trades'],
        information_ratio=information_ratio,
        beta=beta,
        alpha=alpha,
        tracking_error=tracking_error
    )


def calculate_risk_metrics(returns: pd.Series, portfolio_value: float = 100000.0) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics for portfolio management.
    
    Args:
        returns: Series of portfolio returns
        portfolio_value: Current portfolio value
        
    Returns:
        Dictionary of risk metrics
    """
    if len(returns) == 0:
        return {
            'var_95': 0.0,
            'var_99': 0.0,
            'cvar_95': 0.0,
            'cvar_99': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'beta': 0.0,
            'alpha': 0.0,
            'tracking_error': 0.0,
            'information_ratio': 0.0
        }
    
    # Basic risk metrics
    var_95 = RiskMetricsCalculator.calculate_var(returns, 0.95)
    var_99 = RiskMetricsCalculator.calculate_var(returns, 0.99)
    cvar_95 = RiskMetricsCalculator.calculate_cvar(returns, 0.95)
    cvar_99 = RiskMetricsCalculator.calculate_cvar(returns, 0.99)
    volatility = returns.std() * np.sqrt(252)
    
    # Drawdown analysis
    equity_curve = portfolio_value * (1 + returns.cumsum())
    drawdown_analyzer = DrawdownAnalyzer()
    for equity in equity_curve:
        drawdown_analyzer.update(equity)
    max_drawdown = drawdown_analyzer.max_drawdown
    
    # Risk-adjusted ratios
    sharpe_ratio = RiskMetricsCalculator.calculate_sharpe_ratio(returns)
    sortino_ratio = RiskMetricsCalculator.calculate_sortino_ratio(returns)
    calmar_ratio = RiskMetricsCalculator.calculate_calmar_ratio(returns, max_drawdown)
    
    # Additional metrics (relative to S&P 500 as benchmark)
    benchmark_returns = returns * 0.8  # Simplified benchmark
    if len(returns) == len(benchmark_returns):
        excess_returns = returns - benchmark_returns
        if excess_returns.std() > 0:
            information_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        else:
            information_ratio = 0.0
        
        if benchmark_returns.std() > 0:
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            beta = covariance / (benchmark_returns.var() + 1e-8)
            alpha = returns.mean() * 252 - (0.02 + beta * (benchmark_returns.mean() * 252 - 0.02))
            tracking_error = returns.std() * np.sqrt(252)
        else:
            beta = 0.0
            alpha = 0.0
            tracking_error = 0.0
    else:
        information_ratio = 0.0
        beta = 0.0
        alpha = 0.0
        tracking_error = 0.0
    
    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'beta': beta,
        'alpha': alpha,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio
    }