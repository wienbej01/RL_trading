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


def calculate_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Calculate drawdown from equity curve."""
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)
        return_numpy = True
    else:
        return_numpy = False
    
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    # Return numpy array if input was numpy array
    if return_numpy:
        return drawdown.values
    
    return drawdown


def calculate_max_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> float:
    """Calculate maximum drawdown from equity curve."""
    drawdown = calculate_drawdown(equity_curve)
    return float(np.min(drawdown))


def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], 
                          risk_free_rate: float = 0.0, 
                          periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
    
    # Check for infinite values
    if np.any(np.isinf(returns)):
        raise ValueError("Returns array contains infinite values")
    
    if returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return float(excess_returns.mean() / returns.std() * np.sqrt(periods_per_year))


def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray], 
                           risk_free_rate: float = 0.0, 
                           periods_per_year: int = 252) -> float:
    """Calculate Sortino ratio."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    return float(excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year))


def calculate_calmar_ratio(returns: Union[pd.Series, np.ndarray], 
                          equity_curve: Union[pd.Series, np.ndarray] = None,
                          periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
    
    annualized_return = float((1 + returns.mean()) ** periods_per_year - 1)
    
    if equity_curve is None:
        equity_curve = (1 + returns).cumprod()
    
    max_dd = calculate_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return 0.0
    
    return float(annualized_return / abs(max_dd))


def calculate_alpha(returns: Union[pd.Series, np.ndarray], 
                   benchmark_returns: Union[pd.Series, np.ndarray],
                   risk_free_rate: float = 0.0,
                   periods_per_year: int = 252) -> float:
    """Calculate alpha relative to benchmark."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)
    
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")
    
    # Remove any NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns_clean = returns[mask]
    benchmark_clean = benchmark_returns[mask]
    
    if len(returns_clean) < 2:
        return 0.0
    
    beta = calculate_beta(returns_clean, benchmark_clean)
    
    # Annualized returns
    portfolio_return = float((1 + returns_clean.mean()) ** periods_per_year - 1)
    benchmark_return = float((1 + benchmark_clean.mean()) ** periods_per_year - 1)
    
    # Alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    return alpha


def calculate_information_ratio(returns: Union[pd.Series, np.ndarray], 
                              benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate Information ratio."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)
    
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")
    
    # Remove any NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns_clean = returns[mask]
    benchmark_clean = benchmark_returns[mask]
    
    if len(returns_clean) < 2:
        return 0.0
    
    excess_returns = returns_clean - benchmark_clean
    tracking_error = excess_returns.std()
    
    if tracking_error == 0:
        return 0.0
    
    return float(excess_returns.mean() / tracking_error * np.sqrt(252))


def calculate_tracking_error(returns: Union[pd.Series, np.ndarray], 
                           benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate tracking error."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)
    
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")
    
    # Remove any NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns_clean = returns[mask]
    benchmark_clean = benchmark_returns[mask]
    
    if len(returns_clean) < 2:
        return 0.0
    
    excess_returns = returns_clean - benchmark_clean
    return float(excess_returns.std() * np.sqrt(252))


def calculate_upside_capture(returns: Union[pd.Series, np.ndarray], 
                           benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate upside capture ratio."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)
    
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")
    
    # Remove any NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns_clean = returns[mask]
    benchmark_clean = benchmark_returns[mask]
    
    if len(returns_clean) < 1:
        return 0.0
    
    # Only consider periods where benchmark is positive
    positive_benchmark_mask = benchmark_clean > 0
    
    if not positive_benchmark_mask.any():
        return 0.0
    
    portfolio_positive_returns = returns_clean[positive_benchmark_mask]
    benchmark_positive_returns = benchmark_clean[positive_benchmark_mask]
    
    if benchmark_positive_returns.mean() == 0:
        return 0.0
    
    return float(portfolio_positive_returns.mean() / benchmark_positive_returns.mean())


def calculate_downside_capture(returns: Union[pd.Series, np.ndarray], 
                             benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate downside capture ratio."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)
    
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")
    
    # Remove any NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns_clean = returns[mask]
    benchmark_clean = benchmark_returns[mask]
    
    if len(returns_clean) < 1:
        return 0.0
    
    # Only consider periods where benchmark is negative
    negative_benchmark_mask = benchmark_clean < 0
    
    if not negative_benchmark_mask.any():
        return 0.0
    
    portfolio_negative_returns = returns_clean[negative_benchmark_mask]
    benchmark_negative_returns = benchmark_clean[negative_benchmark_mask]
    
    if benchmark_negative_returns.mean() == 0:
        return 0.0
    
    return float(portfolio_negative_returns.mean() / benchmark_negative_returns.mean())


def calculate_var(returns: Union[pd.Series, np.ndarray], confidence: float = 0.05) -> float:
    """Calculate Value at Risk (VaR)."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
    
    return float(np.percentile(returns, confidence * 100))


def calculate_cvar(returns: Union[pd.Series, np.ndarray], confidence: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR)."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
    
    var = calculate_var(returns, confidence)
    tail_returns = returns[returns <= var]
    return float(np.mean(tail_returns))


def calculate_beta(returns: Union[pd.Series, np.ndarray], 
                  benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate beta relative to benchmark."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)
    
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have same length")
    
    # Check if arrays are identical (same object or same values)
    if returns is benchmark_returns or np.array_equal(returns.values, benchmark_returns.values):
        return 1.0
    
    # Remove any NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns_clean = returns[mask]
    benchmark_clean = benchmark_returns[mask]
    
    if len(returns_clean) < 2:
        return 0.0
    
    covariance = np.cov(returns_clean, benchmark_clean)[0, 1]
    benchmark_variance = np.var(benchmark_clean)
    
    if benchmark_variance == 0:
        return 0.0
    
    return float(covariance / benchmark_variance)


def calculate_performance_metrics(returns: Union[pd.Series, np.ndarray],
                                benchmark_returns: Union[pd.Series, np.ndarray] = None,
                                risk_free_rate: float = 0.0,
                                periods_per_year: int = 252) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
    
    metrics = {}
    
    # Basic return metrics
    metrics['total_return'] = float((1 + returns).prod() - 1)
    metrics['annual_return'] = float((1 + returns.mean()) ** periods_per_year - 1)  # Changed from annualized_return
    metrics['annual_volatility'] = float(returns.std() * np.sqrt(periods_per_year))
    
    # Risk-adjusted metrics
    if returns.std() != 0:
        excess_returns = returns - risk_free_rate / periods_per_year
        metrics['sharpe_ratio'] = float(excess_returns.mean() / returns.std() * np.sqrt(periods_per_year))
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            metrics['sortino_ratio'] = float(excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year))
        else:
            metrics['sortino_ratio'] = 0.0
    else:
        metrics['sharpe_ratio'] = 0.0
        metrics['sortino_ratio'] = 0.0
    
    # Drawdown metrics
    cumulative_returns = (1 + returns).cumprod()
    metrics['max_drawdown'] = calculate_max_drawdown(cumulative_returns)
    if metrics['max_drawdown'] != 0:
        metrics['calmar_ratio'] = float(metrics['annual_return'] / abs(metrics['max_drawdown']))
    else:
        metrics['calmar_ratio'] = 0.0
    
    # Relative metrics if benchmark provided
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, np.ndarray):
            benchmark_returns = pd.Series(benchmark_returns)
        
        if len(returns) == len(benchmark_returns):
            metrics['beta'] = calculate_beta(returns, benchmark_returns)
            metrics['alpha'] = calculate_alpha(returns, benchmark_returns, risk_free_rate, periods_per_year)
            metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns)
            metrics['tracking_error'] = calculate_tracking_error(returns, benchmark_returns)
            metrics['upside_capture'] = calculate_upside_capture(returns, benchmark_returns)
            metrics['downside_capture'] = calculate_downside_capture(returns, benchmark_returns)
    
    return metrics


def calculate_risk_metrics(returns: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """Calculate comprehensive risk metrics."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
    
    metrics = {}
    
    # Value at Risk metrics
    metrics['var_95'] = calculate_var(returns, 0.05)
    metrics['var_99'] = calculate_var(returns, 0.01)
    metrics['cvar_95'] = calculate_cvar(returns, 0.05)
    metrics['cvar_99'] = calculate_cvar(returns, 0.01)
    
    # Distribution metrics
    metrics['skewness'] = float(returns.skew())
    metrics['kurtosis'] = float(returns.kurtosis())
    
    # Drawdown metrics
    cumulative_returns = (1 + returns).cumprod()
    metrics['max_drawdown'] = calculate_max_drawdown(cumulative_returns)
    
    return metrics

