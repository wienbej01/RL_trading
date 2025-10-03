"""
Comprehensive backtest reporting for multi-ticker RL trading system.

This module provides comprehensive backtest reporting capabilities for
multi-ticker RL trading strategies, including performance metrics,
risk analysis, and visualization.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from dataclasses import dataclass, asdict
import scipy.stats as stats

from ..utils.logging import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)


@dataclass
class BacktestMetrics:
    """
    Backtest metrics dataclass.
    """
    # Return metrics
    total_return: float
    annual_return: float
    monthly_returns: List[float]
    daily_returns: List[float]
    
    # Risk metrics
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    recovery_factor: float
    payoff_ratio: float
    
    # Long/short metrics
    num_long_trades: int
    num_short_trades: int
    long_win_rate: float
    short_win_rate: float
    avg_pnl_long: float
    avg_pnl_short: float
    avg_duration_min: float
    avg_duration_long_min: float
    avg_duration_short_min: float
    
    # Cost metrics
    total_commission: float
    total_slippage: float
    total_transaction_costs: float
    
    # Equity metrics
    initial_equity: float
    final_equity: float
    peak_equity: float
    
    # Ticker metrics
    ticker_metrics: Dict[str, Dict[str, float]]
    
    # Regime metrics
    regime_metrics: Dict[str, Dict[str, float]]
    
    # Time metrics
    start_date: str
    end_date: str
    trading_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BacktestReporter:
    """
    Comprehensive backtest reporter for multi-ticker RL trading system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize backtest reporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.metrics = None
        self.equity_curve = None
        self.drawdown_curve = None
        self.trade_history = None
        self.position_history = None
        self.returns_history = None
        self.benchmark_returns = None
        
        # Visualization settings
        self.figsize = self.config.get('figsize', (12, 8))
        self.dpi = self.config.get('dpi', 100)
        self.style = self.config.get('style', 'darkgrid')
        self.color_palette = self.config.get('color_palette', 'viridis')
        
        # Set plotting style
        sns.set_style(self.style)
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
        
    def set_data(
        self,
        equity_curve: List[float],
        timestamps: List[datetime],
        trade_history: List[Dict[str, Any]],
        position_history: List[Dict[str, Any]],
        returns_history: List[float],
        benchmark_returns: Optional[List[float]] = None,
        ticker_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        regime_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """
        Set backtest data.
        
        Args:
            equity_curve: Equity curve values
            timestamps: Timestamps for equity curve
            trade_history: Trade history
            position_history: Position history
            returns_history: Returns history
            benchmark_returns: Benchmark returns (optional)
            ticker_metrics: Ticker-specific metrics (optional)
            regime_metrics: Regime-specific metrics (optional)
        """
        self.equity_curve = equity_curve
        self.timestamps = timestamps
        self.trade_history = trade_history
        self.position_history = position_history
        self.returns_history = returns_history
        self.benchmark_returns = benchmark_returns
        self.ticker_metrics = ticker_metrics or {}
        self.regime_metrics = regime_metrics or {}
        
        # Calculate drawdown curve
        self.drawdown_curve = self._calculate_drawdown_curve(equity_curve)
        
        # Calculate metrics
        self.metrics = self._calculate_metrics()
        
    def _calculate_drawdown_curve(self, equity_curve: List[float]) -> List[float]:
        """
        Calculate drawdown curve.
        
        Args:
            equity_curve: Equity curve values
            
        Returns:
            Drawdown curve values
        """
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        return drawdown.fillna(0).tolist()
        
    def _calculate_metrics(self) -> BacktestMetrics:
        """
        Calculate backtest metrics.
        
        Returns:
            BacktestMetrics object
        """
        if not self.equity_curve or not self.returns_history:
            raise ValueError("Equity curve and returns history must be set")
            
        # Convert to pandas Series for easier calculations
        equity_series = pd.Series(self.equity_curve)
        returns_series = pd.Series(self.returns_history)
        
        # Time metrics
        start_date = self.timestamps[0]
        end_date = self.timestamps[-1]
        trading_days = len(self.timestamps)
        
        # Return metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        days = (end_date - start_date).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Monthly returns
        equity_df = pd.DataFrame({'equity': equity_series}, index=self.timestamps)
        monthly_returns = equity_df['equity'].resample('M').last().pct_change().dropna().tolist()
        
        # Risk metrics
        annual_volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        max_drawdown = min(self.drawdown_curve)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Drawdown metrics
        avg_drawdown = np.mean(self.drawdown_curve)
        drawdown_duration = self._calculate_drawdown_duration()
        
        # VaR and CVaR
        var_95 = returns_series.quantile(0.05)
        var_99 = returns_series.quantile(0.01)
        cvar_95 = returns_series[returns_series <= var_95].mean()
        cvar_99 = returns_series[returns_series <= var_99].mean()
        
        # Beta and alpha (if benchmark provided)
        beta = 0
        alpha = 0
        information_ratio = 0
        tracking_error = 0
        
        if self.benchmark_returns and len(self.benchmark_returns) == len(returns_series):
            benchmark_series = pd.Series(self.benchmark_returns)
            
            # Beta
            covariance = np.cov(returns_series, benchmark_series)[0, 1]
            benchmark_variance = np.var(benchmark_series)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha
            alpha = (annual_return - (0.02 + beta * (benchmark_series.mean() * 252))) if len(benchmark_series) > 0 else 0
            
            # Information ratio
            active_returns = returns_series - benchmark_series
            information_ratio = active_returns.mean() / active_returns.std() if active_returns.std() > 0 else 0
            
            # Tracking error
            tracking_error = active_returns.std() * np.sqrt(252)
            
        # Trade metrics
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        trade_pnls = [t.get('pnl', 0) for t in self.trade_history]
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses else 0
        
        largest_win = max(trade_pnls) if trade_pnls else 0
        largest_loss = min(trade_pnls) if trade_pnls else 0
        
        # Trade durations
        durations = []
        for trade in self.trade_history:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                duration = (exit_time - entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
                
        avg_trade_duration = np.mean(durations) if durations else 0
        
        # Consecutive wins/losses
        consecutive_wins = []
        consecutive_losses = []
        current_win_streak = 0
        current_loss_streak = 0
        
        for pnl in trade_pnls:
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
            elif pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
            else:
                current_win_streak = 0
                current_loss_streak = 0
                
            consecutive_wins.append(current_win_streak)
            consecutive_losses.append(current_loss_streak)
            
        max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
        max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
        
        # Recovery factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Payoff ratio
        payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
        
        # Long/short metrics
        long_trades = [t for t in self.trade_history if t.get('direction', '') == 'long']
        short_trades = [t for t in self.trade_history if t.get('direction', '') == 'short']
        
        num_long_trades = len(long_trades)
        num_short_trades = len(short_trades)
        
        long_wins = [t for t in long_trades if t.get('pnl', 0) > 0]
        short_wins = [t for t in short_trades if t.get('pnl', 0) > 0]
        
        long_win_rate = len(long_wins) / num_long_trades if num_long_trades > 0 else 0
        short_win_rate = len(short_wins) / num_short_trades if num_short_trades > 0 else 0
        
        long_pnls = [t.get('pnl', 0) for t in long_trades]
        short_pnls = [t.get('pnl', 0) for t in short_trades]
        
        avg_pnl_long = np.mean(long_pnls) if long_pnls else 0
        avg_pnl_short = np.mean(short_pnls) if short_pnls else 0
        
        # Long/short durations
        long_durations = []
        for trade in long_trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                duration = (exit_time - entry_time).total_seconds() / 60  # minutes
                long_durations.append(duration)
                
        short_durations = []
        for trade in short_trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                duration = (exit_time - entry_time).total_seconds() / 60  # minutes
                short_durations.append(duration)
                
        avg_duration_min = np.mean(durations) * 60 if durations else 0  # convert to minutes
        avg_duration_long_min = np.mean(long_durations) if long_durations else 0
        avg_duration_short_min = np.mean(short_durations) if short_durations else 0
        
        # Cost metrics
        total_commission = sum(t.get('commission', 0) for t in self.trade_history)
        total_slippage = sum(t.get('slippage', 0) for t in self.trade_history)
        total_transaction_costs = total_commission + total_slippage
        
        # Equity metrics
        initial_equity = equity_series.iloc[0]
        final_equity = equity_series.iloc[-1]
        peak_equity = equity_series.max()
        
        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_returns=monthly_returns,
            daily_returns=returns_series.tolist(),
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_duration=drawdown_duration,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            recovery_factor=recovery_factor,
            payoff_ratio=payoff_ratio,
            num_long_trades=num_long_trades,
            num_short_trades=num_short_trades,
            long_win_rate=long_win_rate,
            short_win_rate=short_win_rate,
            avg_pnl_long=avg_pnl_long,
            avg_pnl_short=avg_pnl_short,
            avg_duration_min=avg_duration_min,
            avg_duration_long_min=avg_duration_long_min,
            avg_duration_short_min=avg_duration_short_min,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_transaction_costs=total_transaction_costs,
            initial_equity=initial_equity,
            final_equity=final_equity,
            peak_equity=peak_equity,
            ticker_metrics=self.ticker_metrics,
            regime_metrics=self.regime_metrics,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            trading_days=trading_days
        )
        
    def _calculate_drawdown_duration(self) -> int:
        """
        Calculate maximum drawdown duration in days.
        
        Returns:
            Maximum drawdown duration in days
        """
        drawdown_series = pd.Series(self.drawdown_curve, index=self.timestamps)
        
        # Find drawdown periods
        is_drawdown = drawdown_series < 0
        drawdown_periods = []
        
        start_date = None
        for date, is_dd in is_drawdown.items():
            if is_dd and start_date is None:
                start_date = date
            elif not is_dd and start_date is not None:
                drawdown_periods.append((start_date, date))
                start_date = None
                
        # If still in drawdown at the end
        if start_date is not None:
            drawdown_periods.append((start_date, is_drawdown.index[-1]))
            
        # Calculate maximum duration
        max_duration = 0
        for start, end in drawdown_periods:
            duration = (end - start).days
            if duration > max_duration:
                max_duration = duration
                
        return max_duration
        
    def generate_report(self, output_dir: str, format: str = 'html') -> None:
        """
        Generate comprehensive backtest report.
        
        Args:
            output_dir: Output directory
            format: Report format ('html', 'pdf', 'json')
        """
        if not self.metrics:
            raise ValueError("Metrics not calculated. Call set_data() first.")
            
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        self._generate_plots(output_path)
        
        # Generate report
        if format == 'html':
            self._generate_html_report(output_path)
        elif format == 'json':
            self._generate_json_report(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Backtest report generated in {output_path}")
        
    def _generate_plots(self, output_path: Path) -> None:
        """
        Generate backtest plots.
        
        Args:
            output_path: Output path
        """
        # Create plots directory
        plots_dir = output_path / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Generate equity curve plot
        self._plot_equity_curve(plots_dir / 'equity_curve.png')
        
        # Generate drawdown curve plot
        self._plot_drawdown_curve(plots_dir / 'drawdown_curve.png')
        
        # Generate returns distribution plot
        self._plot_returns_distribution(plots_dir / 'returns_distribution.png')
        
        # Generate monthly returns heatmap
        self._plot_monthly_returns_heatmap(plots_dir / 'monthly_returns_heatmap.png')
        
        # Generate trade P&L distribution
        self._plot_trade_pnl_distribution(plots_dir / 'trade_pnl_distribution.png')
        
        # Generate rolling metrics plot
        self._plot_rolling_metrics(plots_dir / 'rolling_metrics.png')
        
        # Generate ticker performance plot if ticker metrics available
        if self.ticker_metrics:
            self._plot_ticker_performance(plots_dir / 'ticker_performance.png')
            
        # Generate regime performance plot if regime metrics available
        if self.regime_metrics:
            self._plot_regime_performance(plots_dir / 'regime_performance.png')
            
    def _plot_equity_curve(self, output_path: Path) -> None:
        """
        Plot equity curve.
        
        Args:
            output_path: Output path
        """
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(self.timestamps, self.equity_curve, label='Portfolio Equity', linewidth=2)
        
        # Plot benchmark if available
        if self.benchmark_returns:
            benchmark_equity = [self.metrics.initial_equity]
            for ret in self.benchmark_returns:
                benchmark_equity.append(benchmark_equity[-1] * (1 + ret))
                
            plt.plot(self.timestamps, benchmark_equity[1:], label='Benchmark', linewidth=2, alpha=0.7)
            
        # Add horizontal lines
        plt.axhline(y=self.metrics.initial_equity, color='red', linestyle='--', alpha=0.7, label='Initial Equity')
        plt.axhline(y=self.metrics.peak_equity, color='green', linestyle='--', alpha=0.7, label='Peak Equity')
        
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
    def _plot_drawdown_curve(self, output_path: Path) -> None:
        """
        Plot drawdown curve.
        
        Args:
            output_path: Output path
        """
        plt.figure(figsize=(12, 6))
        
        # Plot drawdown curve
        plt.fill_between(self.timestamps, self.drawdown_curve, 0, alpha=0.3, color='red')
        plt.plot(self.timestamps, self.drawdown_curve, color='red', linewidth=2)
        
        plt.title('Portfolio Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
    def _plot_returns_distribution(self, output_path: Path) -> None:
        """
        Plot returns distribution.
        
        Args:
            output_path: Output path
        """
        plt.figure(figsize=(12, 6))
        
        # Plot returns histogram
        plt.hist(self.metrics.daily_returns, bins=50, alpha=0.7, density=True, edgecolor='black')
        
        # Plot normal distribution overlay
        mu, sigma = stats.norm.fit(self.metrics.daily_returns)
        x = np.linspace(min(self.metrics.daily_returns), max(self.metrics.daily_returns), 100)
        y = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
        
        # Add VaR lines
        plt.axvline(x=self.metrics.var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {self.metrics.var_95:.2%}')
        plt.axvline(x=self.metrics.var_99, color='red', linestyle='--', linewidth=2, label=f'VaR 99%: {self.metrics.var_99:.2%}')
        
        plt.title('Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis as percentage
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
    def _plot_monthly_returns_heatmap(self, output_path: Path) -> None:
        """
        Plot monthly returns heatmap.
        
        Args:
            output_path: Output path
        """
        if not self.metrics.monthly_returns:
            return
            
        # Create equity DataFrame
        equity_df = pd.DataFrame({'equity': self.equity_curve}, index=self.timestamps)
        
        # Calculate monthly returns
        monthly_returns = equity_df['equity'].resample('M').last().pct_change().dropna()
        
        # Create year-month matrix
        monthly_returns.index = monthly_returns.index.to_period('M')
        years = monthly_returns.index.year.unique()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create matrix
        matrix = np.zeros((len(years), 12))
        matrix[:] = np.nan
        
        for i, year in enumerate(years):
            year_returns = monthly_returns[monthly_returns.index.year == year]
            for j, month in enumerate(range(1, 13)):
                month_period = f"{year}-{month:02d}"
                if month_period in year_returns.index.astype(str):
                    matrix[i, j] = year_returns[month_period]
                    
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            center=0,
            yticklabels=years,
            xticklabels=months,
            cbar_kws={'label': 'Monthly Return'}
        )
        
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
    def _plot_trade_pnl_distribution(self, output_path: Path) -> None:
        """
        Plot trade P&L distribution.
        
        Args:
            output_path: Output path
        """
        if not self.trade_history:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Extract trade P&Ls
        trade_pnls = [t.get('pnl', 0) for t in self.trade_history]
        
        # Plot histogram
        plt.hist(trade_pnls, bins=30, alpha=0.7, edgecolor='black')
        
        # Add zero line
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        
        plt.title('Trade P&L Distribution')
        plt.xlabel('P&L ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
    def _plot_rolling_metrics(self, output_path: Path) -> None:
        """
        Plot rolling metrics.
        
        Args:
            output_path: Output path
        """
        if not self.returns_history:
            return
            
        # Create returns DataFrame
        returns_df = pd.DataFrame({'returns': self.returns_history}, index=self.timestamps)
        
        # Calculate rolling metrics
        window = 63  # ~3 months
        rolling_sharpe = returns_df['returns'].rolling(window).mean() / returns_df['returns'].rolling(window).std() * np.sqrt(252)
        rolling_vol = returns_df['returns'].rolling(window).std() * np.sqrt(252)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot rolling Sharpe ratio
        axes[0].plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
        axes[0].set_title(f'Rolling Sharpe Ratio ({window} days)')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].grid(True, alpha=0.3)
        
        # Plot rolling volatility
        axes[1].plot(rolling_vol.index, rolling_vol, linewidth=2, color='red')
        axes[1].set_title(f'Rolling Volatility ({window} days)')
        axes[1].set_ylabel('Volatility')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        # Format y-axis as percentage for volatility
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
    def _plot_ticker_performance(self, output_path: Path) -> None:
        """
        Plot ticker performance.
        
        Args:
            output_path: Output path
        """
        if not self.ticker_metrics:
            return
            
        # Create ticker metrics DataFrame
        ticker_data = []
        for ticker, metrics in self.ticker_metrics.items():
            ticker_data.append({
                'ticker': ticker,
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0)
            })
            
        ticker_df = pd.DataFrame(ticker_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot total return
        axes[0, 0].bar(ticker_df['ticker'], ticker_df['total_return'])
        axes[0, 0].set_title('Total Return by Ticker')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        # Plot Sharpe ratio
        axes[0, 1].bar(ticker_df['ticker'], ticker_df['sharpe_ratio'])
        axes[0, 1].set_title('Sharpe Ratio by Ticker')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot max drawdown
        axes[1, 0].bar(ticker_df['ticker'], ticker_df['max_drawdown'])
        axes[1, 0].set_title('Max Drawdown by Ticker')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        # Plot win rate
        axes[1, 1].bar(ticker_df['ticker'], ticker_df['win_rate'])
        axes[1, 1].set_title('Win Rate by Ticker')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
    def _plot_regime_performance(self, output_path: Path) -> None:
        """
        Plot regime performance.
        
        Args:
            output_path: Output path
        """
        if not self.regime_metrics:
            return
            
        # Create regime metrics DataFrame
        regime_data = []
        for regime, metrics in self.regime_metrics.items():
            regime_data.append({
                'regime': regime,
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0)
            })
            
        regime_df = pd.DataFrame(regime_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot total return
        axes[0, 0].bar(regime_df['regime'], regime_df['total_return'])
        axes[0, 0].set_title('Total Return by Regime')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        # Plot Sharpe ratio
        axes[0, 1].bar(regime_df['regime'], regime_df['sharpe_ratio'])
        axes[0, 1].set_title('Sharpe Ratio by Regime')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot max drawdown
        axes[1, 0].bar(regime_df['regime'], regime_df['max_drawdown'])
        axes[1, 0].set_title('Max Drawdown by Regime')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        # Plot win rate
        axes[1, 1].bar(regime_df['regime'], regime_df['win_rate'])
        axes[1, 1].set_title('Win Rate by Regime')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
    def _generate_html_report(self, output_path: Path) -> None:
        """
        Generate HTML report.
        
        Args:
            output_path: Output path
        """
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Ticker RL Trading System Backtest Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .metric {{
                    display: inline-block;
                    width: 200px;
                    margin: 10px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
                .positive {{
                    color: #27ae60;
                }}
                .negative {{
                    color: #e74c3c;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <h1>Multi-Ticker RL Trading System Backtest Report</h1>
            
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-value {'positive' if self.metrics.total_return > 0 else 'negative'}">{self.metrics.total_return:.2%}</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if self.metrics.annual_return > 0 else 'negative'}">{self.metrics.annual_return:.2%}</div>
                <div class="metric-label">Annual Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.metrics.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{self.metrics.max_drawdown:.2%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.metrics.win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.metrics.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Start Date</td>
                    <td>{self.metrics.start_date}</td>
                </tr>
                <tr>
                    <td>End Date</td>
                    <td>{self.metrics.end_date}</td>
                </tr>
                <tr>
                    <td>Trading Days</td>
                    <td>{self.metrics.trading_days}</td>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td class="{'positive' if self.metrics.total_return > 0 else 'negative'}">{self.metrics.total_return:.2%}</td>
                </tr>
                <tr>
                    <td>Annual Return</td>
                    <td class="{'positive' if self.metrics.annual_return > 0 else 'negative'}">{self.metrics.annual_return:.2%}</td>
                </tr>
                <tr>
                    <td>Annual Volatility</td>
                    <td>{self.metrics.annual_volatility:.2%}</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{self.metrics.sharpe_ratio:.2f}</td>
                </tr>
                <tr>
                    <td>Sortino Ratio</td>
                    <td>{self.metrics.sortino_ratio:.2f}</td>
                </tr>
                <tr>
                    <td>Calmar Ratio</td>
                    <td>{self.metrics.calmar_ratio:.2f}</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td class="negative">{self.metrics.max_drawdown:.2%}</td>
                </tr>
                <tr>
                    <td>Avg Drawdown</td>
                    <td class="negative">{self.metrics.avg_drawdown:.2%}</td>
                </tr>
                <tr>
                    <td>Drawdown Duration</td>
                    <td>{self.metrics.drawdown_duration} days</td>
                </tr>
                <tr>
                    <td>VaR 95%</td>
                    <td>{self.metrics.var_95:.2%}</td>
                </tr>
                <tr>
                    <td>VaR 99%</td>
                    <td>{self.metrics.var_99:.2%}</td>
                </tr>
                <tr>
                    <td>CVaR 95%</td>
                    <td>{self.metrics.cvar_95:.2%}</td>
                </tr>
                <tr>
                    <td>CVaR 99%</td>
                    <td>{self.metrics.cvar_99:.2%}</td>
                </tr>
                <tr>
                    <td>Beta</td>
                    <td>{self.metrics.beta:.2f}</td>
                </tr>
                <tr>
                    <td>Alpha</td>
                    <td class="{'positive' if self.metrics.alpha > 0 else 'negative'}">{self.metrics.alpha:.2%}</td>
                </tr>
                <tr>
                    <td>Information Ratio</td>
                    <td>{self.metrics.information_ratio:.2f}</td>
                </tr>
                <tr>
                    <td>Tracking Error</td>
                    <td>{self.metrics.tracking_error:.2%}</td>
                </tr>
            </table>
            
            <h2>Trade Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{self.metrics.total_trades}</td>
                </tr>
                <tr>
                    <td>Winning Trades</td>
                    <td>{self.metrics.winning_trades}</td>
                </tr>
                <tr>
                    <td>Losing Trades</td>
                    <td>{self.metrics.losing_trades}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{self.metrics.win_rate:.1%}</td>
                </tr>
                <tr>
                    <td>Average Win</td>
                    <td class="positive">${self.metrics.avg_win:.2f}</td>
                </tr>
                <tr>
                    <td>Average Loss</td>
                    <td class="negative">${abs(self.metrics.avg_loss):.2f}</td>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td>{self.metrics.profit_factor:.2f}</td>
                </tr>
                <tr>
                    <td>Largest Win</td>
                    <td class="positive">${self.metrics.largest_win:.2f}</td>
                </tr>
                <tr>
                    <td>Largest Loss</td>
                    <td class="negative">${abs(self.metrics.largest_loss):.2f}</td>
                </tr>
                <tr>
                    <td>Average Trade Duration</td>
                    <td>{self.metrics.avg_trade_duration:.1f} hours</td>
                </tr>
                <tr>
                    <td>Max Consecutive Wins</td>
                    <td>{self.metrics.max_consecutive_wins}</td>
                </tr>
                <tr>
                    <td>Max Consecutive Losses</td>
                    <td>{self.metrics.max_consecutive_losses}</td>
                </tr>
                <tr>
                    <td>Recovery Factor</td>
                    <td>{self.metrics.recovery_factor:.2f}</td>
                </tr>
                <tr>
                    <td>Payoff Ratio</td>
                    <td>{self.metrics.payoff_ratio:.2f}</td>
                </tr>
            </table>
            
            <h2>Long/Short Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Long</th>
                    <th>Short</th>
                </tr>
                <tr>
                    <td>Number of Trades</td>
                    <td>{self.metrics.num_long_trades}</td>
                    <td>{self.metrics.num_short_trades}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{self.metrics.long_win_rate:.1%}</td>
                    <td>{self.metrics.short_win_rate:.1%}</td>
                </tr>
                <tr>
                    <td>Average P&L</td>
                    <td class="{'positive' if self.metrics.avg_pnl_long > 0 else 'negative'}">${self.metrics.avg_pnl_long:.2f}</td>
                    <td class="{'positive' if self.metrics.avg_pnl_short > 0 else 'negative'}">${self.metrics.avg_pnl_short:.2f}</td>
                </tr>
                <tr>
                    <td>Average Duration</td>
                    <td>{self.metrics.avg_duration_long_min:.1f} min</td>
                    <td>{self.metrics.avg_duration_short_min:.1f} min</td>
                </tr>
            </table>
            
            <h2>Cost Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Commission</td>
                    <td class="negative">${abs(self.metrics.total_commission):.2f}</td>
                </tr>
                <tr>
                    <td>Total Slippage</td>
                    <td class="negative">${abs(self.metrics.total_slippage):.2f}</td>
                </tr>
                <tr>
                    <td>Total Transaction Costs</td>
                    <td class="negative">${abs(self.metrics.total_transaction_costs):.2f}</td>
                </tr>
            </table>
            
            <h2>Equity Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Initial Equity</td>
                    <td>${self.metrics.initial_equity:.2f}</td>
                </tr>
                <tr>
                    <td>Final Equity</td>
                    <td class="{'positive' if self.metrics.final_equity > self.metrics.initial_equity else 'negative'}">${self.metrics.final_equity:.2f}</td>
                </tr>
                <tr>
                    <td>Peak Equity</td>
                    <td>${self.metrics.peak_equity:.2f}</td>
                </tr>
            </table>
        """
        
        # Add ticker metrics section if available
        if self.ticker_metrics:
            html_content += """
            <h2>Ticker Performance</h2>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Total Return</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                    <th>Number of Trades</th>
                </tr>
            """
            
            for ticker, metrics in self.ticker_metrics.items():
                html_content += f"""
                <tr>
                    <td>{ticker}</td>
                    <td class="{'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">{metrics.get('total_return', 0):.2%}</td>
                    <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                    <td class="negative">{metrics.get('max_drawdown', 0):.2%}</td>
                    <td>{metrics.get('win_rate', 0):.1%}</td>
                    <td>{metrics.get('num_trades', 0)}</td>
                </tr>
                """
                
            html_content += "</table>"
            
        # Add regime metrics section if available
        if self.regime_metrics:
            html_content += """
            <h2>Regime Performance</h2>
            <table>
                <tr>
                    <th>Regime</th>
                    <th>Total Return</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                    <th>Number of Trades</th>
                </tr>
            """
            
            for regime, metrics in self.regime_metrics.items():
                html_content += f"""
                <tr>
                    <td>{regime}</td>
                    <td class="{'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">{metrics.get('total_return', 0):.2%}</td>
                    <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                    <td class="negative">{metrics.get('max_drawdown', 0):.2%}</td>
                    <td>{metrics.get('win_rate', 0):.1%}</td>
                    <td>{metrics.get('num_trades', 0)}</td>
                </tr>
                """
                
            html_content += "</table>"
            
        # Add plots section
        html_content += """
            <h2>Plots</h2>
            
            <h3>Equity Curve</h3>
            <img src="plots/equity_curve.png" alt="Equity Curve">
            
            <h3>Drawdown Curve</h3>
            <img src="plots/drawdown_curve.png" alt="Drawdown Curve">
            
            <h3>Returns Distribution</h3>
            <img src="plots/returns_distribution.png" alt="Returns Distribution">
            
            <h3>Monthly Returns Heatmap</h3>
            <img src="plots/monthly_returns_heatmap.png" alt="Monthly Returns Heatmap">
            
            <h3>Trade P&L Distribution</h3>
            <img src="plots/trade_pnl_distribution.png" alt="Trade P&L Distribution">
            
            <h3>Rolling Metrics</h3>
            <img src="plots/rolling_metrics.png" alt="Rolling Metrics">
        """
        
        # Add ticker performance plot if available
        if self.ticker_metrics:
            html_content += """
            <h3>Ticker Performance</h3>
            <img src="plots/ticker_performance.png" alt="Ticker Performance">
            """
            
        # Add regime performance plot if available
        if self.regime_metrics:
            html_content += """
            <h3>Regime Performance</h3>
            <img src="plots/regime_performance.png" alt="Regime Performance">
            """
            
        # Add footer
        html_content += f"""
            <hr>
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_path / 'report.html', 'w') as f:
            f.write(html_content)
            
    def _generate_json_report(self, output_path: Path) -> None:
        """
        Generate JSON report.
        
        Args:
            output_path: Output path
        """
        # Create report data
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'start_date': self.metrics.start_date,
                'end_date': self.metrics.end_date,
                'trading_days': self.metrics.trading_days
            },
            'metrics': self.metrics.to_dict(),
            'equity_curve': [
                {
                    'timestamp': ts.isoformat(),
                    'equity': eq
                }
                for ts, eq in zip(self.timestamps, self.equity_curve)
            ],
            'drawdown_curve': [
                {
                    'timestamp': ts.isoformat(),
                    'drawdown': dd
                }
                for ts, dd in zip(self.timestamps, self.drawdown_curve)
            ],
            'trade_history': self.trade_history,
            'position_history': self.position_history
        }
        
        # Write JSON file
        with open(output_path / 'report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
            
    def create_interactive_dashboard(self, output_path: str) -> None:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            output_path: Output HTML file path
        """
        if not self.metrics:
            raise ValueError("Metrics not calculated. Call set_data() first.")
            
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve', 'Drawdown Curve',
                'Returns Distribution', 'Monthly Returns',
                'Rolling Sharpe Ratio', 'Rolling Volatility'
            ),
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.timestamps,
                y=self.equity_curve,
                name='Portfolio Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Benchmark if available
        if self.benchmark_returns:
            benchmark_equity = [self.metrics.initial_equity]
            for ret in self.benchmark_returns:
                benchmark_equity.append(benchmark_equity[-1] * (1 + ret))
                
            fig.add_trace(
                go.Scatter(
                    x=self.timestamps,
                    y=benchmark_equity[1:],
                    name='Benchmark',
                    line=dict(color='green', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
        # Drawdown curve
        fig.add_trace(
            go.Scatter(
                x=self.timestamps,
                y=self.drawdown_curve,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(
                x=self.metrics.daily_returns,
                name='Returns Distribution',
                nbinsx=50
            ),
            row=2, col=1
        )
        
        # Monthly returns heatmap
        if self.metrics.monthly_returns:
            equity_df = pd.DataFrame({'equity': self.equity_curve}, index=self.timestamps)
            monthly_returns = equity_df['equity'].resample('M').last().pct_change().dropna()
            
            # Create year-month matrix
            monthly_returns.index = monthly_returns.index.to_period('M')
            years = monthly_returns.index.year.unique()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Create matrix
            matrix = np.zeros((len(years), 12))
            matrix[:] = np.nan
            
            for i, year in enumerate(years):
                year_returns = monthly_returns[monthly_returns.index.year == year]
                for j, month in enumerate(range(1, 13)):
                    month_period = f"{year}-{month:02d}"
                    if month_period in year_returns.index.astype(str):
                        matrix[i, j] = year_returns[month_period]
                        
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=months,
                    y=years,
                    colorscale='RdYlGn',
                    name='Monthly Returns'
                ),
                row=2, col=2
            )
            
        # Rolling metrics
        if self.returns_history:
            returns_df = pd.DataFrame({'returns': self.returns_history}, index=self.timestamps)
            window = 63  # ~3 months
            
            rolling_sharpe = returns_df['returns'].rolling(window).mean() / returns_df['returns'].rolling(window).std() * np.sqrt(252)
            rolling_vol = returns_df['returns'].rolling(window).std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe,
                    name='Rolling Sharpe',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    name='Rolling Volatility',
                    line=dict(color='red', width=2)
                ),
                row=3, col=2
            )
            
        # Update layout
        fig.update_layout(
            title="Multi-Ticker RL Trading System Backtest Dashboard",
            height=1200,
            showlegend=True
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=3, col=2)
        
        # Write HTML file
        fig.write_html(output_path)
        
        logger.info(f"Interactive dashboard saved to {output_path}")