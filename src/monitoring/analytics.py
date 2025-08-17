"""
Analytics module for the RL trading system.

This module provides comprehensive analytics capabilities including
performance tracking, trade analysis, and market insights.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..utils.metrics import calculate_performance_metrics, calculate_risk_metrics

logger = get_logger(__name__)


@dataclass
class TradeAnalysis:
    """Trade analysis results."""
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
    avg_holding_period: float
    win_loss_ratio: float
    recovery_factor: float
    payoff_ratio: float


@dataclass
class PerformanceAnalysis:
    """Performance analysis results."""
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    upside_capture: float
    downside_capture: float


class TradingAnalytics:
    """
    Comprehensive trading analytics system.
    
    This class provides detailed analysis of trading performance,
    trade statistics, risk metrics, and market insights.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize trading analytics.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        
        # Data storage
        self.equity_curve = []
        self.trade_history = []
        self.position_history = []
        self.market_data = []
        
        # Analysis results
        self.trade_analysis = None
        self.performance_analysis = None
        self.risk_analysis = None
        
        # Configuration
        self.benchmark_symbol = settings.get("analytics", "benchmark_symbol", default="SPY")
        self.risk_free_rate = settings.get("analytics", "risk_free_rate", default=0.02)
        self.timeframe = settings.get("analytics", "timeframe", default="daily")
        
        logger.info("Trading analytics initialized")
    
    def add_equity_data(self, timestamp: datetime, equity: float, pnl: float = 0.0):
        """
        Add equity data point.
        
        Args:
            timestamp: Timestamp
            equity: Equity value
            pnl: P&L value
        """
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'pnl': pnl
        })
    
    def add_trade(self, trade: Dict[str, Any]):
        """
        Add trade to history.
        
        Args:
            trade: Trade dictionary
        """
        self.trade_history.append(trade)
    
    def add_position(self, position: Dict[str, Any]):
        """
        Add position to history.
        
        Args:
            position: Position dictionary
        """
        self.position_history.append(position)
    
    def add_market_data(self, market_data: Dict[str, Any]):
        """
        Add market data point.
        
        Args:
            market_data: Market data dictionary
        """
        self.market_data.append(market_data)
    
    def analyze_trades(self) -> TradeAnalysis:
        """
        Analyze trade history.
        
        Returns:
            Trade analysis results
        """
        if not self.trade_history:
            return TradeAnalysis(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                avg_trade_duration=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                avg_holding_period=0.0,
                win_loss_ratio=0.0,
                recovery_factor=0.0,
                payoff_ratio=0.0
            )
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(self.trade_history)
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # P&L statistics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0.0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0.0
        largest_win = trades_df['pnl'].max()
        largest_loss = trades_df['pnl'].min()
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Duration analysis
        if 'duration' in trades_df.columns:
            avg_trade_duration = trades_df['duration'].mean()
        else:
            avg_trade_duration = 0.0
        
        # Consecutive analysis
        max_consecutive_wins = self._calculate_consecutive(trades_df['pnl'] > 0)
        max_consecutive_losses = self._calculate_consecutive(trades_df['pnl'] < 0)
        
        # Holding period
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            holding_periods = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
            avg_holding_period = holding_periods.dt.total_seconds().mean() / 60  # minutes
        else:
            avg_holding_period = 0.0
        
        # Ratios
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        payoff_ratio = largest_win / abs(largest_loss) if largest_loss != 0 else 0.0
        
        # Recovery factor
        total_net_profit = trades_df['pnl'].sum()
        max_drawdown = self._calculate_max_drawdown()
        recovery_factor = total_net_profit / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        self.trade_analysis = TradeAnalysis(
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
            avg_holding_period=avg_holding_period,
            win_loss_ratio=win_loss_ratio,
            recovery_factor=recovery_factor,
            payoff_ratio=payoff_ratio
        )
        
        return self.trade_analysis
    
    def analyze_performance(self) -> PerformanceAnalysis:
        """
        Analyze performance metrics.
        
        Returns:
            Performance analysis results
        """
        if not self.equity_curve:
            return PerformanceAnalysis(
                total_return=0.0,
                annual_return=0.0,
                annual_volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                beta=0.0,
                alpha=0.0,
                information_ratio=0.0,
                tracking_error=0.0,
                upside_capture=0.0,
                downside_capture=0.0
            )
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Basic performance metrics
        total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
        sortino_ratio = (annual_return - self.risk_free_rate) / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0.0
        
        # Drawdown analysis
        cumulative_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk
        returns_sorted = np.sort(returns)
        cvar_95 = returns_sorted[returns_sorted <= var_95].mean() if len(returns_sorted[returns_sorted <= var_95]) > 0 else var_95
        cvar_99 = returns_sorted[returns_sorted <= var_99].mean() if len(returns_sorted[returns_sorted <= var_99]) > 0 else var_99
        
        # Benchmark analysis (simplified)
        beta = 1.0  # Placeholder - would need benchmark data
        alpha = annual_return - (self.risk_free_rate + beta * (0.08 - self.risk_free_rate))  # Assuming 8% market return
        information_ratio = (annual_return - 0.08) / annual_volatility  # Assuming 8% benchmark return
        tracking_error = annual_volatility  # Placeholder
        upside_capture = 1.0  # Placeholder
        downside_capture = 1.0  # Placeholder
        
        self.performance_analysis = PerformanceAnalysis(
            total_return=total_return,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            upside_capture=upside_capture,
            downside_capture=downside_capture
        )
        
        return self.performance_analysis
    
    def analyze_risk(self) -> Dict[str, Any]:
        """
        Analyze risk metrics.
        
        Returns:
            Risk analysis results
        """
        if not self.equity_curve:
            return {}
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Risk metrics
        risk_metrics = calculate_risk_metrics(returns)
        
        # Additional risk analysis
        risk_analysis = {
            'volatility_analysis': {
                'annual_volatility': risk_metrics.get('volatility', 0.0),
                'monthly_volatility': risk_metrics.get('volatility', 0.0) / np.sqrt(12),
                'weekly_volatility': risk_metrics.get('volatility', 0.0) / np.sqrt(52),
                'daily_volatility': risk_metrics.get('volatility', 0.0) / np.sqrt(252)
            },
            'drawdown_analysis': {
                'max_drawdown': risk_metrics.get('max_drawdown', 0.0),
                'avg_drawdown': risk_metrics.get('avg_drawdown', 0.0),
                'drawdown_duration': risk_metrics.get('drawdown_duration', 0.0),
                'time_to_recovery': risk_metrics.get('time_to_recovery', 0.0)
            },
            'risk_adjusted_metrics': {
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': risk_metrics.get('sortino_ratio', 0.0),
                'calmar_ratio': risk_metrics.get('calmar_ratio', 0.0),
                'omega_ratio': self._calculate_omega_ratio(returns),
                'kappa_ratio': self._calculate_kappa_ratio(returns)
            },
            'distribution_analysis': {
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'jarque_bera': jarque_bera(returns)[:2],  # statistic, p-value
                'shapiro_wilk': shapiro(returns) if len(returns) < 5000 else (np.nan, np.nan),
                'normality_test': kstest(returns, 'norm', args=(returns.mean(), returns.std()))[:2]
            }
        }
        
        self.risk_analysis = risk_analysis
        return risk_analysis
    
    def _calculate_consecutive(self, series: pd.Series) -> int:
        """
        Calculate maximum consecutive values.
        
        Args:
            series: Boolean series
            
        Returns:
            Maximum consecutive count
        """
        if len(series) == 0:
            return 0
        
        max_count = 0
        current_count = 0
        
        for val in series:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown
        """
        if not self.equity_curve:
            return 0.0
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        cumulative_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cumulative_max) / cumulative_max
        
        return drawdown.min()
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            returns: Returns series
            threshold: Threshold for positive returns
            
        Returns:
            Omega ratio
        """
        returns_above_threshold = returns[returns > threshold]
        returns_below_threshold = returns[returns <= threshold]
        
        if len(returns_below_threshold) == 0:
            return np.inf
        
        return returns_above_threshold.sum() / abs(returns_below_threshold.sum())
    
    def _calculate_kappa_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Kappa ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate
            
        Returns:
            Kappa ratio
        """
        excess_returns = returns - risk_free_rate / 252
        downside_deviation = excess_returns[excess_returns < 0].std()
        
        if downside_deviation == 0:
            return np.inf
        
        return excess_returns.mean() / downside_deviation
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate comprehensive analytics report.
        
        Args:
            output_path: Output path for report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Perform analyses
        trade_analysis = self.analyze_trades()
        performance_analysis = self.analyze_performance()
        risk_analysis = self.analyze_risk()
        
        # Prepare report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'trade_analysis': {
                'total_trades': trade_analysis.total_trades,
                'winning_trades': trade_analysis.winning_trades,
                'losing_trades': trade_analysis.losing_trades,
                'win_rate': trade_analysis.win_rate,
                'avg_win': trade_analysis.avg_win,
                'avg_loss': trade_analysis.avg_loss,
                'profit_factor': trade_analysis.profit_factor,
                'largest_win': trade_analysis.largest_win,
                'largest_loss': trade_analysis.largest_loss,
                'avg_trade_duration': trade_analysis.avg_trade_duration,
                'max_consecutive_wins': trade_analysis.max_consecutive_wins,
                'max_consecutive_losses': trade_analysis.max_consecutive_losses,
                'avg_holding_period': trade_analysis.avg_holding_period,
                'win_loss_ratio': trade_analysis.win_loss_ratio,
                'recovery_factor': trade_analysis.recovery_factor,
                'payoff_ratio': trade_analysis.payoff_ratio
            },
            'performance_analysis': {
                'total_return': performance_analysis.total_return,
                'annual_return': performance_analysis.annual_return,
                'annual_volatility': performance_analysis.annual_volatility,
                'sharpe_ratio': performance_analysis.sharpe_ratio,
                'sortino_ratio': performance_analysis.sortino_ratio,
                'calmar_ratio': performance_analysis.calmar_ratio,
                'max_drawdown': performance_analysis.max_drawdown,
                'current_drawdown': performance_analysis.current_drawdown,
                'var_95': performance_analysis.var_95,
                'var_99': performance_analysis.var_99,
                'cvar_95': performance_analysis.cvar_95,
                'cvar_99': performance_analysis.cvar_99,
                'beta': performance_analysis.beta,
                'alpha': performance_analysis.alpha,
                'information_ratio': performance_analysis.information_ratio,
                'tracking_error': performance_analysis.tracking_error,
                'upside_capture': performance_analysis.upside_capture,
                'downside_capture': performance_analysis.downside_capture
            },
            'risk_analysis': risk_analysis,
            'data_summary': {
                'equity_points': len(self.equity_curve),
                'trades': len(self.trade_history),
                'positions': len(self.position_history),
                'market_data_points': len(self.market_data)
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Analytics report saved to {output_path}")
    
    def generate_plots(self, output_dir: str) -> None:
        """
        Generate analytics plots.
        
        Args:
            output_dir: Output directory for plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.equity_curve:
            logger.warning("No equity data available for plotting")
            return
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Equity curve
        axes[0, 0].plot(equity_df.index, equity_df['equity'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Equity ($)')
        axes[0, 0].grid(True)
        
        # Drawdown curve
        cumulative_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cumulative_max) / cumulative_max
        axes[0, 1].fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(equity_df.index, drawdown)
        axes[0, 1].set_title('Drawdown Curve')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # Returns distribution
        returns = equity_df['equity'].pct_change().dropna()
        axes[0, 2].hist(returns, bins=50, alpha=0.7, density=True, edgecolor='black')
        axes[0, 2].axvline(returns.mean(), color='red', linestyle='--', linewidth=2)
        axes[0, 2].set_title('Returns Distribution')
        axes[0, 2].set_xlabel('Returns')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].grid(True)
        
        # Trade P&L
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            axes[1, 0].hist(trades_df['pnl'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_title('Trade P&L Distribution')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # Monthly returns heatmap
        monthly_returns = equity_df['equity'].resample('M').last().pct_change()
        monthly_returns = monthly_returns.dropna()
        
        if len(monthly_returns) > 0:
            # Create year-month matrix
            monthly_returns.index = monthly_returns.index.to_period('M')
            monthly_returns_matrix = monthly_returns.unstack()
            
            sns.heatmap(monthly_returns_matrix, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Monthly Returns Heatmap')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Year')
        
        # Rolling metrics
        window = 252  # 1 year
        if len(returns) > window:
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            axes[1, 2].plot(rolling_sharpe.index, rolling_sharpe)
            axes[1, 2].set_title(f'Rolling Sharpe Ratio ({window} days)')
            axes[1, 2].set_xlabel('Date')
            axes[1, 2].set_ylabel('Sharpe Ratio')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "analytics_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Analytics plots saved to {output_dir}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        # Perform analyses
        trade_analysis = self.analyze_trades()
        performance_analysis = self.analyze_performance()
        risk_analysis = self.analyze_risk()
        
        return {
            'trade_analysis': {
                'total_trades': trade_analysis.total_trades,
                'win_rate': trade_analysis.win_rate,
                'profit_factor': trade_analysis.profit_factor,
                'avg_win': trade_analysis.avg_win,
                'avg_loss': trade_analysis.avg_loss,
                'largest_win': trade_analysis.largest_win,
                'largest_loss': trade_analysis.largest_loss,
                'max_consecutive_wins': trade_analysis.max_consecutive_wins,
                'max_consecutive_losses': trade_analysis.max_consecutive_losses
            },
            'performance_analysis': {
                'total_return': performance_analysis.total_return,
                'annual_return': performance_analysis.annual_return,
                'sharpe_ratio': performance_analysis.sharpe_ratio,
                'sortino_ratio': performance_analysis.sortino_ratio,
                'calmar_ratio': performance_analysis.calmar_ratio,
                'max_drawdown': performance_analysis.max_drawdown,
                'current_drawdown': performance_analysis.current_drawdown,
                'var_95': performance_analysis.var_95,
                'var_99': performance_analysis.var_99,
                'cvar_95': performance_analysis.cvar_95,
                'cvar_99': performance_analysis.cvar_99
            },
            'risk_analysis': risk_analysis,
            'data_summary': {
                'equity_points': len(self.equity_curve),
                'trades': len(self.trade_history),
                'positions': len(self.position_history),
                'market_data_points': len(self.market_data)
            }
        }
    
    def reset_data(self):
        """Reset all data."""
        self.equity_curve = []
        self.trade_history = []
        self.position_history = []
        self.market_data = []
        self.trade_analysis = None
        self.performance_analysis = None
        self.risk_analysis = None
        
        logger.info("Analytics data reset")