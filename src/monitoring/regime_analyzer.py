"""
Regime-specific performance analysis for multi-ticker RL trading system.

This module provides comprehensive regime analysis capabilities for
multi-ticker RL trading strategies, including regime detection,
regime-specific performance metrics, and regime transition analysis.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.logging import get_logger
from .performance_tracker import PerformanceTracker, PerformanceMetrics

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class RegimeDetector:
    """
    Market regime detector using various methods.
    """
    
    def __init__(self, method: str = 'volatility', window: int = 20, n_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            method: Detection method ('volatility', 'trend', 'clustering', 'pca')
            window: Lookback window for detection
            n_regimes: Number of regimes to detect
        """
        self.method = method
        self.window = window
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.pca = PCA(n_components=min(3, n_regimes))
        
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market regimes from data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with regime labels
        """
        if self.method == 'volatility':
            return self._detect_volatility_regimes(data)
        elif self.method == 'trend':
            return self._detect_trend_regimes(data)
        elif self.method == 'clustering':
            return self._detect_clustering_regimes(data)
        elif self.method == 'pca':
            return self._detect_pca_regimes(data)
        else:
            raise ValueError(f"Unknown regime detection method: {self.method}")
            
    def _detect_volatility_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect regimes based on volatility.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with regime labels
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column for volatility regime detection")
            
        # Calculate rolling volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=self.window).std()
        
        # Classify into regimes based on volatility percentiles
        low_vol = volatility.quantile(0.33)
        high_vol = volatility.quantile(0.67)
        
        regimes = pd.Series(0, index=volatility.index)
        regimes[volatility <= low_vol] = 0  # Low volatility
        regimes[(volatility > low_vol) & (volatility <= high_vol)] = 1  # Medium volatility
        regimes[volatility > high_vol] = 2  # High volatility
        
        return regimes
        
    def _detect_trend_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect regimes based on trend.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with regime labels
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column for trend regime detection")
            
        # Calculate rolling trend
        returns = data['close'].pct_change().dropna()
        trend = returns.rolling(window=self.window).mean()
        
        # Classify into regimes based on trend
        regimes = pd.Series(1, index=trend.index)  # Neutral
        regimes[trend > 0.001] = 2  # Upward
        regimes[trend < -0.001] = 0  # Downward
        
        return regimes
        
    def _detect_clustering_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect regimes using clustering.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with regime labels
        """
        # Select features for clustering
        features = []
        
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            features.append(returns.rolling(window=self.window).std().rename('volatility'))
            features.append(returns.rolling(window=self.window).mean().rename('trend'))
            
        if 'volume' in data.columns:
            volume_change = data['volume'].pct_change().dropna()
            features.append(volume_change.rolling(window=self.window).mean().rename('volume_trend'))
            
        if not features:
            raise ValueError("No suitable features for clustering regime detection")
            
        # Combine features
        feature_df = pd.concat(features, axis=1).dropna()
        
        if len(feature_df) < self.n_regimes:
            return pd.Series(0, index=data.index)
            
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_df)
        
        # Cluster
        labels = self.kmeans.fit_predict(scaled_features)
        
        # Create regime series
        regimes = pd.Series(0, index=data.index)
        regimes.loc[feature_df.index] = labels
        
        # Forward fill to cover all dates
        regimes = regimes.ffill()
        
        return regimes
        
    def _detect_pca_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect regimes using PCA.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with regime labels
        """
        # Select features for PCA
        features = []
        
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            features.append(returns.rolling(window=self.window).std().rename('volatility'))
            features.append(returns.rolling(window=self.window).skew().rename('skewness'))
            features.append(returns.rolling(window=self.window).kurtosis().rename('kurtosis'))
            
        if 'volume' in data.columns:
            volume_change = data['volume'].pct_change().dropna()
            features.append(volume_change.rolling(window=self.window).std().rename('volume_volatility'))
            
        if not features:
            raise ValueError("No suitable features for PCA regime detection")
            
        # Combine features
        feature_df = pd.concat(features, axis=1).dropna()
        
        if len(feature_df) < self.n_regimes:
            return pd.Series(0, index=data.index)
            
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_df)
        
        # Apply PCA
        pca_features = self.pca.fit_transform(scaled_features)
        
        # Cluster on PCA features
        labels = self.kmeans.fit_predict(pca_features)
        
        # Create regime series
        regimes = pd.Series(0, index=data.index)
        regimes.loc[feature_df.index] = labels
        
        # Forward fill to cover all dates
        regimes = regimes.ffill()
        
        return regimes


class RegimePerformance:
    """
    Container for regime-specific performance metrics.
    """
    
    def __init__(self, regime: str):
        """
        Initialize regime performance container.
        
        Args:
            regime: Regime identifier
        """
        self.regime = regime
        self.total_return = 0.0
        self.annual_return = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.max_drawdown = 0.0
        self.volatility = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        self.duration = 0  # Number of days in regime
        self.transitions = 0  # Number of transitions to this regime
        self.equity_curve = []
        self.trade_history = []
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert regime metrics to dictionary.
        
        Returns:
            Dictionary representation of regime metrics
        """
        return {
            'regime': self.regime,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'duration': self.duration,
            'transitions': self.transitions
        }


class RegimeAnalyzer:
    """
    Regime-specific performance analyzer for multi-ticker RL trading system.
    """
    
    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        detector_method: str = 'volatility',
        detector_window: int = 20,
        n_regimes: int = 3,
        output_dir: str = "regime_analysis"
    ):
        """
        Initialize regime analyzer.
        
        Args:
            performance_tracker: Performance tracker instance
            detector_method: Method for regime detection
            detector_window: Window for regime detection
            n_regimes: Number of regimes to detect
            output_dir: Directory to save analysis results
        """
        self.performance_tracker = performance_tracker
        self.detector = RegimeDetector(method=detector_method, window=detector_window, n_regimes=n_regimes)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Regime data
        self.regimes = pd.Series(dtype=int)
        self.regime_performance = {}
        self.regime_transitions = []
        self.regime_durations = {}
        
        # Market data for regime detection
        self.market_data = None
        
    def set_market_data(self, data: pd.DataFrame) -> None:
        """
        Set market data for regime detection.
        
        Args:
            data: DataFrame with market data
        """
        self.market_data = data.copy()
        
    def detect_regimes(self) -> None:
        """Detect market regimes."""
        if self.market_data is None:
            logger.error("No market data provided for regime detection")
            return
            
        # Detect regimes
        self.regimes = self.detector.detect_regimes(self.market_data)
        
        # Analyze regime transitions
        self._analyze_regime_transitions()
        
        # Analyze regime durations
        self._analyze_regime_durations()
        
        logger.info(f"Detected {len(self.regimes.unique())} regimes using {self.detector.method} method")
        
    def _analyze_regime_transitions(self) -> None:
        """Analyze regime transitions."""
        if len(self.regimes) == 0:
            return
            
        # Get regime changes
        regime_changes = self.regimes.diff().dropna()
        transition_points = regime_changes.index
        
        # Record transitions
        for i in range(len(transition_points)):
            if i == 0:
                from_regime = self.regimes.iloc[0]
            else:
                from_regime = self.regimes.loc[transition_points[i-1]]
                
            to_regime = self.regimes.loc[transition_points[i]]
            
            self.regime_transitions.append({
                'timestamp': transition_points[i],
                'from_regime': from_regime,
                'to_regime': to_regime
            })
            
    def _analyze_regime_durations(self) -> None:
        """Analyze regime durations."""
        if len(self.regimes) == 0:
            return
            
        # Group consecutive regimes
        regime_groups = (self.regimes != self.regimes.shift()).cumsum()
        
        # Calculate durations
        for regime in self.regimes.unique():
            regime_mask = self.regimes == regime
            regime_groups_for_regime = regime_groups[regime_mask]
            
            durations = []
            for group_id in regime_groups_for_regime.unique():
                group_mask = regime_groups == group_id
                duration = len(regime_groups[group_mask])
                durations.append(duration)
                
            self.regime_durations[regime] = {
                'mean_duration': np.mean(durations) if durations else 0,
                'median_duration': np.median(durations) if durations else 0,
                'min_duration': np.min(durations) if durations else 0,
                'max_duration': np.max(durations) if durations else 0,
                'std_duration': np.std(durations) if len(durations) > 1 else 0
            }
            
    def calculate_regime_performance(self) -> None:
        """Calculate regime-specific performance metrics."""
        if len(self.regimes) == 0 or not self.performance_tracker.timestamps:
            logger.error("No regimes or performance data available")
            return
            
        # Create DataFrame with timestamps and equity
        perf_df = pd.DataFrame({
            'timestamp': self.performance_tracker.timestamps,
            'equity': self.performance_tracker.equity_curve
        })
        perf_df.set_index('timestamp', inplace=True)
        
        # Align regimes with performance data
        aligned_regimes = self.regimes.reindex(perf_df.index, method='ffill')
        
        # Calculate performance for each regime
        for regime in aligned_regimes.unique():
            regime_mask = aligned_regimes == regime
            
            # Get performance data for this regime
            regime_equity = perf_df[regime_mask]['equity']
            
            if len(regime_equity) < 2:
                continue
                
            # Create regime performance object
            regime_perf = RegimePerformance(str(regime))
            
            # Calculate return metrics
            regime_perf.total_return = (regime_equity.iloc[-1] - regime_equity.iloc[0]) / regime_equity.iloc[0]
            
            # Calculate annualized return
            days_elapsed = (regime_equity.index[-1] - regime_equity.index[0]).days
            if days_elapsed > 0:
                years_elapsed = days_elapsed / 365.25
                regime_perf.annual_return = (1 + regime_perf.total_return) ** (1 / years_elapsed) - 1
            else:
                regime_perf.annual_return = 0.0
                
            # Calculate volatility
            returns = regime_equity.pct_change().dropna()
            if len(returns) > 0:
                regime_perf.volatility = returns.std() * np.sqrt(252)
                
                # Calculate Sharpe ratio
                if regime_perf.volatility > 0:
                    regime_perf.sharpe_ratio = regime_perf.annual_return / regime_perf.volatility
                else:
                    regime_perf.sharpe_ratio = 0.0
                    
                # Calculate Sortino ratio
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_deviation = downside_returns.std() * np.sqrt(252)
                    if downside_deviation > 0:
                        regime_perf.sortino_ratio = regime_perf.annual_return / downside_deviation
                    else:
                        regime_perf.sortino_ratio = 0.0
                else:
                    regime_perf.sortino_ratio = 0.0
                    
            # Calculate drawdown
            cumulative_max = regime_equity.cummax()
            drawdown = (cumulative_max - regime_equity) / cumulative_max
            regime_perf.max_drawdown = drawdown.max()
            
            # Calculate duration
            regime_perf.duration = len(regime_equity)
            
            # Calculate transitions
            regime_perf.transitions = sum(1 for t in self.regime_transitions if t['to_regime'] == regime)
            
            # Get trades for this regime
            regime_trades = []
            for trade in self.performance_tracker.trade_history:
                trade_time = pd.to_datetime(trade.get('exit_time', trade.get('entry_time')))
                if trade_time in aligned_regimes.index and aligned_regimes.loc[trade_time] == regime:
                    regime_trades.append(trade)
                    
            regime_perf.trade_history = regime_trades
            regime_perf.total_trades = len(regime_trades)
            
            # Calculate trade metrics
            if regime_perf.total_trades > 0:
                # Convert to DataFrame
                trades_df = pd.DataFrame(regime_trades)
                
                # Calculate P&L for each trade
                trades_df['pnl'] = trades_df['exit_price'] * trades_df['quantity'] - trades_df['entry_price'] * trades_df['quantity']
                trades_df['pnl'] -= trades_df.get('commission', 0) + trades_df.get('slippage', 0)
                
                # Winning and losing trades
                winning_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] < 0]
                
                regime_perf.winning_trades = len(winning_trades)
                regime_perf.win_rate = regime_perf.winning_trades / regime_perf.total_trades
                
                # Average win and loss
                if len(winning_trades) > 0:
                    regime_perf.avg_win = winning_trades['pnl'].mean()
                else:
                    regime_perf.avg_win = 0.0
                    
                if len(losing_trades) > 0:
                    regime_perf.avg_loss = losing_trades['pnl'].mean()
                else:
                    regime_perf.avg_loss = 0.0
                    
                # Profit factor
                if regime_perf.avg_loss != 0:
                    regime_perf.profit_factor = abs(regime_perf.avg_win / regime_perf.avg_loss) * (regime_perf.winning_trades / (regime_perf.total_trades - regime_perf.winning_trades))
                else:
                    regime_perf.profit_factor = float('inf') if regime_perf.avg_win > 0 else 0.0
                    
            # Store regime performance
            self.regime_performance[str(regime)] = regime_perf
            
            # Add to performance tracker
            self.performance_tracker.add_regime_performance(str(regime), regime_perf.to_dict())
            
    def generate_regime_report(self, output_path: str) -> None:
        """
        Generate regime analysis report.
        
        Args:
            output_path: Path to save report
        """
        # Calculate regime performance before generating report
        self.calculate_regime_performance()
        
        # Create report
        report = {
            'detection_method': self.detector.method,
            'detection_window': self.detector.window,
            'n_regimes': self.detector.n_regimes,
            'regime_performance': {regime: perf.to_dict() for regime, perf in self.regime_performance.items()},
            'regime_durations': self.regime_durations,
            'regime_transitions': self.regime_transitions,
            'regime_summary': self._get_regime_summary()
        }
        
        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Regime analysis report saved to {output_path}")
        
    def _get_regime_summary(self) -> Dict[str, Any]:
        """
        Get regime summary statistics.
        
        Returns:
            Dictionary with regime summary
        """
        if not self.regime_performance:
            return {}
            
        summary = {}
        
        # Overall statistics
        total_days = sum(perf.duration for perf in self.regime_performance.values())
        
        for regime, perf in self.regime_performance.items():
            summary[regime] = {
                'percentage_of_time': perf.duration / total_days if total_days > 0 else 0,
                'avg_duration': self.regime_durations.get(int(regime), {}).get('mean_duration', 0),
                'transition_probability': perf.transitions / len(self.regime_transitions) if self.regime_transitions else 0,
                'performance_rank': 0  # Will be set below
            }
            
        # Rank regimes by Sharpe ratio
        sorted_regimes = sorted(self.regime_performance.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
        for rank, (regime, _) in enumerate(sorted_regimes):
            summary[regime]['performance_rank'] = rank + 1
            
        return summary
        
    def create_regime_plots(self, output_dir: str) -> None:
        """
        Create regime analysis plots.
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate regime performance before creating plots
        self.calculate_regime_performance()
        
        # Create regime equity curve plot
        self._plot_regime_equity_curve(output_dir / "regime_equity_curve.png")
        
        # Create regime performance comparison plot
        self._plot_regime_performance_comparison(output_dir / "regime_performance_comparison.png")
        
        # Create regime duration plot
        self._plot_regime_durations(output_dir / "regime_durations.png")
        
        # Create regime transition matrix plot
        self._plot_regime_transition_matrix(output_dir / "regime_transition_matrix.png")
        
        # Create regime trade analysis plot
        self._plot_regime_trade_analysis(output_dir / "regime_trade_analysis.png")
        
        logger.info(f"Regime analysis plots saved to {output_dir}")
        
    def _plot_regime_equity_curve(self, output_path: Path) -> None:
        """
        Plot regime-specific equity curves.
        
        Args:
            output_path: Path to save plot
        """
        if not self.performance_tracker.timestamps or len(self.regimes) == 0:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Create DataFrame with timestamps, equity, and regimes
        perf_df = pd.DataFrame({
            'timestamp': self.performance_tracker.timestamps,
            'equity': self.performance_tracker.equity_curve
        })
        perf_df.set_index('timestamp', inplace=True)
        
        # Align regimes with performance data
        aligned_regimes = self.regimes.reindex(perf_df.index, method='ffill')
        
        # Plot equity curve colored by regime
        for regime in aligned_regimes.unique():
            regime_mask = aligned_regimes == regime
            regime_equity = perf_df[regime_mask]['equity']
            
            if len(regime_equity) > 0:
                plt.plot(regime_equity.index, regime_equity, label=f'Regime {regime}', linewidth=2)
                
        plt.title('Portfolio Equity Curve by Regime')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_regime_performance_comparison(self, output_path: Path) -> None:
        """
        Plot regime performance comparison.
        
        Args:
            output_path: Path to save plot
        """
        if not self.regime_performance:
            return
            
        # Create DataFrame with regime performance
        regimes = list(self.regime_performance.keys())
        metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
        
        for i, metric in enumerate(metrics):
            values = [getattr(self.regime_performance[regime], metric) for regime in regimes]
            
            bars = axes[i].bar(regimes, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Regime')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{value:.3f}', ha='center', va='bottom')
                
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_regime_durations(self, output_path: Path) -> None:
        """
        Plot regime durations.
        
        Args:
            output_path: Path to save plot
        """
        if not self.regime_durations:
            return
            
        plt.figure(figsize=(10, 6))
        
        regimes = list(self.regime_durations.keys())
        mean_durations = [self.regime_durations[regime]['mean_duration'] for regime in regimes]
        std_durations = [self.regime_durations[regime]['std_duration'] for regime in regimes]
        
        bars = plt.bar(regimes, mean_durations, yerr=std_durations, capsize=5)
        plt.title('Average Regime Duration')
        plt.xlabel('Regime')
        plt.ylabel('Duration (days)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, duration in zip(bars, mean_durations):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{duration:.1f}', ha='center', va='bottom')
                    
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_regime_transition_matrix(self, output_path: Path) -> None:
        """
        Plot regime transition matrix.
        
        Args:
            output_path: Path to save plot
        """
        if not self.regime_transitions:
            return
            
        # Create transition matrix
        regimes = sorted(list(set(t['from_regime'] for t in self.regime_transitions) | 
                            set(t['to_regime'] for t in self.regime_transitions)))
        
        transition_matrix = np.zeros((len(regimes), len(regimes)))
        
        for transition in self.regime_transitions:
            from_idx = regimes.index(transition['from_regime'])
            to_idx = regimes.index(transition['to_regime'])
            transition_matrix[from_idx, to_idx] += 1
            
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=regimes, yticklabels=regimes)
        plt.title('Regime Transition Matrix')
        plt.xlabel('To Regime')
        plt.ylabel('From Regime')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_regime_trade_analysis(self, output_path: Path) -> None:
        """
        Plot regime trade analysis.
        
        Args:
            output_path: Path to save plot
        """
        if not self.regime_performance:
            return
            
        # Create DataFrame with regime trade metrics
        regimes = list(self.regime_performance.keys())
        metrics = ['total_trades', 'win_rate', 'profit_factor']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))
        
        for i, metric in enumerate(metrics):
            values = [getattr(self.regime_performance[regime], metric) for regime in regimes]
            
            bars = axes[i].bar(regimes, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Regime')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{value:.3f}', ha='center', va='bottom')
                
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()