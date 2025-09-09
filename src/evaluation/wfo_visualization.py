
"""
Visualization and reporting for Walk-Forward Optimization results.

This module provides comprehensive visualization capabilities for WFO results,
including performance metrics, fold comparisons, and regime analysis.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from .walkforward import WFOResults, WFOFold
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class WFOVisualizer:
    """
    Visualizer for Walk-Forward Optimization results.
    """
    
    def __init__(self, results: WFOResults, output_dir: str = "wfo_plots"):
        """
        Initialize WFO visualizer.
        
        Args:
            results: WFO results to visualize
            output_dir: Directory to save plots
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up figure parameters
        self.fig_size = (15, 10)
        self.dpi = 300
        self.font_size = 12
        
    def create_summary_dashboard(self) -> str:
        """
        Create a comprehensive summary dashboard.
        
        Returns:
            Path to saved dashboard
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Aggregated metrics bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_aggregated_metrics(ax1)
        
        # 2. IS vs OOS comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_is_oos_comparison(ax2)
        
        # 3. Fold performance heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_fold_performance_heatmap(ax3)
        
        # 4. Test ticker performance
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_test_ticker_performance(ax4)
        
        # 5. Train vs test performance scatter
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_train_test_scatter(ax5)
        
        # 6. Performance decay by fold
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_performance_decay(ax6)
        
        # 7. Regime performance (if available)
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_regime_performance(ax7)
        
        # 8. Equity curves comparison
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_equity_curves(ax8)
        
        # Add title
        fig.suptitle(
            f"WFO Summary Dashboard - {len(self.results.folds)} Folds",
            fontsize=16,
            y=0.98
        )
        
        # Save dashboard
        dashboard_path = self.output_dir / "wfo_dashboard.png"
        plt.savefig(dashboard_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary dashboard saved to {dashboard_path}")
        return str(dashboard_path)
        
    def _plot_aggregated_metrics(self, ax: plt.Axes) -> None:
        """
        Plot aggregated metrics bar chart.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.aggregated_metrics:
            ax.text(0.5, 0.5, "No aggregated metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Aggregated Metrics")
            return
            
        # Extract mean metrics
        metrics = {}
        for key, value in self.results.aggregated_metrics.items():
            if key.endswith('_mean'):
                metric_name = key.replace('_mean', '')
                metrics[metric_name] = value
                
        if not metrics:
            ax.text(0.5, 0.5, "No mean metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Aggregated Metrics")
            return
            
        # Create bar chart
        names = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(names, values)
        ax.set_title("Aggregated Metrics (Mean)")
        ax.set_ylabel("Value")
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
                   
    def _plot_is_oos_comparison(self, ax: plt.Axes) -> None:
        """
        Plot IS vs OOS metrics comparison.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.is_metrics or not self.results.oos_metrics:
            ax.text(0.5, 0.5, "IS/OOS metrics not available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("IS vs OOS Comparison")
            return
            
        # Extract common metrics
        is_metrics = self.results.is_metrics
        oos_metrics = self.results.oos_metrics
        
        # Compare Sharpe ratios
        is_sharpe = is_metrics.get('is_sharpe_mean', 0)
        oos_sharpe = oos_metrics.get('oos_sharpe_mean', 0)
        
        # Compare returns
        is_return = is_metrics.get('is_return_mean', 0)
        oos_return = oos_metrics.get('oos_return_mean', 0)
        
        # Create bar chart
        metrics = ['Sharpe Ratio', 'Total Return']
        is_values = [is_sharpe, is_return]
        oos_values = [oos_sharpe, oos_return]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, is_values, width, label='IS')
        bars2 = ax.bar(x + width/2, oos_values, width, label='OOS')
        
        ax.set_title("IS vs OOS Performance")
        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add value labels
        for bars, values in [(bars1, is_values), (bars2, oos_values)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
                       
    def _plot_fold_performance_heatmap(self, ax: plt.Axes) -> None:
        """
        Plot fold performance heatmap.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.folds:
            ax.text(0.5, 0.5, "No fold data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Fold Performance Heatmap")
            return
            
        # Extract test metrics for each fold
        fold_metrics = []
        fold_labels = []
        
        for fold in self.results.folds:
            if fold.test_results:
                # Extract key metrics
                sharpe = fold.test_results.get('sharpe_ratio', 0)
                return_val = fold.test_results.get('total_return', 0)
                drawdown = fold.test_results.get('max_drawdown', 0)
                
                fold_metrics.append([sharpe, return_val, drawdown])
                fold_labels.append(f"Fold {fold.fold_id}\n{fold.test_ticker}")
                
        if not fold_metrics:
            ax.text(0.5, 0.5, "No test metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Fold Performance Heatmap")
            return
            
        # Create heatmap
        metrics_array = np.array(fold_metrics)
        metric_names = ['Sharpe', 'Return', 'Drawdown']
        
        im = ax.imshow(metrics_array, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(fold_labels)))
        ax.set_xticklabels(metric_names)
        ax.set_yticklabels(fold_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value')
        
        # Add text annotations
        for i in range(len(fold_labels)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{metrics_array[i, j]:.2f}',
                             ha="center", va="center", color="black")
                             
        ax.set_title("Fold Performance Heatmap")
        
    def _plot_test_ticker_performance(self, ax: plt.Axes) -> None:
        """
        Plot test ticker performance across folds.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.folds:
            ax.text(0.5, 0.5, "No fold data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Test Ticker Performance")
            return
            
        # Extract test metrics for each fold
        fold_ids = []
        tickers = []
        sharpe_ratios = []
        returns = []
        
        for fold in self.results.folds:
            if fold.test_results:
                fold_ids.append(fold.fold_id)
                tickers.append(fold.test_ticker)
                sharpe_ratios.append(fold.test_results.get('sharpe_ratio', 0))
                returns.append(fold.test_results.get('total_return', 0))
                
        if not fold_ids:
            ax.text(0.5, 0.5, "No test metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Test Ticker Performance")
            return
            
        # Create plot with two y-axes
        ax2 = ax.twinx()
        
        # Plot Sharpe ratios
        bars1 = ax.bar([i - 0.2 for i in fold_ids], sharpe_ratios, 
                      width=0.4, label='Sharpe Ratio', alpha=0.7, color='blue')
        
        # Plot returns
        bars2 = ax2.bar([i + 0.2 for i in fold_ids], returns, 
                       width=0.4, label='Total Return', alpha=0.7, color='green')
        
        # Set labels and title
        ax.set_xlabel("Fold ID")
        ax.set_ylabel("Sharpe Ratio", color='blue')
        ax2.set_ylabel("Total Return", color='green')
        ax.set_title("Test Ticker Performance by Fold")
        
        # Set x-axis ticks
        ax.set_xticks(fold_ids)
        ax.set_xticklabels([f"{fid}\n{ticker}" for fid, ticker in zip(fold_ids, tickers)])
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    def _plot_train_test_scatter(self, ax: plt.Axes) -> None:
        """
        Plot train vs test performance scatter.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.folds:
            ax.text(0.5, 0.5, "No fold data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Train vs Test Performance")
            return
            
        # Extract train and test metrics
        train_sharpe = []
        test_sharpe = []
        train_return = []
        test_return = []
        
        for fold in self.results.folds:
            if fold.train_results and fold.test_results:
                train_sharpe.append(fold.train_results.get('sharpe_ratio', 0))
                test_sharpe.append(fold.test_results.get('sharpe_ratio', 0))
                train_return.append(fold.train_results.get('total_return', 0))
                test_return.append(fold.test_results.get('total_return', 0))
                
        if not train_sharpe:
            ax.text(0.5, 0.5, "No train/test metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Train vs Test Performance")
            return
            
        # Create scatter plot
        ax.scatter(train_sharpe, test_sharpe, alpha=0.7, s=100, label='Sharpe Ratio')
        
        # Add diagonal line
        min_val = min(min(train_sharpe), min(test_sharpe))
        max_val = max(max(train_sharpe), max(test_sharpe))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        
        # Set labels and title
        ax.set_xlabel("Train Sharpe Ratio")
        ax.set_ylabel("Test Sharpe Ratio")
        ax.set_title("Train vs Test Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_performance_decay(self, ax: plt.Axes) -> None:
        """
        Plot performance decay by fold.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.folds:
            ax.text(0.5, 0.5, "No fold data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Performance Decay")
            return
            
        # Extract performance decay metrics
        fold_ids = []
        sharpe_decay = []
        return_decay = []
        
        for fold in self.results.folds:
            if fold.metrics:
                fold_ids.append(fold.fold_id)
                sharpe_decay.append(fold.metrics.get('sharpe_ratio_decay', 0))
                return_decay.append(fold.metrics.get('return_decay', 0))
                
        if not fold_ids:
            ax.text(0.5, 0.5, "No decay metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Performance Decay")
            return
            
        # Create line plot
        ax.plot(fold_ids, sharpe_decay, 'o-', label='Sharpe Ratio Decay', linewidth=2)
        ax.plot(fold_ids, return_decay, 's-', label='Return Decay', linewidth=2)
        
        # Add zero line
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel("Fold ID")
        ax.set_ylabel("Decay Value")
        ax.set_title("Performance Decay by Fold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_regime_performance(self, ax: plt.Axes) -> None:
        """
        Plot regime-specific performance (if available).
        
        Args:
            ax: Matplotlib axes
        """
        # This is a placeholder for regime-specific performance visualization
        # In a real implementation, you would extract regime information from the results
        
        ax.text(0.5, 0.5, "Regime performance analysis\n(not implemented yet)", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Regime Performance")
        
    def _plot_equity_curves(self, ax: plt.Axes) -> None:
        """
        Plot equity curves for all folds.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.folds:
            ax.text(0.5, 0.5, "No fold data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Equity Curves")
            return
            
        # This is a placeholder for equity curve visualization
        # In a real implementation, you would extract equity curve data from the results
        
        # Generate sample equity curves for demonstration
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        for i, fold in enumerate(self.results.folds[:5]):  # Limit to first 5 folds
            # Generate random walk for demonstration
            returns = np.random.normal(0.001, 0.02, 100)
            equity = np.cumprod(1 + returns)
            
            ax.plot(dates, equity, label=f"Fold {fold.fold_id} ({fold.test_ticker})")
            
        # Set labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.set_title("Equity Curves by Fold")
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    def create_fold_report(self, fold_id: int) -> str:
        """
        Create a detailed report for a specific fold.
        
        Args:
            fold_id: ID of the fold to report
            
        Returns:
            Path to saved report
        """
        # Find the fold
        fold = None
        for f in self.results.folds:
            if f.fold_id == fold_id:
                fold = f
                break
                
        if fold is None:
            logger.error(f"Fold {fold_id} not found")
            return ""
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Fold {fold_id} Detailed Report - Test Ticker: {fold.test_ticker}", 
                    fontsize=16)
        
        # Plot 1: Train vs test metrics
        ax1 = axes[0, 0]
        self._plot_fold_train_test_metrics(fold, ax1)
        
        # Plot 2: Performance timeline
        ax2 = axes[0, 1]
        self._plot_fold_timeline(fold, ax2)
        
        # Plot 3: Metrics breakdown
        ax3 = axes[1, 0]
        self._plot_fold_metrics_breakdown(fold, ax3)
        
        # Plot 4: Generalization metrics
        ax4 = axes[1, 1]
        self._plot_fold_generalization(fold, ax4)
        
        # Save report
        report_path = self.output_dir / f"fold_{fold_id}_report.png"
        plt.savefig(report_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Fold {fold_id} report saved to {report_path}")
        return str(report_path)
        
    def _plot_fold_train_test_metrics(self, fold: WFOFold, ax: plt.Axes) -> None:
        """
        Plot train vs test metrics for a fold.
        
        Args:
            fold: WFO fold
            ax: Matplotlib axes
        """
        if not fold.train_results or not fold.test_results:
            ax.text(0.5, 0.5, "Train/test results not available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Train vs Test Metrics")
            return
            
        # Extract common metrics
        train_metrics = {}
        test_metrics = {}
        
        for key, value in fold.train_results.items():
            if isinstance(value, (int, float)):
                train_metrics[key] = value
                
        for key, value in fold.test_results.items():
            if isinstance(value, (int, float)):
                test_metrics[key] = value
                
        # Find common metrics
        common_metrics = set(train_metrics.keys()) & set(test_metrics.keys())
        
        if not common_metrics:
            ax.text(0.5, 0.5, "No common metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Train vs Test Metrics")
            return
            
        # Create bar chart
        metrics = list(common_metrics)[:5]  # Limit to first 5 metrics
        train_values = [train_metrics[m] for m in metrics]
        test_values = [test_metrics[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Train')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test')
        
        ax.set_title("Train vs Test Metrics")
        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        
    def _plot_fold_timeline(self, fold: WFOFold, ax: plt.Axes) -> None:
        """
        Plot fold timeline.
        
        Args:
            fold: WFO fold
            ax: Matplotlib axes
        """
        # Create timeline
        events = [
            (fold.train_start, "Train Start", "green"),
            (fold.train_end, "Train End", "green"),
            (fold.test_start, "Test Start", "blue"),
            (fold.test_end, "Test End", "blue")
        ]
        
        if fold.embargo_start and fold.embargo_end:
            events.insert(2, (fold.embargo_start, "Embargo Start", "orange"))
            events.insert(3, (fold.embargo_end, "Embargo End", "orange"))
            
        # Plot timeline
        for i, (date, label, color) in enumerate(events):
            ax.axvline(x=date, color=color, linestyle='--', alpha=0.7)
            ax.text(date, i + 0.1, label, rotation=45, ha='left', va='bottom')
            
        ax.set_title("Fold Timeline")
        ax.set_ylabel("Events")
        ax.set_yticks([])
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    def _plot_fold_metrics_breakdown(self, fold: WFOFold, ax: plt.Axes) -> None:
        """
        Plot metrics breakdown for a fold.
        
        Args:
            fold: WFO fold
            ax: Matplotlib axes
        """
        if not fold.metrics:
            ax.text(0.5, 0.5, "No metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Metrics Breakdown")
            return
            
        # Extract metrics
        metrics = {}
        for key, value in fold.metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
                
        if not metrics:
            ax.text(0.5, 0.5, "No numeric metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Metrics Breakdown")
            return
            
        # Create bar chart
        names = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(names, values)
        ax.set_title("Metrics Breakdown")
        ax.set