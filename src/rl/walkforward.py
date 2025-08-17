"""
Walk-forward validation module for the RL trading system.

This module provides walk-forward validation functionality for
time-series cross-validation and robust model evaluation.
"""
import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WalkWindow:
    """Walk-forward window configuration."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    fold_number: int


class WalkForwardValidator:
    """
    Walk-forward validation for time-series data.
    
    This class provides robust time-series cross-validation with
    embargo periods to prevent data leakage.
    """
    
    def __init__(self, 
                 settings: Settings,
                 data_index: pd.DatetimeIndex,
                 train_days: int = 30,
                 test_days: int = 10,
                 embargo_minutes: int = 60):
        """
        Initialize walk-forward validator.
        
        Args:
            settings: Configuration settings
            data_index: Datetime index of the data
            train_days: Number of training days per fold
            test_days: Number of testing days per fold
            embargo_minutes: Embargo period between train and test
        """
        self.settings = settings
        self.data_index = data_index
        self.train_days = train_days
        self.test_days = test_days
        self.embargo_minutes = embargo_minutes
        
        # Get unique dates
        self.unique_dates = pd.to_datetime(pd.Series(data_index.date).unique())
        
        # Generate windows
        self.windows = self._generate_windows()
        
        logger.info(f"Generated {len(self.windows)} walk-forward windows")
    
    def _generate_windows(self) -> List[WalkWindow]:
        """
        Generate walk-forward windows.
        
        Returns:
            List of walk-forward windows
        """
        windows = []
        
        for i in range(0, len(self.unique_dates) - self.train_days - self.test_days + 1, self.test_days):
            train_start = pd.Timestamp(self.unique_dates[i])
            train_end = pd.Timestamp(self.unique_dates[i + self.train_days - 1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
            
            # Apply embargo
            test_start_day = self.unique_dates[i + self.train_days]
            test_start = pd.Timestamp(test_start_day) + pd.Timedelta(minutes=self.embargo_minutes)
            test_end = pd.Timestamp(self.unique_dates[i + self.train_days + self.test_days - 1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
            
            window = WalkWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_number=i // self.test_days + 1
            )
            
            windows.append(window)
        
        return windows
    
    def get_train_test_split(self, fold_number: int) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Get train and test indices for a specific fold.
        
        Args:
            fold_number: Fold number (1-based)
            
        Returns:
            Tuple of train and test indices
        """
        if fold_number < 1 or fold_number > len(self.windows):
            raise ValueError(f"Fold number {fold_number} is out of range")
        
        window = self.windows[fold_number - 1]
        
        train_mask = (self.data_index >= window.train_start) & (self.data_index <= window.train_end)
        test_mask = (self.data_index >= window.test_start) & (self.data_index <= window.test_end)
        
        train_idx = self.data_index[train_mask]
        test_idx = self.data_index[test_mask]
        
        return train_idx, test_idx
    
    def get_all_splits(self) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Get all train-test splits.
        
        Returns:
            List of train-test index tuples
        """
        return [self.get_train_test_split(fold) for fold in range(1, len(self.windows) + 1)]
    
    def get_window_info(self, fold_number: int) -> WalkWindow:
        """
        Get window information for a specific fold.
        
        Args:
            fold_number: Fold number (1-based)
            
        Returns:
            Walk window information
        """
        if fold_number < 1 or fold_number > len(self.windows):
            raise ValueError(f"Fold number {fold_number} is out of range")
        
        return self.windows[fold_number - 1]
    
    def get_train_size(self, fold_number: int) -> int:
        """
        Get training size for a specific fold.
        
        Args:
            fold_number: Fold number (1-based)
            
        Returns:
            Number of training samples
        """
        train_idx, _ = self.get_train_test_split(fold_number)
        return len(train_idx)
    
    def get_test_size(self, fold_number: int) -> int:
        """
        Get testing size for a specific fold.
        
        Args:
            fold_number: Fold number (1-based)
            
        Returns:
            Number of testing samples
        """
        _, test_idx = self.get_train_test_split(fold_number)
        return len(test_idx)
    
    def get_total_samples(self) -> int:
        """
        Get total number of samples across all folds.
        
        Returns:
            Total number of samples
        """
        return len(self.data_index)
    
    def get_coverage_percentage(self) -> float:
        """
        Get percentage of data covered by walk-forward validation.
        
        Returns:
            Coverage percentage
        """
        total_possible = len(self.unique_dates)
        covered = len(self.windows) * self.test_days
        return (covered / total_possible) * 100
    
    def validate_no_overlap(self) -> bool:
        """
        Validate that there's no overlap between train and test sets.
        
        Returns:
            True if no overlap, False otherwise
        """
        for i, window in enumerate(self.windows):
            # Check overlap with previous test set
            if i > 0:
                prev_window = self.windows[i - 1]
                if window.train_start < prev_window.test_end:
                    logger.warning(f"Overlap detected between fold {i} and {i+1}")
                    return False
            
            # Check overlap with next train set
            if i < len(self.windows) - 1:
                next_window = self.windows[i + 1]
                if window.test_end > next_window.train_start:
                    logger.warning(f"Overlap detected between fold {i+1} and {i+2}")
                    return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of walk-forward validation.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'total_folds': len(self.windows),
            'train_days': self.train_days,
            'test_days': self.test_days,
            'embargo_minutes': self.embargo_minutes,
            'total_samples': self.get_total_samples(),
            'coverage_percentage': self.get_coverage_percentage(),
            'no_overlap': self.validate_no_overlap(),
            'windows': []
        }
        
        for window in self.windows:
            window_info = {
                'fold_number': window.fold_number,
                'train_start': window.train_start,
                'train_end': window.train_end,
                'test_start': window.test_start,
                'test_end': window.test_end,
                'train_size': self.get_train_size(window.fold_number),
                'test_size': self.get_test_size(window.fold_number)
            }
            summary['windows'].append(window_info)
        
        return summary
    
    def save_summary(self, filepath: str) -> None:
        """
        Save walk-forward summary to file.
        
        Args:
            filepath: Path to save file
        """
        summary = self.get_summary()
        
        # Convert to DataFrame for easier saving
        df = pd.DataFrame([{
            'fold_number': w['fold_number'],
            'train_start': w['train_start'],
            'train_end': w['train_end'],
            'test_start': w['test_start'],
            'test_end': w['test_end'],
            'train_size': w['train_size'],
            'test_size': w['test_size']
        } for w in summary['windows']])
        
        df.to_csv(filepath, index=False)
        
        logger.info(f"Walk-forward summary saved to {filepath}")
    
    @classmethod
    def from_settings(cls, settings: Settings, data_index: pd.DatetimeIndex) -> 'WalkForwardValidator':
        """
        Create walk-forward validator from settings.
        
        Args:
            settings: Configuration settings
            data_index: Datetime index of the data
            
        Returns:
            Walk-forward validator instance
        """
        train_days = settings.get("walkforward", "train_days", default=30)
        test_days = settings.get("walkforward", "test_days", default=10)
        embargo_minutes = settings.get("walkforward", "embargo_minutes", default=60)
        
        return cls(
            settings=settings,
            data_index=data_index,
            train_days=train_days,
            test_days=test_days,
            embargo_minutes=embargo_minutes
        )


def rolling_windows(idx: pd.DatetimeIndex,
                   train_days: int,
                   test_days: int,
                   embargo_minutes: int = 0) -> Iterator[WalkWindow]:
    """
    Generate rolling train/test windows over a time index.
    
    Args:
        idx: Datetime index
        train_days: Number of training days
        test_days: Number of testing days
        embargo_minutes: Embargo period between train and test
        
    Yields:
        WalkWindow objects
    """
    day_index = pd.to_datetime(pd.Series(idx.date).unique())
    i = 0
    
    while i + train_days + test_days <= len(day_index):
        tr_start = pd.Timestamp(day_index[i])
        tr_end = pd.Timestamp(day_index[i + train_days - 1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        
        # embargo
        test_start_day = day_index[i + train_days]
        test_start = pd.Timestamp(test_start_day) + pd.Timedelta(minutes=embargo_minutes)
        te_end = pd.Timestamp(day_index[i + train_days + test_days - 1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        
        yield WalkWindow(
            train_start=tr_start,
            train_end=tr_end,
            test_start=test_start,
            test_end=te_end,
            fold_number=i // test_days + 1
        )
        
        i += test_days


class WalkForwardResults:
    """
    Container for walk-forward validation results.
    
    This class stores and manages results from walk-forward validation.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize walk-forward results.
        
        Args:
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict] = []
        self.summary_metrics: Dict[str, float] = {}
        
    def add_fold_result(self, 
                       fold_number: int,
                       train_metrics: Dict[str, float],
                       test_metrics: Dict[str, float],
                       model_path: str,
                       equity_curve: pd.Series,
                       trades: List[Dict] = None) -> None:
        """
        Add fold result.
        
        Args:
            fold_number: Fold number
            train_metrics: Training metrics
            test_metrics: Testing metrics
            model_path: Path to trained model
            equity_curve: Equity curve
            trades: List of trades
        """
        result = {
            'fold_number': fold_number,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model_path': model_path,
            'equity_curve': equity_curve,
            'trades': trades or []
        }
        
        self.results.append(result)
        
        # Save individual fold results
        fold_dir = self.output_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        pd.DataFrame([test_metrics]).to_csv(fold_dir / "test_metrics.csv", index=False)
        
        # Save equity curve
        equity_curve.to_csv(fold_dir / "equity_curve.csv", index=False)
        
        # Save trades
        if trades:
            pd.DataFrame(trades).to_csv(fold_dir / "trades.csv", index=False)
        
        logger.info(f"Added fold {fold_number} results to {fold_dir}")
    
    def calculate_summary_metrics(self) -> Dict[str, float]:
        """
        Calculate summary metrics across all folds.
        
        Returns:
            Summary metrics
        """
        if not self.results:
            return {}
        
        # Extract test metrics
        test_metrics_list = [r['test_metrics'] for r in self.results]
        
        # Calculate mean and std for each metric
        summary = {}
        for metric in test_metrics_list[0].keys():
            values = [m[metric] for m in test_metrics_list]
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
            summary[f'min_{metric}'] = np.min(values)
            summary[f'max_{metric}'] = np.max(values)
        
        # Calculate additional metrics
        returns = [m['total_return'] for m in test_metrics_list]
        sharpe_ratios = [m['sharpe_ratio'] for m in test_metrics_list]
        
        summary.update({
            'mean_total_return': np.mean(returns),
            'std_total_return': np.std(returns),
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'max_drawdown': min(m['max_drawdown'] for m in test_metrics_list),
            'win_rate': np.mean([m['win_rate'] for m in test_metrics_list]),
            'profit_factor': np.mean([m['profit_factor'] for m in test_metrics_list])
        })
        
        self.summary_metrics = summary
        return summary
    
    def save_summary(self) -> None:
        """
        Save summary metrics to file.
        """
        if not self.summary_metrics:
            self.calculate_summary_metrics()
        
        pd.DataFrame([self.summary_metrics]).to_csv(self.output_dir / "summary_metrics.csv", index=False)
        
        # Save detailed results
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                'fold_number': result['fold_number'],
                **result['test_metrics']
            })
        
        pd.DataFrame(detailed_results).to_csv(self.output_dir / "detailed_results.csv", index=False)
        
        logger.info(f"Summary metrics saved to {self.output_dir}")
    
    def get_best_fold(self, metric: str = 'sharpe_ratio') -> Dict:
        """
        Get the best performing fold based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Best fold result
        """
        if not self.results:
            return {}
        
        best_fold = max(self.results, key=lambda x: x['test_metrics'][metric])
        return best_fold
    
    def get_worst_fold(self, metric: str = 'sharpe_ratio') -> Dict:
        """
        Get the worst performing fold based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Worst fold result
        """
        if not self.results:
            return {}
        
        worst_fold = min(self.results, key=lambda x: x['test_metrics'][metric])
        return worst_fold
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Calculate stability metrics.
        
        Returns:
            Stability metrics
        """
        if not self.results:
            return {}
        
        # Extract returns and Sharpe ratios
        returns = [r['test_metrics']['total_return'] for r in self.results]
        sharpe_ratios = [r['test_metrics']['sharpe_ratio'] for r in self.results]
        
        # Calculate stability metrics
        stability = {
            'return_volatility': np.std(returns),
            'sharpe_volatility': np.std(sharpe_ratios),
            'return_consistency': 1 - (np.std(returns) / np.mean(returns)) if np.mean(returns) != 0 else 0,
            'sharpe_consistency': 1 - (np.std(sharpe_ratios) / np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) != 0 else 0,
            'max_drawdown_consistency': 1 - (max(r['test_metrics']['max_drawdown'] for r in self.results) / min(r['test_metrics']['max_drawdown'] for r in self.results))
        }
        
        return stability
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report.
        
        Returns:
            Report string
        """
        if not self.results:
            return "No results available"
        
        report = []
        report.append("=" * 60)
        report.append("WALK-FORWARD VALIDATION REPORT")
        report.append("=" * 60)
        
        # Summary metrics
        report.append("\nSUMMARY METRICS:")
        report.append("-" * 30)
        for key, value in self.summary_metrics.items():
            report.append(f"{key}: {value:.4f}")
        
        # Stability metrics
        stability = self.get_stability_metrics()
        report.append("\nSTABILITY METRICS:")
        report.append("-" * 30)
        for key, value in stability.items():
            report.append(f"{key}: {value:.4f}")
        
        # Best and worst folds
        best_fold = self.get_best_fold()
        worst_fold = self.get_worst_fold()
        
        report.append("\nBEST FOLD:")
        report.append("-" * 30)
        report.append(f"Fold {best_fold['fold_number']}:")
        for key, value in best_fold['test_metrics'].items():
            report.append(f"  {key}: {value:.4f}")
        
        report.append("\nWORST FOLD:")
        report.append("-" * 30)
        report.append(f"Fold {worst_fold['fold_number']}:")
        for key, value in worst_fold['test_metrics'].items():
            report.append(f"  {key}: {value:.4f}")
        
        # Individual fold results
        report.append("\nINDIVIDUAL FOLD RESULTS:")
        report.append("-" * 30)
        for result in self.results:
            report.append(f"Fold {result['fold_number']}:")
            for key, value in result['test_metrics'].items():
                report.append(f"  {key}: {value:.4f}")
        
        return "\n".join(report)
    
    def save_report(self, filepath: str) -> None:
        """
        Save report to file.
        
        Args:
            filepath: Path to save file
        """
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {filepath}")