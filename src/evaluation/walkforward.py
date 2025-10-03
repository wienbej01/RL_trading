"""
Walk-Forward Optimization (WFO) with Leave-One-Ticker-Out (LOT-O) cross-validation.

This module implements WFO for multi-ticker RL trading systems, including
LOT-O cross-validation, embargo periods, fold aggregation, and regime-aware
fold splitting.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from ..utils.logging import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)


class WFOConfig:
    """
    Configuration for Walk-Forward Optimization.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        test_size: float = 0.2,
        embargo_days: int = 5,
        regime_aware: bool = True,
        regime_column: str = 'regime',
        random_state: Optional[int] = None,
        output_dir: str = 'wfo_results'
    ):
        """
        Initialize WFO configuration.
        
        Args:
            n_folds: Number of folds for cross-validation
            test_size: Proportion of data to use for testing in each fold
            embargo_days: Number of days to embargo between training and test sets
            regime_aware: Whether to use regime-aware fold splitting
            regime_column: Column name for regime information
            random_state: Random state for reproducibility
            output_dir: Directory to save WFO results
        """
        self.n_folds = n_folds
        self.test_size = test_size
        self.embargo_days = embargo_days
        self.regime_aware = regime_aware
        self.regime_column = regime_column
        self.random_state = random_state
        self.output_dir = output_dir
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def save(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        config_dict = {
            'n_folds': self.n_folds,
            'test_size': self.test_size,
            'embargo_days': self.embargo_days,
            'regime_aware': self.regime_aware,
            'regime_column': self.regime_column,
            'random_state': self.random_state,
            'output_dir': self.output_dir
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    @classmethod
    def load(cls, filepath: str) -> 'WFOConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            WFOConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)


class WFOFold:
    """
    Represents a single fold in Walk-Forward Optimization.
    """
    
    def __init__(
        self,
        fold_id: int,
        train_tickers: List[str],
        test_ticker: str,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        embargo_start: Optional[datetime] = None,
        embargo_end: Optional[datetime] = None
    ):
        """
        Initialize WFO fold.
        
        Args:
            fold_id: Fold identifier
            train_tickers: List of tickers for training
            test_ticker: Ticker for testing
            train_start: Start date for training
            train_end: End date for training
            test_start: Start date for testing
            test_end: End date for testing
            embargo_start: Start date for embargo period
            embargo_end: End date for embargo period
        """
        self.fold_id = fold_id
        self.train_tickers = train_tickers
        self.test_ticker = test_ticker
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.embargo_start = embargo_start
        self.embargo_end = embargo_end
        
        # Results storage
        self.train_results = {}
        self.test_results = {}
        self.model_path = None
        self.metrics = {}
        
    def get_train_period(self) -> Tuple[datetime, datetime]:
        """
        Get training period.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        return self.train_start, self.train_end
        
    def get_test_period(self) -> Tuple[datetime, datetime]:
        """
        Get testing period.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        return self.test_start, self.test_end
        
    def get_embargo_period(self) -> Optional[Tuple[datetime, datetime]]:
        """
        Get embargo period.
        
        Returns:
            Tuple of (start_date, end_date) or None if no embargo
        """
        if self.embargo_start and self.embargo_end:
            return self.embargo_start, self.embargo_end
        return None
        
    def save_results(self, filepath: str) -> None:
        """
        Save fold results to file.
        
        Args:
            filepath: Path to save results
        """
        results = {
            'fold_id': self.fold_id,
            'train_tickers': self.train_tickers,
            'test_ticker': self.test_ticker,
            'train_start': self.train_start.isoformat(),
            'train_end': self.train_end.isoformat(),
            'test_start': self.test_start.isoformat(),
            'test_end': self.test_end.isoformat(),
            'embargo_start': self.embargo_start.isoformat() if self.embargo_start else None,
            'embargo_end': self.embargo_end.isoformat() if self.embargo_end else None,
            'train_results': self.train_results,
            'test_results': self.test_results,
            'model_path': self.model_path,
            'metrics': self.metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
            
    @classmethod
    def load_results(cls, filepath: str) -> 'WFOFold':
        """
        Load fold results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            WFOFold instance
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
            
        fold = cls(
            fold_id=results['fold_id'],
            train_tickers=results['train_tickers'],
            test_ticker=results['test_ticker'],
            train_start=datetime.fromisoformat(results['train_start']),
            train_end=datetime.fromisoformat(results['train_end']),
            test_start=datetime.fromisoformat(results['test_start']),
            test_end=datetime.fromisoformat(results['test_end']),
            embargo_start=datetime.fromisoformat(results['embargo_start']) if results['embargo_start'] else None,
            embargo_end=datetime.fromisoformat(results['embargo_end']) if results['embargo_end'] else None
        )
        
        fold.train_results = results['train_results']
        fold.test_results = results['test_results']
        fold.model_path = results['model_path']
        fold.metrics = results['metrics']
        
        return fold


class WFOSplitter:
    """
    Splits data for Walk-Forward Optimization with Leave-One-Ticker-Out cross-validation.
    """
    
    def __init__(self, config: WFOConfig):
        """
        Initialize WFO splitter.
        
        Args:
            config: WFO configuration
        """
        self.config = config
        
    def split(
        self,
        data: Dict[str, pd.DataFrame],
        dates: List[datetime]
    ) -> List[WFOFold]:
        """
        Split data into WFO folds.
        
        Args:
            data: Dictionary mapping tickers to their data
            dates: List of dates for splitting
            
        Returns:
            List of WFOFold instances
        """
        tickers = list(data.keys())
        n_tickers = len(tickers)
        
        if n_tickers < 2:
            raise ValueError("Need at least 2 tickers for LOT-O cross-validation")
            
        folds = []
        
        # Create one fold for each ticker as test set
        for i, test_ticker in enumerate(tickers):
            train_tickers = [t for t in tickers if t != test_ticker]
            
            # Split dates into training and testing periods
            n_dates = len(dates)
            test_size = int(n_dates * self.config.test_size)
            train_size = n_dates - test_size
            
            # Ensure we have enough data for both training and testing
            if train_size < 10 or test_size < 5:
                logger.warning(f"Insufficient data for fold {i}, skipping")
                continue
                
            train_dates = dates[:train_size]
            test_dates = dates[train_size:]
            
            # Create embargo period
            embargo_start = test_dates[0] - timedelta(days=self.config.embargo_days)
            embargo_end = test_dates[0]
            
            # Adjust embargo start if it overlaps with training
            if embargo_start < train_dates[-1]:
                embargo_start = train_dates[-1] + timedelta(days=1)
                
            fold = WFOFold(
                fold_id=i,
                train_tickers=train_tickers,
                test_ticker=test_ticker,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                embargo_start=embargo_start if self.config.embargo_days > 0 else None,
                embargo_end=embargo_end if self.config.embargo_days > 0 else None
            )
            
            folds.append(fold)
            
        return folds
        
    def split_regime_aware(
        self,
        data: Dict[str, pd.DataFrame],
        dates: List[datetime],
        regime_data: pd.DataFrame
    ) -> List[WFOFold]:
        """
        Split data into WFO folds with regime-aware splitting.
        
        Args:
            data: Dictionary mapping tickers to their data
            dates: List of dates for splitting
            regime_data: DataFrame with regime information
            
        Returns:
            List of WFOFold instances
        """
        tickers = list(data.keys())
        n_tickers = len(tickers)
        
        if n_tickers < 2:
            raise ValueError("Need at least 2 tickers for LOT-O cross-validation")
            
        # Group dates by regime
        regime_groups = {}
        for date in dates:
            if date in regime_data.index:
                regime = regime_data.loc[date, self.config.regime_column]
                if regime not in regime_groups:
                    regime_groups[regime] = []
                regime_groups[regime].append(date)
                
        folds = []
        
        # Create one fold for each ticker as test set
        for i, test_ticker in enumerate(tickers):
            train_tickers = [t for t in tickers if t != test_ticker]
            
            # Split each regime group into training and testing
            train_dates = []
            test_dates = []
            
            for regime, regime_dates in regime_groups.items():
                n_regime_dates = len(regime_dates)
                test_size = int(n_regime_dates * self.config.test_size)
                train_size = n_regime_dates - test_size
                
                if train_size < 5 or test_size < 2:
                    logger.warning(f"Insufficient data for regime {regime} in fold {i}, using all dates for training")
                    train_dates.extend(regime_dates)
                    continue
                    
                train_dates.extend(regime_dates[:train_size])
                test_dates.extend(regime_dates[train_size:])
                
            # Sort dates
            train_dates = sorted(train_dates)
            test_dates = sorted(test_dates)
            
            # Create embargo period
            embargo_start = test_dates[0] - timedelta(days=self.config.embargo_days)
            embargo_end = test_dates[0]
            
            # Adjust embargo start if it overlaps with training
            if embargo_start < train_dates[-1]:
                embargo_start = train_dates[-1] + timedelta(days=1)
                
            fold = WFOFold(
                fold_id=i,
                train_tickers=train_tickers,
                test_ticker=test_ticker,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                embargo_start=embargo_start if self.config.embargo_days > 0 else None,
                embargo_end=embargo_end if self.config.embargo_days > 0 else None
            )
            
            folds.append(fold)
            
        return folds


class WFOResults:
    """
    Container for WFO results and metrics.
    """
    
    def __init__(self, config: WFOConfig):
        """
        Initialize WFO results.
        
        Args:
            config: WFO configuration
        """
        self.config = config
        self.folds = []
        self.aggregated_metrics = {}
        self.oos_metrics = {}
        self.is_metrics = {}
        
    def add_fold(self, fold: WFOFold) -> None:
        """
        Add a fold to the results.
        
        Args:
            fold: WFOFold instance
        """
        self.folds.append(fold)
        
    def aggregate_metrics(self) -> Dict[str, float]:
        """
        Aggregate metrics across all folds.
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not self.folds:
            return {}
            
        # Collect all metrics
        all_metrics = []
        for fold in self.folds:
            if fold.metrics:
                all_metrics.append(fold.metrics)
                
        if not all_metrics:
            return {}
            
        # Aggregate by metric name
        aggregated = {}
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [metrics.get(metric_name, 0) for metrics in all_metrics]
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)
            aggregated[f'{metric_name}_min'] = np.min(values)
            aggregated[f'{metric_name}_max'] = np.max(values)
            
        self.aggregated_metrics = aggregated
        return aggregated
        
    def calculate_oos_metrics(self) -> Dict[str, float]:
        """
        Calculate out-of-sample metrics.
        
        Returns:
            Dictionary of OOS metrics
        """
        if not self.folds:
            return {}
            
        # Collect test results
        all_test_results = []
        for fold in self.folds:
            if fold.test_results:
                all_test_results.append(fold.test_results)
                
        if not all_test_results:
            return {}
            
        # Calculate OOS metrics
        oos = {}
        
        # Example OOS metrics
        test_sharpe = [results.get('sharpe_ratio', 0) for results in all_test_results]
        test_returns = [results.get('total_return', 0) for results in all_test_results]
        test_drawdowns = [results.get('max_drawdown', 0) for results in all_test_results]
        
        oos['oos_sharpe_mean'] = np.mean(test_sharpe)
        oos['oos_sharpe_std'] = np.std(test_sharpe)
        oos['oos_return_mean'] = np.mean(test_returns)
        oos['oos_return_std'] = np.std(test_returns)
        oos['oos_drawdown_mean'] = np.mean(test_drawdowns)
        oos['oos_drawdown_std'] = np.std(test_drawdowns)
        
        # Calculate stability metrics
        oos['sharpe_stability'] = 1.0 - (np.std(test_sharpe) / (np.mean(np.abs(test_sharpe)) + 1e-8))
        oos['return_stability'] = 1.0 - (np.std(test_returns) / (np.mean(np.abs(test_returns)) + 1e-8))
        
        self.oos_metrics = oos
        return oos
        
    def calculate_is_metrics(self) -> Dict[str, float]:
        """
        Calculate in-sample metrics.
        
        Returns:
            Dictionary of IS metrics
        """
        if not self.folds:
            return {}
            
        # Collect train results
        all_train_results = []
        for fold in self.folds:
            if fold.train_results:
                all_train_results.append(fold.train_results)
                
        if not all_train_results:
            return {}
            
        # Calculate IS metrics
        is_metrics = {}
        
        # Example IS metrics
        train_sharpe = [results.get('sharpe_ratio', 0) for results in all_train_results]
        train_returns = [results.get('total_return', 0) for results in all_train_results]
        train_drawdowns = [results.get('max_drawdown', 0) for results in all_train_results]
        
        is_metrics['is_sharpe_mean'] = np.mean(train_sharpe)
        is_metrics['is_sharpe_std'] = np.std(train_sharpe)
        is_metrics['is_return_mean'] = np.mean(train_returns)
        is_metrics['is_return_std'] = np.std(train_returns)
        is_metrics['is_drawdown_mean'] = np.mean(train_drawdowns)
        is_metrics['is_drawdown_std'] = np.std(train_drawdowns)
        
        self.is_metrics = is_metrics
        return is_metrics
        
    def save(self, filepath: str) -> None:
        """
        Save WFO results to file.
        
        Args:
            filepath: Path to save results
        """
        results = {
            'config': {
                'n_folds': self.config.n_folds,
                'test_size': self.config.test_size,
                'embargo_days': self.config.embargo_days,
                'regime_aware': self.config.regime_aware,
                'regime_column': self.config.regime_column,
                'random_state': self.config.random_state,
                'output_dir': self.config.output_dir
            },
            'folds': [
                {
                    'fold_id': fold.fold_id,
                    'train_tickers': fold.train_tickers,
                    'test_ticker': fold.test_ticker,
                    'train_start': fold.train_start.isoformat(),
                    'train_end': fold.train_end.isoformat(),
                    'test_start': fold.test_start.isoformat(),
                    'test_end': fold.test_end.isoformat(),
                    'embargo_start': fold.embargo_start.isoformat() if fold.embargo_start else None,
                    'embargo_end': fold.embargo_end.isoformat() if fold.embargo_end else None,
                    'train_results': fold.train_results,
                    'test_results': fold.test_results,
                    'model_path': fold.model_path,
                    'metrics': fold.metrics
                }
                for fold in self.folds
            ],
            'aggregated_metrics': self.aggregated_metrics,
            'oos_metrics': self.oos_metrics,
            'is_metrics': self.is_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
            
    @classmethod
    def load(cls, filepath: str) -> 'WFOResults':
        """
        Load WFO results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            WFOResults instance
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
            
        # Reconstruct config
        config = WFOConfig(
            n_folds=results['config']['n_folds'],
            test_size=results['config']['test_size'],
            embargo_days=results['config']['embargo_days'],
            regime_aware=results['config']['regime_aware'],
            regime_column=results['config']['regime_column'],
            random_state=results['config']['random_state'],
            output_dir=results['config']['output_dir']
        )
        
        # Create WFOResults instance
        wfo_results = cls(config)
        wfo_results.aggregated_metrics = results['aggregated_metrics']
        wfo_results.oos_metrics = results['oos_metrics']
        wfo_results.is_metrics = results['is_metrics']
        
        # Reconstruct folds
        for fold_data in results['folds']:
            fold = WFOFold(
                fold_id=fold_data['fold_id'],
                train_tickers=fold_data['train_tickers'],
                test_ticker=fold_data['test_ticker'],
                train_start=datetime.fromisoformat(fold_data['train_start']),
                train_end=datetime.fromisoformat(fold_data['train_end']),
                test_start=datetime.fromisoformat(fold_data['test_start']),
                test_end=datetime.fromisoformat(fold_data['test_end']),
                embargo_start=datetime.fromisoformat(fold_data['embargo_start']) if fold_data['embargo_start'] else None,
                embargo_end=datetime.fromisoformat(fold_data['embargo_end']) if fold_data['embargo_end'] else None
            )
            
            fold.train_results = fold_data['train_results']
            fold.test_results = fold_data['test_results']
            fold.model_path = fold_data['model_path']
            fold.metrics = fold_data['metrics']
            
            wfo_results.add_fold(fold)
            
        return wfo_results


class WalkForwardOptimizer:
    """
    Main class for Walk-Forward Optimization with Leave-One-Ticker-Out cross-validation.
    """
    
    def __init__(
        self,
        config: WFOConfig,
        train_func: Callable,
        evaluate_func: Callable,
        data_loader_func: Callable
    ):
        """
        Initialize Walk-Forward Optimizer.
        
        Args:
            config: WFO configuration
            train_func: Function to train models
            evaluate_func: Function to evaluate models
            data_loader_func: Function to load data for tickers and dates
        """
        self.config = config
        self.train_func = train_func
        self.evaluate_func = evaluate_func
        self.data_loader_func = data_loader_func
        self.splitter = WFOSplitter(config)
        self.results = WFOResults(config)
        
    def run(
        self,
        tickers: List[str],
        dates: List[datetime],
        regime_data: Optional[pd.DataFrame] = None
    ) -> WFOResults:
        """
        Run Walk-Forward Optimization.
        
        Args:
            tickers: List of ticker symbols
            dates: List of dates for WFO
            regime_data: Optional DataFrame with regime information
            
        Returns:
            WFOResults instance
        """
        logger.info(f"Starting WFO with {len(tickers)} tickers and {len(dates)} dates")
        
        # Load data for all tickers
        data = self.data_loader_func(tickers, dates)
        
        # Create folds
        if self.config.regime_aware and regime_data is not None:
            logger.info("Using regime-aware fold splitting")
            folds = self.splitter.split_regime_aware(data, dates, regime_data)
        else:
            logger.info("Using standard fold splitting")
            folds = self.splitter.split(data, dates)
            
        logger.info(f"Created {len(folds)} folds")
        
        # Process each fold
        for i, fold in enumerate(folds):
            logger.info(f"Processing fold {i+1}/{len(folds)}")
            
            # Load training data
            train_dates = pd.date_range(fold.train_start, fold.train_end).tolist()
            train_data = self.data_loader_func(fold.train_tickers, train_dates)
            
            # Train model
            logger.info(f"Training model on tickers: {fold.train_tickers}")
            model_path = os.path.join(
                self.config.output_dir,
                f"model_fold_{fold.fold_id}.pkl"
            )
            
            train_results = self.train_func(
                train_data,
                model_path=model_path,
                fold_id=fold.fold_id
            )
            
            fold.train_results = train_results
            fold.model_path = model_path
            
            # Load test data
            test_dates = pd.date_range(fold.test_start, fold.test_end).tolist()
            test_data = self.data_loader_func([fold.test_ticker], test_dates)
            
            # Evaluate model
            logger.info(f"Evaluating model on ticker: {fold.test_ticker}")
            test_results = self.evaluate_func(
                test_data,
                model_path=model_path,
                fold_id=fold.fold_id
            )
            
            fold.test_results = test_results
            
            # Calculate fold metrics
            fold.metrics = self._calculate_fold_metrics(fold)
            
            # Save fold results
            fold_path = os.path.join(
                self.config.output_dir,
                f"fold_{fold.fold_id}_results.pkl"
            )
            fold.save_results(fold_path)
            
            # Add to results
            self.results.add_fold(fold)
            
        # Calculate aggregated metrics
        self.results.aggregate_metrics()
        self.results.calculate_oos_metrics()
        self.results.calculate_is_metrics()
        
        # Save final results
        results_path = os.path.join(self.config.output_dir, "wfo_results.pkl")
        self.results.save(results_path)
        
        logger.info("WFO completed successfully")
        return self.results
        
    def _calculate_fold_metrics(self, fold: WFOFold) -> Dict[str, float]:
        """
        Calculate metrics for a fold.
        
        Args:
            fold: WFOFold instance
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Extract metrics from train results
        if fold.train_results:
            for key, value in fold.train_results.items():
                if isinstance(value, (int, float)):
                    metrics[f'train_{key}'] = value
                    
        # Extract metrics from test results
        if fold.test_results:
            for key, value in fold.test_results.items():
                if isinstance(value, (int, float)):
                    metrics[f'test_{key}'] = value
                    
        # Calculate generalization metrics
        if 'train_sharpe_ratio' in metrics and 'test_sharpe_ratio' in metrics:
            metrics['sharpe_ratio_decay'] = (
                metrics['train_sharpe_ratio'] - metrics['test_sharpe_ratio']
            ) / (abs(metrics['train_sharpe_ratio']) + 1e-8)
            
        if 'train_total_return' in metrics and 'test_total_return' in metrics:
            metrics['return_decay'] = (
                metrics['train_total_return'] - metrics['test_total_return']
            ) / (abs(metrics['train_total_return']) + 1e-8)
            
        return metrics
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive WFO report.
        
        Returns:
            Dictionary with report data
        """
        if not self.results.folds:
            return {}
            
        report = {
            'config': {
                'n_folds': self.config.n_folds,
                'test_size': self.config.test_size,
                'embargo_days': self.config.embargo_days,
                'regime_aware': self.config.regime_aware
            },
            'summary': {
                'n_folds_completed': len(self.results.folds),
                'aggregated_metrics': self.results.aggregated_metrics,
                'oos_metrics': self.results.oos_metrics,
                'is_metrics': self.results.is_metrics
            },
            'folds': []
        }
        
        # Add fold details
        for fold in self.results.folds:
            fold_report = {
                'fold_id': fold.fold_id,
                'train_tickers': fold.train_tickers,
                'test_ticker': fold.test_ticker,
                'train_period': {
                    'start': fold.train_start.isoformat(),
                    'end': fold.train_end.isoformat()
                },
                'test_period': {
                    'start': fold.test_start.isoformat(),
                    'end': fold.test_end.isoformat()
                },
                'embargo_period': {
                    'start': fold.embargo_start.isoformat() if fold.embargo_start else None,
                    'end': fold.embargo_end.isoformat() if fold.embargo_end else None
                },
                'metrics': fold.metrics
            }
            
            report['folds'].append(fold_report)
            
        return report


def create_wfo_config_from_yaml(config_path: str) -> WFOConfig:
    """
    Create WFOConfig from YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        WFOConfig instance
    """
    config = load_config(config_path)
    wfo_config_dict = config.get('walkforward', {})
    
    return WFOConfig(
        n_folds=wfo_config_dict.get('n_folds', 5),
        test_size=wfo_config_dict.get('test_size', 0.2),
        embargo_days=wfo_config_dict.get('embargo_days', 5),
        regime_aware=wfo_config_dict.get('regime_aware', True),
        regime_column=wfo_config_dict.get('regime_column', 'regime'),
        random_state=wfo_config_dict.get('random_state'),
        output_dir=wfo_config_dict.get('output_dir', 'wfo_results')
    )