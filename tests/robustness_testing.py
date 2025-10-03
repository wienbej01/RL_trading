
"""
Robustness testing suite for multi-ticker RL trading system.

This module provides a comprehensive robustness testing suite for the multi-ticker RL trading system,
including stress testing, sensitivity analysis, and resilience evaluation.
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


class RobustnessTest:
    """
    Robustness test configuration.
    
    This class represents a single robustness test configuration,
    specifying what type of robustness to test and how to evaluate it.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        test_type: str,
        test_params: Dict[str, Any],
        evaluation_metrics: List[str],
        expected_resilience: str = "medium"
    ):
        """
        Initialize robustness test.
        
        Args:
            name: Test name
            description: Test description
            test_type: Type of robustness test
            test_params: Parameters for the test
            evaluation_metrics: Metrics to evaluate
            expected_resilience: Expected resilience level
        """
        self.name = name
        self.description = description
        self.test_type = test_type
        self.test_params = test_params
        self.evaluation_metrics = evaluation_metrics
        self.expected_resilience = expected_resilience
        self.results = {}
        self.resilience_score = 0.0
        self.resilience_level = ""
        
    def run(self, data: Dict[str, pd.DataFrame], model: MultiTickerPPOLSTMPolicy) -> Dict[str, Any]:
        """
        Run the robustness test.
        
        Args:
            data: Market data for all tickers
            model: Trained model
            
        Returns:
            Test results
        """
        print(f"Running robustness test: {self.name}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Run test based on type
            if self.test_type == "data_corruption":
                results = self._run_data_corruption_test(data, model, temp_dir)
            elif self.test_type == "market_regime":
                results = self._run_market_regime_test(data, model, temp_dir)
            elif self.test_type == "parameter_sensitivity":
                results = self._run_parameter_sensitivity_test(data, model, temp_dir)
            elif self.test_type == "extreme_events":
                results = self._run_extreme_events_test(data, model, temp_dir)
            elif self.test_type == "latency_resilience":
                results = self._run_latency_resilience_test(data, model, temp_dir)
            else:
                raise ValueError(f"Unknown test type: {self.test_type}")
            
            # Calculate resilience score
            self.resilience_score = self._calculate_resilience_score(results)
            
            # Determine resilience level
            self.resilience_level = self._determine_resilience_level(self.resilience_score)
            
            # Store results
            self.results = {
                'test_results': results,
                'resilience_score': self.resilience_score,
                'resilience_level': self.resilience_level,
                'expected_resilience': self.expected_resilience,
                'meets_expectations': self._meets_expectations()
            }
            
            return self.results
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
    def _run_data_corruption_test(
        self,
        data: Dict[str, pd.DataFrame],
        model: MultiTickerPPOLSTMPolicy,
        temp_dir: str
    ) -> Dict[str, Any]:
        """
        Run data corruption robustness test.
        
        Args:
            data: Market data for all tickers
            model: Trained model
            temp_dir: Temporary directory for results
            
        Returns:
            Test results
        """
        corruption_levels = self.test_params.get('corruption_levels', [0.01, 0.05, 0.1, 0.2])
        corruption_types = self.test_params.get('corruption_types', ['noise', 'missing', 'outliers'])
        
        results = {}
        
        # Get baseline performance
        baseline_results = self._evaluate_model(data, model, temp_dir, "baseline")
        results['baseline'] = baseline_results
        
        # Test each corruption type and level
        for corruption_type in corruption_types:
            results[corruption_type] = {}
            
            for corruption_level in corruption_levels:
                # Create corrupted data
                corrupted_data = self._corrupt_data(data, corruption_type, corruption_level)
                
                # Evaluate model on corrupted data
                test_name = f"{corruption_type}_{corruption_level}"
                corrupted_results = self._evaluate_model(corrupted_data, model, temp_dir, test_name)
                
                # Calculate performance degradation
                degradation = self._calculate_performance_degradation(baseline_results, corrupted_results)
                
                results[corruption_type][corruption_level] = {
                    'results': corrupted_results,
                    'degradation': degradation
                }
        
        return results
        
    def _run_market_regime_test(
        self,
        data: Dict[str, pd.DataFrame],
        model: MultiTickerPPOLSTMPolicy,
        temp_dir: str
    ) -> Dict[str, Any]:
        """
        Run market regime robustness test.
        
        Args:
            data: Market data for all tickers
            model: Trained model
            temp_dir: Temporary directory for results
            
        Returns:
            Test results
        """
        regimes = self.test_params.get('regimes', ['trending', 'ranging', 'volatile', 'crash'])
        
        results = {}
        
        # Test each regime
        for regime in regimes:
            # Create regime-specific data
            regime_data = self._create_regime_data(data, regime)
            
            # Evaluate model on regime data
            regime_results = self._evaluate_model(regime_data, model, temp_dir, regime)
            
            results[regime] = regime_results
        
        return results
        
    def _run_parameter_sensitivity_test(
        self,
        data: Dict[str, pd.DataFrame],
        model: MultiTickerPPOLSTMPolicy,
        temp_dir: str
    ) -> Dict[str, Any]:
        """
        Run parameter sensitivity robustness test.
        
        Args:
            data: Market data for all tickers
            model: Trained model
            temp_dir: Temporary directory for results
            
        Returns:
            Test results
        """
        parameters = self.test_params.get('parameters', {})
        
        results = {}
        
        # Get baseline performance
        baseline_results = self._evaluate_model(data, model, temp_dir, "baseline")
        results['baseline'] = baseline_results
        
        # Test each parameter variation
        for param_name, param_values in parameters.items():
            results[param_name] = {}
            
            for param_value in param_values:
                # Create modified environment with parameter variation
                modified_results = self._evaluate_with_parameter_variation(
                    data, model, param_name, param_value, temp_dir
                )
                
                # Calculate performance change
                performance_change = self._calculate_performance_change(baseline_results, modified_results)
                
                results[param_name][param_value] = {
                    'results': modified_results,
                    'performance_change': performance_change
                }
        
        return results
        
    def _run_extreme_events_test(
        self,
        data: Dict[str, pd.DataFrame],
        model: MultiTickerPPOLSTMPolicy,
        temp_dir: str
    ) -> Dict[str, Any]:
        """
        Run extreme events robustness test.
        
        Args:
            data: Market data for all tickers
            model: Trained model
            temp_dir: Temporary directory for results
            
        Returns:
            Test results
        """
        events = self.test_params.get('events', ['flash_crash', 'gap_down', 'gap_up', 'high_volatility'])
        
        results = {}
        
        # Get baseline performance
        baseline_results = self._evaluate_model(data, model, temp_dir, "baseline")
        results['baseline'] = baseline_results
        
        # Test each event
        for event in events:
            # Create data with extreme event
            event_data = self._create_extreme_event_data(data, event)
            
            # Evaluate model on event data
            event_results = self._evaluate_model(event_data, model, temp_dir, event)
            
            # Calculate performance impact
            impact = self._calculate_performance_impact(baseline_results, event_results)
            
            results[event] = {
                'results': event_results,
                'impact': impact
            }
        
        return results
        
    def _run_latency_resilience_test(
        self,
        data: Dict[str, pd.DataFrame],
        model: MultiTickerPPOLSTMPolicy,
        temp_dir: str
    ) -> Dict[str, Any]:
        """
        Run latency resilience robustness test.
        
        Args:
            data: Market data for all tickers
            model: Trained model
            temp_dir: Temporary directory for results
            
        Returns:
            Test results
        """
        latency_levels = self.test_params.get('latency_levels', [0, 10, 50, 100, 500])  # in milliseconds
        
        results = {}
        
        # Get baseline performance
        baseline_results = self._evaluate_model(data, model, temp_dir, "baseline")
        results['baseline'] = baseline_results
        
        # Test each latency level
        for latency in latency_levels:
            # Evaluate model with simulated latency
            latency_results = self._evaluate_with_latency(data, model, latency, temp_dir)
            
            # Calculate performance impact
            impact = self._calculate_performance_impact(baseline_results, latency_results)
            
            results[latency] = {
                'results': latency_results,
                'impact': impact
            }
        
        return results
        
    def _corrupt_data(
        self,
        data: Dict[str, pd.DataFrame],
        corruption_type: str,
        corruption_level: float
    ) -> Dict[str, pd.DataFrame]:
        """
        Corrupt data based on type and level.
        
        Args:
            data: Original market data
            corruption_type: Type of corruption
            corruption_level: Level of corruption
            
        Returns:
            Corrupted data
        """
        corrupted_data = {}
        
        for ticker, ticker_data in data.items():
            corrupted_ticker_data = ticker_data.copy()
            
            if corruption_type == 'noise':
                # Add Gaussian noise
                noise = np.random.normal(0, corruption_level, len(ticker_data))
                corrupted_ticker_data['close'] = ticker_data['close'] * (1 + noise)
                
            elif corruption_type == 'missing':
                # Randomly set values to NaN
                mask = np.random.random(len(ticker_data)) < corruption_level
                corrupted_ticker_data.loc[mask, 'close'] = np.nan
                # Forward fill NaN values
                corrupted_ticker_data['close'] = corrupted_ticker_data['close'].fillna(method='ffill')
                
            elif corruption_type == 'outliers':
                # Add outliers
                outlier_mask = np.random.random(len(ticker_data)) < corruption_level
                outlier_values = ticker_data['close'] * np.random.choice([1.5, 0.5], size=np.sum(outlier_mask))
                corrupted_ticker_data.loc[outlier_mask, 'close'] = outlier_values
            
            corrupted_data[ticker] = corrupted_ticker_data
        
        return corrupted_data
        
    def _create_regime_data(
        self,
        data: Dict[str, pd.DataFrame],
        regime: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Create data for a specific market regime.
        
        Args:
            data: Original market data
            regime: Market regime to simulate
            
        Returns:
            Regime-specific data
        """
        regime_data = {}
        
        for ticker, ticker_data in data.items():
            regime_ticker_data = ticker_data.copy()
            
            if regime == 'trending':
                # Add strong trend
                trend = np.linspace(0, 0.5, len(ticker_data))
                regime_ticker_data['close'] = ticker_data['close'] * (1 + trend)
                
            elif regime == 'ranging':
                # Add range-bound movement
                range_factor = 0.1 * np.sin(np.linspace(0, 4*np.pi, len(ticker_data)))
                regime_ticker_data['close'] = ticker_data['close'] * (1 + range_factor)
                
            elif regime == 'volatile':
                # Increase volatility
                volatility_factor = 1 + 2 * np.random.random(len(ticker_data))
                regime_ticker_data['close'] = ticker_data['close'] * volatility_factor
                
            elif regime == 'crash':
                # Simulate market crash
                crash_point = len(ticker_data) // 2
                crash_factor = np.ones(len(ticker_data))
                crash_factor[crash_point:] = np.linspace(1, 0.7, len(ticker_data) - crash_point)
                regime_ticker_data['close'] = ticker_data['close'] * crash_factor
            
            regime_data[ticker] = regime_ticker_data
        
        return regime_data
        
    def _create_extreme_event_data(
        self,
        data: Dict[str, pd.DataFrame],
        event: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Create data with extreme events.
        
        Args:
            data: Original market data
            event: Type of extreme event
            
        Returns:
            Data with extreme events
        """
        event_data = {}
        
        for ticker, ticker_data in data.items():
            event_ticker_data = ticker_data.copy()
            
            if event == 'flash_crash':
                # Simulate flash crash (quick drop and recovery)
                crash_start = len(ticker_data) // 3
                crash_end = crash_start + 10
                
                # Drop
                event_ticker_data.loc[crash_start:crash_end, 'close'] *= 0.8
                # Recovery
                event_ticker_data.loc[crash_end+1:crash_end+20, 'close'] *= 1.25
                
            elif event == 'gap_down':
                # Simulate gap down
                gap_point = len(ticker_data) // 2
                event_ticker_data.loc[gap_point:, 'close'] *= 0.9
                
            elif event == 'gap_up':
                # Simulate gap up
                gap_point = len(ticker_data) // 2
                event_ticker_data.loc[gap_point:, 'close'] *= 1.1
                
            elif event == 'high_volatility':
                # Simulate high volatility period
                vol_start = len(ticker_data) // 3
                vol_end = 2 * len(ticker_data) // 3
                
                high_vol_factor = 1 + 3 * np.random.random(vol_end - vol_start)
                event_ticker_data.loc[vol_start:vol_end, 'close'] *= high_vol_factor
            
            event_data[ticker] = event_ticker_data
        
        return event_data
        
    def _evaluate_model(
        self,
        data: Dict[str, pd.DataFrame],
        model: MultiTickerPPOLSTMPolicy,
        temp_dir: str,
        test_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate model on data.
        
        Args:
            data: Market data for all tickers
            model: Trained model
            temp_dir: Temporary directory for results
            test_name: Name of the test
            
        Returns:
            Evaluation results
        """
        # Initialize evaluator
        evaluator = MultiTickerEvaluator(self.config.get('evaluation', {}))
        
        # Evaluate model
        results = evaluator.evaluate_model(model, data)
        
        return results
        
    def _evaluate_with_parameter_variation(
        self,
        data: Dict[str, pd.DataFrame],
        model: MultiTickerPPOLSTMPolicy,
        param_name: str,
        param_value: Any,
        temp_dir: str
    ) -> Dict[str, Any]:
        """
        Evaluate model with parameter variation.
        
        Args:
            data: Market data for all tickers
            model: Trained model
            param_name: Name of parameter to vary
            param_value: Value to set parameter to
            temp_dir: Temporary directory for results
            
        Returns:
            Evaluation results
        """
        # Create modified config
        modified_config = self.config.copy()
        
        # Set parameter variation
        if param_name == 'position_size':
            modified_config['environment']['position_size'] = param_value
        elif param_name == 'max_positions':
            modified_config['environment']['max_positions'] = param_value
        elif param_name == 'commission':
            modified_config['environment']['commission'] = param_value
        elif param_name == 'slippage':
            modified_config['environment']['slippage'] = param_value
        else:
            # Try to set parameter in environment config
            modified_config['environment'][param_name] = param_value
        
        # Create environment with modified config
        env = MultiTickerIntradayRLEnv(modified_config, data)
        
        # Evaluate model
        evaluator = MultiTickerEvaluator(modified_config.get('evaluation', {}))
        results = evaluator.evaluate_model_in_env(model, env)
        
        return results
        
    def _evaluate_with_latency(
        self,
        data: Dict[str, pd.DataFrame],
        model: MultiTickerPPOLSTMPolicy,
        latency: int,
        temp_dir: str
    ) -> Dict[str, Any]:
        """
        Evaluate model with simulated latency.
        
        Args:
            data: Market data for all tickers
            model: Trained model
            latency: Latency in milliseconds
            temp_dir: Temporary directory for results
            
        Returns:
            Evaluation results
        """
        # Create environment with latency simulation
        env = MultiTickerIntradayRLEnv(self.config, data)
        
        # Add latency to environment
        env.latency = latency
        
        # Evaluate model
        evaluator = MultiTickerEvaluator(self.config.get('evaluation', {}))
        results = evaluator.evaluate_model_in_env(model, env)
        
        return results
        
    def _calculate_performance_degradation(
        self,
        baseline_results: Dict[str, Any],
        test_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate performance degradation.
        
        Args:
            baseline_results: Baseline performance results
            test_results: Test performance results
            
        Returns:
            Performance degradation for each metric
        """
        degradation = {}
        
        for metric in self.evaluation_metrics:
            if metric in baseline_results and metric in test_results:
                baseline_value = baseline_results[metric]
                test_value = test_results[metric]
                
                if baseline_value != 0:
                    # For metrics where higher is better (e.g., Sharpe ratio)
                    if metric in ['sharpe_ratio', 'total_return', 'win_rate']:
                        degradation[metric] = (baseline_value - test_value) / abs(baseline_value)
                    # For metrics where lower is better (e.g., max drawdown)
                    elif metric in ['max_drawdown']:
                        degradation[metric] = (test_value - baseline_value) / abs(baseline_value)
                    else:
                        # Default: assume higher is better
                        degradation[metric] = (baseline_value - test_value) / abs(baseline_value)
                else:
                    degradation[metric] = 0.0 if test_value == 0 else float('inf')
        
        return degradation
        
    def _calculate_performance_change(
        self,
        baseline_results: Dict[str, Any],
        test_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate performance change.
        
        Args:
            baseline_results: Baseline performance results
            test_results: Test performance results
            
        Returns:
            Performance change for each metric
        """
        change = {}
        
        for metric in self.evaluation_metrics:
            if metric in baseline_results and metric in test_results:
                baseline_value = baseline_results[metric]
                test_value = test_results[metric]
                
                if baseline_value != 0:
                    change[metric] = (test_value - baseline_value) / abs(baseline_value)
                else:
                    change[metric] = 0.0 if test_value == 0 else float('inf')
        
        return change
        
    def _calculate_performance_impact(
        self,
        baseline_results: Dict[str, Any],
        test_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate performance impact.
        
        Args:
            baseline_results: Baseline performance results
            test_results: Test performance results
            
        Returns:
            Performance impact for each metric
        """
        impact = {}
        
        for metric in self.evaluation_metrics:
            if metric in baseline_results and metric in test_results:
                baseline_value = baseline_results[metric]
                test_value = test_results[metric]
                
                if baseline_value != 0:
                    # For metrics where higher is better (e.g., Sharpe ratio)
                    if metric in ['sharpe_ratio', 'total_return', 'win_rate']:
                        impact[metric] = (test_value - baseline_value) / abs(baseline_value)
                    # For metrics where lower is better (e.g., max drawdown)
                    elif metric in ['max_drawdown']:
                        impact[metric] = (baseline_value - test_value) / abs(baseline_value)
                    else:
                        # Default: assume higher is better
                        impact[metric] = (test_value - baseline_value) / abs(baseline_value)
                else:
                    impact[metric] = 0.0 if test_value == 0 else float('inf')
        
        return impact
        
    def _calculate_resilience_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate resilience score from test results.
        
        Args:
            results: Test results
            
        Returns:
            Resilience score (0-1)
        """
        # This is a simplified resilience score calculation
        # In practice, this would be more sophisticated and tailored to each test type
        
        if self.test_type == "data_corruption":
            # Average performance across corruption types and levels
            total_score = 0.0
            count = 0
            
            for corruption_type, corruption_levels in results.items():
                if corruption_type == 'baseline':
                    continue
                    
                for corruption_level, level_results in corruption_levels.items():
                    degradation = level_results.get('degradation', {})
                    
                    # Calculate average degradation
                    avg_degradation = np.mean(list(degradation.values())) if degradation else 0.0
                    
                    # Convert to resilience score (lower degradation = higher resilience)
                    resilience = max(0, 1 - avg_degradation)
                    
                    total_score += resilience
                    count += 1
            
            return total_score / count if count > 0 else 0.0
            
        elif self.test_type == "market_regime":
            # Minimum performance across regimes
            min_performance = float('inf')
            
            for regime, regime_results in results.items():
                # Calculate average performance across metrics
                performance_values = [regime_results.get(metric, 0) for metric in self.evaluation_metrics]
                avg_performance = np.mean(performance_values)
                
                min_performance = min(min_performance, avg_performance)
            
            # Normalize to 0-1 range
            return max(0, min_performance)
            
        elif self.test_type == "parameter_sensitivity":
            # Average performance across parameter variations
            total_score = 0.0
            count = 0
            
            baseline = results.get('baseline', {})
            baseline_performance = np.mean([baseline.get(metric, 0) for metric in self.evaluation_metrics])
            
            for param_name, param_values in results.items():
                if param_name == 'baseline':
                    continue
                    
                for param_value, value_results in param_values.items():
                    performance_change = value_results.get('performance_change', {})
                    
                    # Calculate average performance change
                    avg_change = np.mean(list(performance_change.values())) if performance_change else 0.0
                    
                    # Convert to resilience score (lower change = higher resilience)
                    resilience = max(0, 1 - abs(avg_change))
                    
                    total_score += resilience
                    count += 1
            
            return total_score / count if count > 0 else 0.0
            
        elif self.test_type == "extreme_events":
            # Minimum performance across events
            min_performance = float('inf')
            
            for event, event_results in results.items():
                if event == 'baseline':
                    continue
                    
                impact = event_results.get('impact', {})
                
                # Calculate average impact
                avg_impact = np.mean(list(impact.values())) if impact else 0.0
                
                # Convert to resilience score (lower negative impact = higher resilience)
                resilience = max(0, 1 + avg_impact)
                
                min_performance = min(min_performance, resilience)
            
            return min_performance if min_performance != float('inf') else 0.0
            
        elif self.test_type == "latency_resilience":
            # Performance at highest latency level
            max_latency = max([k for k in results.keys() if isinstance(k, int)])
            latency_results = results.get(max_latency, {})
            
            impact = latency_results.get('impact', {})
            
            # Calculate average impact
            avg_impact = np.mean(list(impact.values())) if impact else 0.0
            
            # Convert to resilience score (lower negative impact = higher resilience)
            resilience = max(0, 1 + avg_impact)
            
            return resilience
        
        return 0.0
        
    def _determine_resilience_level(self, resilience_score: float) -> str:
        """
        Determine resilience level from resilience score.
        
        Args:
            resilience_score: Resilience score (0-1)
            
        Returns:
            Resilience level
        """
        if resilience_score >= 0.8:
            return "high"
        elif resilience_score >= 0.6:
            return "medium"
        elif resilience_score >= 0.4:
            return "low"
        else:
            return "very_low"
            
    def _meets_expectations(self) -> bool:
        """
        Check if resilience meets expectations.
        
        Returns:
            True if resilience meets or exceeds expectations
        """
        resilience_levels = {
            'very_low': 0,
            'low': 1,
            'medium': 2,
            'high': 3
        }
        
        actual_level = resilience_levels.get(self.resilience_level, 0)
        expected_level = resilience_levels.get(self.expected_resilience, 0)
        
        return actual_level >= expected_level
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert robustness test to dictionary.
        
        Returns:
            Robustness test as dictionary
        """
        return {
            'name': self.name,
            'description': self.description,
            'test_type': self.test_type,
            'test_params': self.test_params,
            'evaluation_metrics': self.evaluation_metrics,
            'expected_resilience': self.expected_resilience,
            'results': self.results,
            'resilience_score': self.resilience_score,
            'resilience_level': self.resilience_level
        }


class RobustnessTestingFramework:
    """
    Robustness testing framework for multi-ticker RL trading system.
    
    This class provides a comprehensive framework for running robustness tests
    on the multi-ticker RL trading system, allowing for systematic evaluation
    of system resilience under various adverse conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize robustness testing framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tests = []
        self.results = {}
        self.summary = {}
        
        # Framework settings
        self.output_dir = config.get('robustness_testing', {}).get('output_dir', '/tmp/robustness_results')
        self.parallel_workers = config.get('robustness_testing', {}).get('parallel_workers', 4)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def add_test(self, test: RobustnessTest):
        """
        Add a robustness test.
        
        Args:
            test: Robustness test to add
        """
        self.tests.append(test)
        
    def add_data_corruption_tests(self, evaluation_metrics: List[str]) -> List[RobustnessTest]:
        """
        Add data corruption robustness tests.
        
        Args:
            evaluation_metrics: Metrics to evaluate
            
        Returns:
            List of added robustness tests
        """
        added_tests = []
        
        # Test 1: Noise corruption
        test = RobustnessTest(
            name="noise_corruption",
            description="Test resilience to noise corruption in market data",
            test_type="data_corruption",
            test_params={
                'corruption_levels': [0.01, 0.05, 0.1, 0.2],
                'corruption_types': ['noise']
            },
            evaluation_metrics=evaluation_metrics,
            expected_resilience="medium"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 2: Missing data corruption
        test = RobustnessTest(
            name="missing_data_corruption",
            description="Test resilience to missing data in market data",
            test_type="data_corruption",
            test_params={
                'cor