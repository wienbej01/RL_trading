
"""
Monte Carlo validation for multi-ticker RL trading system.

This module provides Monte Carlo validation capabilities for the multi-ticker RL trading system,
including scenario analysis, stress testing, and robustness evaluation.
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


class MonteCarloValidator:
    """
    Monte Carlo validation for multi-ticker RL trading system.
    
    This class provides comprehensive Monte Carlo validation capabilities,
    including scenario analysis, stress testing, and robustness evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Monte Carlo validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}
        self.scenarios = {}
        self.stress_tests = {}
        self.robustness_metrics = {}
        
        # Validation settings
        self.num_simulations = config.get('monte_carlo', {}).get('num_simulations', 100)
        self.confidence_level = config.get('monte_carlo', {}).get('confidence_level', 0.95)
        self.random_seed = config.get('monte_carlo', {}).get('random_seed', 42)
        self.parallel_workers = config.get('monte_carlo', {}).get('parallel_workers', 4)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Initialize components
        self.data_loader = MultiTickerDataLoader(config)
        self.feature_pipeline = MultiTickerFeaturePipeline(config.get('features', {}))
        self.evaluator = MultiTickerEvaluator(config.get('evaluation', {}))
        
    def run_scenario_analysis(
        self,
        base_data: Dict[str, pd.DataFrame],
        scenarios: Dict[str, Dict[str, Any]],
        model: Optional[MultiTickerPPOLSTMPolicy] = None
    ) -> Dict[str, Any]:
        """
        Run scenario analysis.
        
        Args:
            base_data: Base market data for all tickers
            scenarios: Dictionary of scenarios to test
            model: Trained model (optional)
            
        Returns:
            Scenario analysis results
        """
        self.scenarios = {}
        
        # Create temporary directory for results
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Run each scenario
            for scenario_name, scenario_params in scenarios.items():
                print(f"Running scenario: {scenario_name}")
                
                # Generate scenario data
                scenario_data = self._generate_scenario_data(base_data, scenario_params)
                
                # Run simulation
                if model:
                    # Use provided model
                    scenario_results = self._run_simulation_with_model(scenario_data, model)
                else:
                    # Use default strategy
                    scenario_results = self._run_simulation_with_default_strategy(scenario_data)
                
                # Store results
                self.scenarios[scenario_name] = {
                    'params': scenario_params,
                    'results': scenario_results
                }
                
                # Save intermediate results
                with open(os.path.join(temp_dir, f'{scenario_name}_results.pkl'), 'wb') as f:
                    pickle.dump(scenario_results, f)
                    
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
        # Analyze scenario results
        scenario_analysis = self._analyze_scenario_results()
        
        return scenario_analysis
        
    def run_stress_tests(
        self,
        base_data: Dict[str, pd.DataFrame],
        stress_tests: Dict[str, Dict[str, Any]],
        model: Optional[MultiTickerPPOLSTMPolicy] = None
    ) -> Dict[str, Any]:
        """
        Run stress tests.
        
        Args:
            base_data: Base market data for all tickers
            stress_tests: Dictionary of stress tests to run
            model: Trained model (optional)
            
        Returns:
            Stress test results
        """
        self.stress_tests = {}
        
        # Create temporary directory for results
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Run each stress test
            for test_name, test_params in stress_tests.items():
                print(f"Running stress test: {test_name}")
                
                # Generate stress test data
                stress_data = self._generate_stress_test_data(base_data, test_params)
                
                # Run simulation
                if model:
                    # Use provided model
                    test_results = self._run_simulation_with_model(stress_data, model)
                else:
                    # Use default strategy
                    test_results = self._run_simulation_with_default_strategy(stress_data)
                
                # Store results
                self.stress_tests[test_name] = {
                    'params': test_params,
                    'results': test_results
                }
                
                # Save intermediate results
                with open(os.path.join(temp_dir, f'{test_name}_results.pkl'), 'wb') as f:
                    pickle.dump(test_results, f)
                    
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
        # Analyze stress test results
        stress_analysis = self._analyze_stress_test_results()
        
        return stress_analysis
        
    def run_robustness_validation(
        self,
        base_data: Dict[str, pd.DataFrame],
        model: MultiTickerPPOLSTMPolicy,
        noise_levels: List[float] = [0.01, 0.02, 0.05, 0.1]
    ) -> Dict[str, Any]:
        """
        Run robustness validation.
        
        Args:
            base_data: Base market data for all tickers
            model: Trained model
            noise_levels: List of noise levels to test
            
        Returns:
            Robustness validation results
        """
        self.robustness_metrics = {}
        
        # Create temporary directory for results
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Run robustness tests for each noise level
            for noise_level in noise_levels:
                print(f"Testing robustness with noise level: {noise_level}")
                
                # Run Monte Carlo simulations with noise
                noise_results = self._run_monte_carlo_with_noise(
                    base_data, model, noise_level, self.num_simulations
                )
                
                # Store results
                self.robustness_metrics[f'noise_{noise_level}'] = noise_results
                
                # Save intermediate results
                with open(os.path.join(temp_dir, f'noise_{noise_level}_results.pkl'), 'wb') as f:
                    pickle.dump(noise_results, f)
                    
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
        # Analyze robustness results
        robustness_analysis = self._analyze_robustness_results()
        
        return robustness_analysis
        
    def run_parameter_s