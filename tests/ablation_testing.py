"""
Ablation testing framework for multi-ticker RL trading system.

This module provides a comprehensive ablation testing framework for the multi-ticker RL trading system,
allowing for systematic evaluation of individual components and their contributions to overall performance.
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


class AblationTest:
    """
    Ablation test configuration.
    
    This class represents a single ablation test configuration,
    specifying which components to enable/disable and the expected impact.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        component: str,
        baseline_config: Dict[str, Any],
        modified_config: Dict[str, Any],
        expected_impact: str = "unknown"
    ):
        """
        Initialize ablation test.
        
        Args:
            name: Test name
            description: Test description
            component: Component being tested
            baseline_config: Baseline configuration
            modified_config: Modified configuration with component disabled/modified
            expected_impact: Expected impact on performance
        """
        self.name = name
        self.description = description
        self.component = component
        self.baseline_config = baseline_config
        self.modified_config = modified_config
        self.expected_impact = expected_impact
        self.results = {}
        self.comparison = {}
        
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run the ablation test.
        
        Args:
            data: Market data for all tickers
            
        Returns:
            Test results
        """
        print(f"Running ablation test: {self.name}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Run baseline test
            print("  Running baseline test...")
            baseline_results = self._run_configuration(self.baseline_config, data, temp_dir, "baseline")
            
            # Run modified test
            print("  Running modified test...")
            modified_results = self._run_configuration(self.modified_config, data, temp_dir, "modified")
            
            # Compare results
            print("  Comparing results...")
            self.comparison = self._compare_results(baseline_results, modified_results)
            
            # Store results
            self.results = {
                'baseline': baseline_results,
                'modified': modified_results,
                'comparison': self.comparison
            }
            
            return self.results
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
    def _run_configuration(
        self,
        config: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        temp_dir: str,
        prefix: str
    ) -> Dict[str, Any]:
        """
        Run a specific configuration.
        
        Args:
            config: Configuration to run
            data: Market data for all tickers
            temp_dir: Temporary directory for results
            prefix: Prefix for result files
            
        Returns:
            Configuration results
        """
        # Initialize components
        data_loader = MultiTickerDataLoader(config)
        feature_pipeline = MultiTickerFeaturePipeline(config.get('features', {}))
        trainer = MultiTickerRLTrainer(config)
        evaluator = MultiTickerEvaluator(config.get('evaluation', {}))
        
        # Load and preprocess data
        processed_data = {}
        for ticker, ticker_data in data.items():
            # Extract features
            features = feature_pipeline.transform(ticker_data)
            processed_data[ticker] = features
        
        # Train model
        model = trainer.train(processed_data)
        
        # Save model
        model_path = os.path.join(temp_dir, f"{prefix}_model.pkl")
        trainer.save_model(model, model_path)
        
        # Evaluate model
        eval_results = evaluator.evaluate_model(model, processed_data)
        
        return eval_results
        
    def _compare_results(
        self,
        baseline_results: Dict[str, Any],
        modified_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare baseline and modified results.
        
        Args:
            baseline_results: Baseline configuration results
            modified_results: Modified configuration results
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        # Compare key metrics
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for metric in key_metrics:
            if metric in baseline_results and metric in modified_results:
                baseline_value = baseline_results[metric]
                modified_value = modified_results[metric]
                
                # Calculate relative change
                if baseline_value != 0:
                    relative_change = (modified_value - baseline_value) / abs(baseline_value)
                else:
                    relative_change = 0 if modified_value == 0 else float('inf')
                
                comparison[metric] = {
                    'baseline': baseline_value,
                    'modified': modified_value,
                    'absolute_change': modified_value - baseline_value,
                    'relative_change': relative_change,
                    'impact': 'positive' if relative_change > 0 else 'negative' if relative_change < 0 else 'neutral'
                }
        
        # Overall assessment
        positive_count = sum(1 for m in comparison.values() if m['impact'] == 'positive')
        negative_count = sum(1 for m in comparison.values() if m['impact'] == 'negative')
        
        if positive_count > negative_count:
            overall_impact = 'positive'
        elif negative_count > positive_count:
            overall_impact = 'negative'
        else:
            overall_impact = 'neutral'
        
        comparison['overall_impact'] = overall_impact
        comparison['assessment'] = self._assess_impact(comparison)
        
        return comparison
        
    def _assess_impact(self, comparison: Dict[str, Any]) -> str:
        """
        Assess the impact of the ablation test.
        
        Args:
            comparison: Comparison results
            
        Returns:
            Impact assessment
        """
        # Check if expected impact matches actual impact
        if self.expected_impact == "unknown":
            return "Impact was unknown, now determined to be " + comparison['overall_impact']
        
        if self.expected_impact == comparison['overall_impact']:
            return f"Expected impact ({self.expected_impact}) matches actual impact"
        else:
            return f"Expected impact ({self.expected_impact}) does not match actual impact ({comparison['overall_impact']})"
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ablation test to dictionary.
        
        Returns:
            Ablation test as dictionary
        """
        return {
            'name': self.name,
            'description': self.description,
            'component': self.component,
            'expected_impact': self.expected_impact,
            'results': self.results,
            'comparison': self.comparison
        }


class AblationTestingFramework:
    """
    Ablation testing framework for multi-ticker RL trading system.
    
    This class provides a comprehensive framework for running ablation tests
    on the multi-ticker RL trading system, allowing for systematic evaluation
    of individual components and their contributions to overall performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ablation testing framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tests = []
        self.results = {}
        self.summary = {}
        
        # Framework settings
        self.output_dir = config.get('ablation_testing', {}).get('output_dir', '/tmp/ablation_results')
        self.parallel_workers = config.get('ablation_testing', {}).get('parallel_workers', 4)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def add_test(self, test: AblationTest):
        """
        Add an ablation test.
        
        Args:
            test: Ablation test to add
        """
        self.tests.append(test)
        
    def add_reward_ablation_tests(self, baseline_config: Dict[str, Any]) -> List[AblationTest]:
        """
        Add reward function ablation tests.
        
        Args:
            baseline_config: Baseline configuration
            
        Returns:
            List of added ablation tests
        """
        added_tests = []
        
        # Test 1: Disable asymmetric drawdown penalty
        modified_config = baseline_config.copy()
        modified_config['environment']['reward_components'] = modified_config['environment'].get('reward_components', {})
        modified_config['environment']['reward_components']['asymmetric_drawdown_penalty'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_asymmetric_drawdown_penalty",
            description="Disable asymmetric drawdown penalty in reward function",
            component="reward_function",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 2: Disable Lagrangian activity shaping
        modified_config = baseline_config.copy()
        modified_config['environment']['reward_components'] = modified_config['environment'].get('reward_components', {})
        modified_config['environment']['reward_components']['lagrangian_activity_shaping'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_lagrangian_activity_shaping",
            description="Disable Lagrangian activity shaping in reward function",
            component="reward_function",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 3: Disable microstructure PCA features
        modified_config = baseline_config.copy()
        modified_config['environment']['reward_components'] = modified_config['environment'].get('reward_components', {})
        modified_config['environment']['reward_components']['microstructure_pca'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_microstructure_pca",
            description="Disable microstructure PCA features in reward function",
            component="reward_function",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 4: Disable opportunity/capture shaping
        modified_config = baseline_config.copy()
        modified_config['environment']['reward_components'] = modified_config['environment'].get('reward_components', {})
        modified_config['environment']['reward_components']['opportunity_capture_shaping'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_opportunity_capture_shaping",
            description="Disable opportunity/capture shaping in reward function",
            component="reward_function",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 5: Disable potential-based reward shaping
        modified_config = baseline_config.copy()
        modified_config['environment']['reward_components'] = modified_config['environment'].get('reward_components', {})
        modified_config['environment']['reward_components']['potential_based_shaping'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_potential_based_shaping",
            description="Disable potential-based reward shaping",
            component="reward_function",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        return added_tests
        
    def add_feature_ablation_tests(self, baseline_config: Dict[str, Any]) -> List[AblationTest]:
        """
        Add feature engineering ablation tests.
        
        Args:
            baseline_config: Baseline configuration
            
        Returns:
            List of added ablation tests
        """
        added_tests = []
        
        # Test 1: Disable technical indicators
        modified_config = baseline_config.copy()
        modified_config['features']['technical'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_technical_indicators",
            description="Disable technical indicators in feature pipeline",
            component="feature_engineering",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 2: Disable microstructure features
        modified_config = baseline_config.copy()
        modified_config['features']['microstructure'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_microstructure_features",
            description="Disable microstructure features in feature pipeline",
            component="feature_engineering",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 3: Disable time features
        modified_config = baseline_config.copy()
        modified_config['features']['time'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_time_features",
            description="Disable time features in feature pipeline",
            component="feature_engineering",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 4: Disable normalization
        modified_config = baseline_config.copy()
        modified_config['features']['normalization'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_normalization",
            description="Disable feature normalization",
            component="feature_engineering",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        return added_tests
        
    def add_model_ablation_tests(self, baseline_config: Dict[str, Any]) -> List[AblationTest]:
        """
        Add model architecture ablation tests.
        
        Args:
            baseline_config: Baseline configuration
            
        Returns:
            List of added ablation tests
        """
        added_tests = []
        
        # Test 1: Disable LSTM layers
        modified_config = baseline_config.copy()
        modified_config['training']['policy_kwargs'] = modified_config['training'].get('policy_kwargs', {})
        modified_config['training']['policy_kwargs']['lstm_hidden_size'] = 0
        
        test = AblationTest(
            name="disable_lstm_layers",
            description="Disable LSTM layers in PPO policy",
            component="model_architecture",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 2: Reduce LSTM size
        modified_config = baseline_config.copy()
        modified_config['training']['policy_kwargs'] = modified_config['training'].get('policy_kwargs', {})
        modified_config['training']['policy_kwargs']['lstm_hidden_size'] = 32
        
        test = AblationTest(
            name="reduce_lstm_size",
            description="Reduce LSTM hidden size to 32",
            component="model_architecture",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 3: Disable curriculum learning
        modified_config = baseline_config.copy()
        modified_config['training']['curriculum_learning'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_curriculum_learning",
            description="Disable curriculum learning",
            component="model_architecture",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 4: Disable entropy annealing
        modified_config = baseline_config.copy()
        modified_config['training']['entropy_annealing'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_entropy_annealing",
            description="Disable entropy annealing",
            component="model_architecture",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        return added_tests
        
    def add_training_ablation_tests(self, baseline_config: Dict[str, Any]) -> List[AblationTest]:
        """
        Add training process ablation tests.
        
        Args:
            baseline_config: Baseline configuration
            
        Returns:
            List of added ablation tests
        """
        added_tests = []
        
        # Test 1: Reduce training timesteps
        modified_config = baseline_config.copy()
        modified_config['training']['total_timesteps'] = 10000
        
        test = AblationTest(
            name="reduce_training_timesteps",
            description="Reduce training timesteps to 10000",
            component="training_process",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 2: Increase batch size
        modified_config = baseline_config.copy()
        modified_config['training']['batch_size'] = 256
        
        test = AblationTest(
            name="increase_batch_size",
            description="Increase batch size to 256",
            component="training_process",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="unknown"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 3: Reduce learning rate
        modified_config = baseline_config.copy()
        modified_config['training']['learning_rate'] = 0.0001
        
        test = AblationTest(
            name="reduce_learning_rate",
            description="Reduce learning rate to 0.0001",
            component="training_process",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="unknown"
        )
        self.add_test(test)
        added_tests.append(test)
        
        # Test 4: Disable WFO
        modified_config = baseline_config.copy()
        modified_config['walkforward'] = {'enabled': False}
        
        test = AblationTest(
            name="disable_wfo",
            description="Disable walk-forward optimization",
            component="training_process",
            baseline_config=baseline_config,
            modified_config=modified_config,
            expected_impact="negative"
        )
        self.add_test(test)
        added_tests.append(test)
        
        return added_tests
        
    def run_tests(self, data: Dict[str, pd.DataFrame], parallel: bool = True) -> Dict[str, Any]:
        """
        Run all ablation tests.
        
        Args:
            data: Market data for all tickers
            parallel: Whether to run tests in parallel
            
        Returns:
            Test results
        """
        print(f"Running {len(self.tests)} ablation tests...")
        
        if parallel and len(self.tests) > 1:
            # Run tests in parallel
            self._run_tests_parallel(data)
        else:
            # Run tests sequentially
            self._run_tests_sequential(data)
        
        # Generate summary
        self.summary = self._generate_summary()
        
        # Save results
        self._save_results()
        
        return self.results
        
    def _run_tests_sequential(self, data: Dict[str, pd.DataFrame]):
        """
        Run ablation tests sequentially.
        
        Args:
            data: Market data for all tickers
        """
        for test in self.tests:
            try:
                result = test.run(data)
                self.results[test.name] = result
                print(f"Completed test: {test.name}")
            except Exception as e:
                print(f"Error running test {test.name}: {str(e)}")
                self.results[test.name] = {'error': str(e)}
                
    def _run_tests_parallel(self, data: Dict[str, pd.DataFrame]):
        """
        Run ablation tests in parallel.
        
        Args:
            data: Market data for all tickers
        """
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(test.run, data): test
                for test in self.tests
            }
            
            # Collect results
            for future in as_completed(future_to_test):
                test = future_to_test[future]
                try:
                    result = future.result()
                    self.results[test.name] = result
                    print(f"Completed test: {test.name}")
                except Exception as e:
                    print(f"Error running test {test.name}: {str(e)}")
                    self.results[test.name] = {'error': str(e)}
                    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary of ablation test results.
        
        Returns:
            Summary of results
        """
        summary = {
            'total_tests': len(self.tests),
            'completed_tests': len([r for r in self.results.values() if 'error' not in r]),
            'failed_tests': len([r for r in self.results.values() if 'error' in r]),
            'component_impacts': {},
            'key_findings': []
        }
        
        # Analyze component impacts
        component_tests = {}
        for test in self.tests:
            if test.name in self.results and 'error' not in self.results[test.name]:
                component = test.component
                if component not in component_tests:
                    component_tests[component] = []
                component_tests[component].append(test)
        
        for component, tests in component_tests.items():
            positive_count = sum(1 for test in tests if test.comparison.get('overall_impact') == 'positive')
            negative_count = sum(1 for test in tests if test.comparison.get('overall_impact') == 'negative')
            neutral_count = sum(1 for test in tests if test.comparison.get('overall_impact') == 'neutral')
            
            summary['component_impacts'][component] = {
                'total_tests': len(tests),
                'positive_impacts': positive_count,
                'negative_impacts': negative_count,
                'neutral_impacts': neutral_count,
                'overall_assessment': 'critical' if negative_count > positive_count else 'beneficial' if positive_count > negative_count else 'neutral'
            }
        
        # Generate key findings
        for component, impact in summary['component_impacts'].items():
            if impact['overall_assessment'] == 'critical':
                summary['key_findings'].append(f"{component} is critical to system performance")
            elif impact['overall_assessment'] == 'beneficial':
                summary['key_findings'].append(f"{component} provides benefits to system performance")
            else:
                summary['key_findings'].append(f"{component} has neutral impact on system performance")
        
        return summary
        
    def _save_results(self):
        """Save ablation test results."""
        # Save individual test results
        for test_name, result in self.results.items():
            result_path = os.path.join(self.output_dir, f"{test_name}_result.json")
            
            # Convert to JSON-serializable format
            json_result = self._convert_to_json_serializable(result)
            
            with open(result_path, 'w') as f:
                json.dump(json_result, f, indent=2)
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "ablation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        # Generate and save report
        report_path = os.path.join(self.output_dir, "ablation_report.html")
        self._generate_report(report_path)
        
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)
            
    def _generate_report(self, report_path: str):
        """
        Generate ablation test report.
        
        Args:
            report_path: Path to save report
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ablation Testing Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .positive { color: green; }
                .negative { color: red; }
                .neutral { color: orange; }
                .critical { font-weight: bold; color: red; }
                .beneficial { font-weight: bold; color: green; }
            </style>
        </head>
        <body>
            <h1>Ablation Testing Report</h1>
            
            <h2>Summary</h2>
            <ul>
                <li>Total Tests: {total_tests}</li>
                <li>Completed Tests: {completed_tests}</li>
                <li>Failed Tests: {failed_tests}</li>
            </ul>
            
            <h2>Component Impacts</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Total Tests</th>
                    <th>Positive Impacts</th>
                    <th>Negative Impacts</th>
                    <th>Neutral Impacts</th>
                    <th>Overall Assessment</th>
                </tr>
                {component_rows}
            </table>
            
            <h2>Key Findings</h2>
            <ul>
                {key_findings}
            </ul>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Component</th>
                    <th>Overall Impact</th>
                    <th>Expected Impact</th>
                    <th>Status</th>
                </tr>
                {test_rows}
            </table>
        </body>
        </html>
        """
        
        # Generate component rows
        component_rows = ""
        for component, impact in self.summary['component_impacts'].items():
            assessment_class = impact['overall_assessment']
            component_rows += f"""
                <tr>
                    <td>{component}</td>
                    <td>{impact['total_tests']}</td>
                    <td class="positive">{impact['positive_impacts']}</td>
                    <td class="negative">{impact['negative_impacts']}</td>
                    <td class="neutral">{impact['neutral_impacts']}</td>
                    <td class="{assessment_class}">{impact['overall_assessment']}</td>
                </tr>
            """
        
        # Generate key findings
        key_findings = ""
        for finding in self.summary['key_findings']:
            key_findings += f"<li>{finding}</li>"
        
        # Generate test rows
        test_rows = ""
        for test in self.tests:
            if test.name in self.results:
                if 'error' in self.results[test.name]:
                    status = "Failed"
                    status_class = "negative"
                    overall_impact = "N/A"
                else:
                    status = "Completed"
                    status_class = "positive"
                    overall_impact = test.comparison.get('overall_impact', 'unknown')
                    overall_impact_class = overall_impact
                
                test_rows += f"""
                    <tr>
                        <td>{test.name}</td>
                        <td>{test.component}</td>
                        <td class="{overall_impact_class}">{overall_impact}</td>
                        <td>{test.expected_impact}</td>
                        <td class="{status_class}">{status}</td>
                    </tr>
                """
        
        # Format HTML
        html = html.format(
            total_tests=self.summary['total_tests'],
            completed_tests=self.summary['completed_tests'],
            failed_tests=self.summary['failed_tests'],
            component_rows=component_rows,
            key_findings=key_findings,
            test_rows=test_rows
        )
        
        # Save HTML report
        with open(report_path, 'w') as f:
            f.write(html)
            
    def get_critical_components(self) -> List[str]:
        """
        Get list of critical components based on ablation test results.
        
        Returns:
            List of critical components
        """
        critical_components = []
        
        for component, impact in self.summary['component_impacts'].items():
            if impact['overall_assessment'] == 'critical':
                critical_components.append(component)
        
        return critical_components
        
    def get_beneficial_components(self) -> List[str]:
        """
        Get list of beneficial components based on ablation test results.
        
        Returns:
            List of beneficial components
        """
        beneficial_components = []
        
        for component, impact in self.summary['component_impacts'].items():
            if impact['overall_assessment'] == 'beneficial':
                beneficial_components.append(component)
        
        return beneficial_components
        
    def get_test_results(self, test_name: str) -> Optional[Dict[str, Any]]:
        """
        Get results for a specific test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Test results or None if not found
        """
        return self.results.get(test_name)
        
    def get_component_results(self, component: str) -> Dict[str, Any]:
        """
        Get results for all tests of a specific component.
        
        Args:
            component: Component name
            
        Returns:
            Component test results
        """
        component_results = {}
        
        for test in self.tests:
            if test.component == component and test.name in self.results:
                component_results[test.name] = self.results[test.name]
        
        return component_results


def create_ablation_test_suite(config: Dict[str, Any]) -> AblationTestingFramework:
    """
    Create a comprehensive ablation test suite.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Ablation testing framework with comprehensive test suite
    """
    # Initialize framework
    framework = AblationTestingFramework(config)
    
    # Get baseline configuration
    baseline_config = config.copy()
    
    # Add reward function ablation tests
    framework.add_reward_ablation_tests(baseline_config)
    
    # Add feature engineering ablation tests
    framework.add_feature_ablation_tests(baseline_config)
    
    # Add model architecture ablation tests
    framework.add_model_ablation_tests(baseline_config)
    
    # Add training process ablation tests
    framework.add_training_ablation_tests(baseline_config)
    
    return framework