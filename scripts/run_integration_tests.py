#!/usr/bin/env python3
"""
Integration test runner for the Multi-Ticker RL Trading System.
"""

import sys
import os
import time
import logging
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.utils.logging import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_test_results.log'),
        logging.StreamHandler()
    ]
)

logger = get_logger(__name__)

def run_test(test_func, test_name):
    """Run a single test and return results."""
    logger.info(f"Running test: {test_name}")
    
    try:
        start_time = time.time()
        result = test_func()
        end_time = time.time()
        
        test_time = end_time - start_time
        
        if result is True:
            logger.info(f"✓ Test passed: {test_name} (Time: {test_time:.2f}s)")
            return {'name': test_name, 'status': 'passed', 'time': test_time}
        else:
            logger.info(f"✓ Test passed: {test_name} (Time: {test_time:.2f}s)")
            return {'name': test_name, 'status': 'passed', 'time': test_time, 'result': result}
    
    except Exception as e:
        logger.error(f"✗ Test failed: {test_name} - {str(e)}")
        return {'name': test_name, 'status': 'failed', 'error': str(e)}

def test_basic_multiticker_training():
    """Test basic multi-ticker training functionality"""
    # Load configuration
    config = load_config()
    
    # Initialize data loader
    from src.data.multiticker_data_loader import MultiTickerDataLoader
    data_loader = MultiTickerDataLoader(config['data'])
    data = data_loader.load_data()
    
    # Initialize feature pipeline
    from src.features.multiticker_pipeline import MultiTickerFeaturePipeline
    feature_pipeline = MultiTickerFeaturePipeline(config['features'])
    features = feature_pipeline.fit_transform(data)
    
    # Initialize trainer
    from src.rl.multiticker_trainer import MultiTickerRLTrainer
    trainer = MultiTickerRLTrainer(config)
    
    # Train model (reduced timesteps for testing)
    config['training']['total_timesteps'] = 1000
    model = trainer.train()
    
    # Verify model training completed successfully
    assert model is not None
    assert hasattr(model, 'policy')
    
    return True

def test_walkforward_optimization():
    """Test walk-forward optimization with LOT-O CV"""
    # Load configuration
    config = load_config()
    
    # Configure WFO
    config['walkforward'] = {
        'enabled': True,
        'n_folds': 3,
        'embargo_days': 5,
        'test_size': 0.2,
        'regime_aware': True
    }
    
    # Initialize trainer
    from src.rl.multiticker_trainer import MultiTickerRLTrainer
    trainer = MultiTickerRLTrainer(config)
    
    # Run WFO
    wfo_results = trainer.walk_forward_training()
    
    # Verify WFO results
    assert 'avg_sharpe_ratio' in wfo_results
    assert 'fold_results' in wfo_results
    assert len(wfo_results['fold_results']) == 3
    
    return True

def test_hyperparameter_optimization():
    """Test hyperparameter optimization with Optuna"""
    # Load configuration
    config = load_config()
    
    # Configure HPO
    config['hpo'] = {
        'enabled': True,
        'n_trials': 5,  # Reduced for testing
        'direction': 'maximize',
        'metric': 'sharpe_ratio',
        'pruner': 'median',
        'sampler': 'tpe',
        'search_space': {
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [32, 64]},
            'gamma': {'type': 'float', 'low': 0.9, 'high': 0.999}
        }
    }
    
    # Initialize trainer
    from src.rl.multiticker_trainer import MultiTickerRLTrainer
    trainer = MultiTickerRLTrainer(config)
    
    # Run HPO
    hpo_results = trainer.optimize_hyperparameters()
    
    # Verify HPO results
    assert 'best_params' in hpo_results
    assert 'best_value' in hpo_results
    assert 'history' in hpo_results
    assert len(hpo_results['history']) == 5
    
    return True

def test_dynamic_universe_selection():
    """Test dynamic universe selection functionality"""
    # Load configuration
    config = load_config()
    
    # Configure dynamic universe
    config['multiticker'] = {
        'universe': {
            'selection_method': 'dynamic',
            'max_tickers': 5,
            'min_tickers': 3,
            'rebalance_freq': '1M',
            'selection_metrics': ['liquidity', 'volatility', 'trend_strength']
        }
    }
    
    # Initialize data loader
    from src.data.multiticker_data_loader import MultiTickerDataLoader
    data_loader = MultiTickerDataLoader(config['data'])
    data = data_loader.load_data()
    
    # Initialize universe selector
    from src.data.multiticker_data_loader import DynamicUniverseSelector
    universe_selector = DynamicUniverseSelector(config['multiticker']['universe'])
    
    # Test universe selection
    universe = universe_selector.select_universe(data)
    
    # Verify universe selection
    assert isinstance(universe, list)
    assert len(universe) >= config['multiticker']['universe']['min_tickers']
    assert len(universe) <= config['multiticker']['universe']['max_tickers']
    
    return True

def test_custom_reward_function():
    """Test custom reward function implementation"""
    # Load configuration
    config = load_config()
    
    # Define custom reward calculator
    from src.sim.multiticker_env import MultiTickerRewardCalculator
    
    class CustomRewardCalculator(MultiTickerRewardCalculator):
        def calculate_reward(self, portfolio_state, action, next_portfolio_state):
            # Calculate base reward
            base_reward = super().calculate_reward(portfolio_state, action, next_portfolio_state)
            
            # Add custom reward for diversification
            positions = portfolio_state['positions']
            if len(positions) > 1:
                total_value = sum(abs(pos) for pos in positions.values())
                if total_value > 0:
                    hhi = sum((abs(pos) / total_value) ** 2 for pos in positions.values())
                    diversification_reward = (1 - hhi) * 0.1
                    base_reward += diversification_reward
            
            return base_reward
    
    # Configure custom reward
    config['environment']['reward_calculator'] = CustomRewardCalculator
    
    # Initialize environment
    from src.sim.multiticker_env import MultiTickerIntradayRLEnv
    env = MultiTickerIntradayRLEnv(config)
    
    # Test custom reward calculation
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    
    # Verify reward calculation
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    
    return True

def test_training_performance():
    """Test training performance metrics"""
    import time
    import psutil
    import torch
    
    # Load configuration
    config = load_config()
    
    # Initialize trainer
    from src.rl.multiticker_trainer import MultiTickerRLTrainer
    trainer = MultiTickerRLTrainer(config)
    
    # Monitor performance
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Train model
    config['training']['total_timesteps'] = 5000
    model = trainer.train()
    
    # Calculate performance metrics
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    training_time = end_time - start_time
    memory_usage = end_memory - start_memory
    
    # Verify performance requirements
    assert training_time < 300  # Training should complete within 5 minutes
    assert memory_usage < 1000  # Memory usage should be less than 1GB
    
    return {
        'training_time': training_time,
        'memory_usage': memory_usage,
        'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    }

def test_inference_performance():
    """Test inference performance metrics"""
    import time
    
    # Load configuration
    config = load_config()
    
    # Initialize environment
    from src.sim.multiticker_env import MultiTickerIntradayRLEnv
    env = MultiTickerIntradayRLEnv(config)
    
    # Measure inference time
    obs = env.reset()
    inference_times = []
    
    for _ in range(100):  # Test 100 inferences
        start_time = time.time()
        action = env.action_space.sample()  # Sample action for testing
        end_time = time.time()
        
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    # Calculate performance metrics
    avg_inference_time = sum(inference_times) / len(inference_times)
    max_inference_time = max(inference_times)
    
    # Verify performance requirements
    assert avg_inference_time < 0.1  # Average inference time should be less than 100ms
    assert max_inference_time < 0.5  # Maximum inference time should be less than 500ms
    
    return {
        'avg_inference_time': avg_inference_time,
        'max_inference_time': max_inference_time
    }

def test_data_loading_performance():
    """Test data loading performance metrics"""
    import time
    
    # Load configuration
    config = load_config()
    
    # Initialize data loader
    from src.data.multiticker_data_loader import MultiTickerDataLoader
    data_loader = MultiTickerDataLoader(config['data'])
    
    # Measure data loading time
    start_time = time.time()
    data = data_loader.load_data()
    end_time = time.time()
    
    data_loading_time = end_time - start_time
    
    # Verify performance requirements
    assert data_loading_time < 60  # Data loading should complete within 1 minute
    
    # Measure feature extraction time
    from src.features.multiticker_pipeline import MultiTickerFeaturePipeline
    feature_pipeline = MultiTickerFeaturePipeline(config['features'])
    start_time = time.time()
    features = feature_pipeline.fit_transform(data)
    end_time = time.time()
    
    feature_extraction_time = end_time - start_time
    
    # Verify performance requirements
    assert feature_extraction_time < 120  # Feature extraction should complete within 2 minutes
    
    return {
        'data_loading_time': data_loading_time,
        'feature_extraction_time': feature_extraction_time
    }

def main():
    """Run all integration tests."""
    logger.info("Starting integration tests for Multi-Ticker RL Trading System")
    
    # Test configuration
    config = load_config()
    
    # Test results
    test_results = []
    
    # Integration tests
    integration_tests = [
        (test_basic_multiticker_training, "Basic Multi-Ticker Training"),
        (test_walkforward_optimization, "Walk-Forward Optimization"),
        (test_hyperparameter_optimization, "Hyperparameter Optimization"),
        (test_dynamic_universe_selection, "Dynamic Universe Selection"),
        (test_custom_reward_function, "Custom Reward Function")
    ]
    
    # Performance tests
    performance_tests = [
        (test_training_performance, "Training Performance"),
        (test_inference_performance, "Inference Performance"),
        (test_data_loading_performance, "Data Loading Performance")
    ]
    
    # Run integration tests
    logger.info("Running integration tests...")
    for test_func, test_name in integration_tests:
        result = run_test(test_func, test_name)
        test_results.append(result)
    
    # Run performance tests
    logger.info("Running performance tests...")
    for test_func, test_name in performance_tests:
        result = run_test(test_func, test_name)
        test_results.append(result)
    
    # Calculate summary statistics
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result['status'] == 'passed')
    failed_tests = sum(1 for result in test_results if result['status'] == 'failed')
    
    total_time = sum(result.get('time', 0) for result in test_results)
    
    # Log summary
    logger.info(f"\nIntegration Test Summary:")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Total time: {total_time:.2f}s")
    
    # Save results to file
    with open('integration_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Exit with appropriate code
    if failed_tests > 0:
        logger.error(f"{failed_tests} test(s) failed. Check logs for details.")
        sys.exit(1)
    else:
        logger.info("All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()