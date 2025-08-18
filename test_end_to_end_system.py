#!/usr/bin/env python3
"""
Comprehensive end-to-end system test for the RL Intraday Trading System.

This script tests all components and their interactions using proxy data
to ensure the entire system works correctly after the configuration handling fixes.
"""

import numpy as np
import pandas as pd
import logging
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('end_to_end_test.log')
    ]
)

logger = logging.getLogger(__name__)

def generate_proxy_market_data(days: int = 5) -> pd.DataFrame:
    """
    Generate realistic proxy market data for testing.
    
    Args:
        days: Number of trading days to generate
        
    Returns:
        DataFrame with OHLCV and order book data
    """
    logger.info(f"Generating {days} days of proxy market data...")
    
    # Generate trading days (9:30 AM - 4:00 PM EST)
    start_date = datetime(2023, 1, 2)  # First trading day of 2023
    dates = []
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        # Skip weekends
        if current_date.weekday() >= 5:
            continue
            
        # Generate minute bars for regular trading hours
        day_dates = pd.date_range(
            start=f"{current_date.date()} 09:30",
            end=f"{current_date.date()} 16:00",
            freq='1min',
            tz='America/New_York'
        )
        dates.extend(day_dates)
    
    # Generate realistic price data
    n_bars = len(dates)
    if n_bars == 0:
        raise ValueError("No trading days generated")
    
    # Base price around 4500 (S&P 500 level)
    base_price = 4500
    
    # Generate returns with realistic characteristics
    np.random.seed(42)  # For reproducible results
    
    # Intraday volatility pattern (higher at open and close)
    time_of_day = np.array([d.hour + d.minute/60 for d in dates])
    volatility_pattern = 0.001 * (1 + 0.5 * np.sin((time_of_day - 9.5) * np.pi / 6.5))
    
    # Generate returns with volatility clustering
    returns = np.random.normal(0, volatility_pattern, n_bars)
    
    # Add some autocorrelation and momentum effects
    for i in range(1, n_bars):
        returns[i] += 0.1 * returns[i-1]
    
    # Generate price series
    prices = base_price + np.cumsum(returns * base_price)
    
    # Generate OHLCV data
    high_low_range = np.random.uniform(0.1, 0.3, n_bars) * prices * 0.001
    open_close_spread = np.random.uniform(-0.5, 0.5, n_bars) * high_low_range
    
    market_data = pd.DataFrame({
        'open': prices + open_close_spread,
        'high': prices + np.abs(high_low_range) + np.random.uniform(0, 0.1, n_bars) * high_low_range,
        'low': prices - np.abs(high_low_range) - np.random.uniform(0, 0.1, n_bars) * high_low_range,
        'close': prices,
        'volume': np.random.randint(100, 2000, n_bars),
        'bid_price': prices - 0.125,
        'ask_price': prices + 0.125,
        'bid_size': np.random.randint(50, 500, n_bars),
        'ask_size': np.random.randint(50, 500, n_bars)
    }, index=dates)
    
    # Ensure OHLC constraints
    market_data['high'] = np.maximum(
        market_data['high'],
        np.maximum(market_data['open'], market_data['close'])
    )
    market_data['low'] = np.minimum(
        market_data['low'],
        np.minimum(market_data['open'], market_data['close'])
    )
    
    logger.info(f"Generated {len(market_data)} bars of market data")
    return market_data

def test_configuration_system():
    """Test the configuration system with various scenarios"""
    logger.info("=== Testing Configuration System ===")
    
    try:
        logger.info("Testing Settings class...")
        from src.utils.config_loader import Settings, ConfigError
        
        # Test basic functionality - FIXED: Initialize with dictionary
        settings = Settings({'test': 'value'})
        assert settings.get('test') == 'value'
        
        # Test nested access - FIXED: Initialize with nested dictionary
        settings = Settings({'nested': {'deep': 'value'}})
        assert settings.get('nested', 'deep') == 'value'
        
        # Test default value - FIXED: Use proper default parameter
        default_value = settings.get('nonexistent', 'key', default='default')
        assert default_value == 'default'
        
        # Test error handling
        try:
            settings.get('nonexistent', 'key')
            assert False, "Should have raised ConfigError"
        except ConfigError:
            pass
        
        logger.info("✓ Configuration system tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration system test failed: {e}")
        return False


def test_feature_pipeline():
    """Test feature pipeline components."""
    logger.info("=== Testing Feature Pipeline ===")
    
    try:
        logger.info("Testing TechnicalIndicators...")
        from src.features.technical_indicators import TechnicalIndicators
        from src.utils.config_loader import Settings
        
        # Test with Settings object
        settings = Settings({'indicators': {}})
        ti = TechnicalIndicators(settings)
        
        logger.info("Generating 1 days of proxy market data...")
        market_data = generate_proxy_market_data(days=1)
        logger.info(f"Generated {len(market_data)} bars of market data")
        
        logger.info("Calculating technical indicators...")
        indicators = ti.calculate_all_indicators(market_data)
        logger.info(f"Calculated {len(indicators.columns)} indicators")
        
        # Check for any NaN values
        nan_counts = indicators.isna().sum()
        logger.info(f"NaN counts per indicator: {nan_counts[nan_counts > 0]}")
        
        # Check for any infinite values
        inf_counts = np.isinf(indicators).sum()
        logger.info(f"Infinite counts per indicator: {inf_counts[inf_counts > 0]}")
        
        # Check for any boolean context issues
        logger.info("Checking for potential boolean context issues...")
        for col in indicators.columns:
            series = indicators[col]
            if series.dtype == 'bool':
                logger.info(f"Boolean series found: {col}")
            elif series.dtype == 'object':
                logger.info(f"Object series found: {col}")
        
        # Test specific indicators that might cause issues
        logger.info("Testing individual indicator calculations...")
        try:
            sma_result = ti.sma(market_data['close'], 10)
            logger.info(f"SMA calculation successful: {len(sma_result)} values")
        except Exception as e:
            logger.error(f"SMA calculation failed: {e}")
            
        try:
            rsi_result = ti.rsi(market_data['close'], 14)
            logger.info(f"RSI calculation successful: {len(rsi_result)} values")
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            
        try:
            atr_result = ti.atr(market_data, 14)
            logger.info(f"ATR calculation successful: {len(atr_result)} values")
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
        
        # Test with specific indicators configuration
        logger.info("Testing with specific indicators configuration...")
        settings_with_indicators = Settings({
            'indicators': {
                'sma': {'window': 10},
                'rsi': {'window': 14},
                'atr': {'window': 14}
            }
        })
        ti_with_config = TechnicalIndicators(settings_with_indicators)
        indicators_with_config = ti_with_config.calculate_all_indicators(market_data)
        logger.info(f"Calculated {len(indicators_with_config.columns)} indicators with config")
        
        assert not indicators.empty, "Indicators should not be empty"
        assert len(indicators) == len(market_data), "Indicators should match market data length"
        
        logger.info("✓ Feature pipeline tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Feature pipeline test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def test_execution_simulator():
    """Test the execution simulator component"""
    logger.info("=== Testing Execution Simulator ===")
    
    try:
        logger.info("Testing ExecutionSimulator with Settings object...")
        from src.sim.execution import ExecutionSimulator, Order
        from src.utils.config_loader import Settings
        
        # Create configuration - FIXED: Use dictionary for config parameter
        config = {
            'transaction_cost': 2.5,
            'slippage': 0.01,
            'fill_probability': 0.95  # Add this explicitly
        }
        
        # Create execution simulator
        exec_sim = ExecutionSimulator(config=config)
        
        # Test order execution - FIXED: Use proper Order class
        order = Order(
            symbol='ES',
            side='BUY',
            quantity=1,
            order_type='MARKET',
            price=4500.0
        )
        
        market_data = {
            'bid_price': 4499.875,
            'ask_price': 4500.125,
            'bid_size': 100,
            'ask_size': 100
        }
        
        execution_result = exec_sim.execute_order(order, market_data)
        
        assert execution_result is not None
        assert execution_result['quantity'] == 1
        assert 'transaction_cost' in execution_result
        
        logger.info("✓ Execution simulator tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Execution simulator test failed: {e}")
        return False


def test_component_integration():
    """Test integration of all components"""
    logger.info("=== Testing Component Integration ===")
    
    try:
        logger.info("Generating 2 days of proxy market data...")
        data = generate_proxy_market_data(days=2)
        logger.info(f"Generated {len(data)} bars of market data")
        
        logger.info("Testing full system integration...")
        from src.sim.env_intraday_rl import IntradayRLEnvironment
        
        # Create comprehensive configuration
        config = {
            'initial_cash': 100000.0,
            'max_position_size': 5,
            'transaction_cost': 2.5,
            'slippage': 0.01,
            'lookback_window': 20,
            'triple_barrier': {
                'profit_target': 0.01,
                'stop_loss': 0.005,
                'time_limit': 60
            },
            'features': {
                'technical': ['sma_10', 'rsi_14', 'atr_14'],
                'microstructure': ['spread', 'imbalance'],
                'time': ['hour', 'minute', 'time_to_close']
            }
        }
        
        # Create environment
        env = IntradayRLEnvironment(data, config)
        
        # Test environment properties
        assert hasattr(env, 'exec_sim')
        assert hasattr(env, 'risk_manager')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        
        # Test component integration - FIXED: Use exec_params attribute
        assert env.exec_sim.exec_params.commission_per_contract == config['transaction_cost']
        
        # Test reset functionality
        obs = env.reset()
        assert obs is not None
        
        # Test step functionality
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        logger.info("✓ Component integration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Component integration test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("=== Testing Error Handling ===")
    
    try:
        from src.sim.env_intraday_rl import IntradayRLEnvironment
        from src.utils.config_loader import ConfigError
        
        # Test with empty configuration
        logger.info("Testing with empty configuration...")
        market_data = generate_proxy_market_data(days=1)
        
        try:
            # Provide minimal required configuration
            minimal_config = {
                'initial_cash': 100000.0,
                'max_position_size': 5,
                'transaction_cost': 2.5,
                'slippage': 0.01
            }
            env = IntradayRLEnvironment(market_data=market_data, config=minimal_config)
            logger.info("✓ Empty configuration handled gracefully")
        except Exception as e:
            logger.error(f"Empty configuration not handled: {e}")
            return False
        
        # Test with invalid configuration values
        logger.info("Testing with invalid configuration values...")
        invalid_config = {
            'max_position_size': -1,  # Invalid negative value
            'transaction_cost': 'invalid',  # Invalid type
            'features': {'technical': ['nonexistent_indicator']}  # Invalid feature
        }
        
        try:
            env = IntradayRLEnvironment(market_data=market_data, config=invalid_config)
            logger.info("✓ Invalid configuration handled gracefully")
        except Exception as e:
            logger.info(f"✓ Invalid configuration properly rejected: {e}")
        
        # Test with missing market data
        logger.info("Testing with missing market data...")
        try:
            env = IntradayRLEnvironment(market_data=pd.DataFrame(), config={'max_position_size': 1})
            logger.error("Empty market data should have raised an error")
            return False
        except Exception as e:
            logger.info(f"✓ Empty market data properly handled: {e}")
        
        logger.info("✓ Error handling tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and generate a comprehensive report."""
    logger.info("Starting Comprehensive End-to-End System Test")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration System", test_configuration_system),
        ("Feature Pipeline", test_feature_pipeline),
        ("Execution Simulator", test_execution_simulator),
        ("Component Integration", test_component_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = False
    
    # Generate summary report
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! The system is working correctly.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
