#!/usr/bin/env python3
"""
Simple test runner for RL trading system without pytest dependency.
This script tests the core functionality of the system components.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_loader():
    """Test configuration loader functionality."""
    print("Testing config loader...")
    try:
        from src.utils.config_loader import Settings, load_config, ConfigError
        from src.utils.config_loader import load_yaml, get_project_root, get_config_path
        
        # Test basic functionality
        config_path = get_config_path()
        settings = Settings.from_paths(config_path)
        
        # Test configuration access
        instrument = settings.get('data', 'instrument')
        print(f"  ✓ Config loaded successfully. Instrument: {instrument}")
        
        # Test validation
        settings.validate()
        print("  ✓ Configuration validation passed")
        
        return True
    except Exception as e:
        print(f"  ✗ Config loader test failed: {e}")
        return False

def test_logging():
    """Test logging functionality."""
    print("Testing logging...")
    try:
        from src.utils.logging import get_logger, setup_logging
        
        # Test logger setup
        logger = get_logger(__name__)
        logger.info("Test log message")
        print("  ✓ Logging setup successful")
        
        return True
    except Exception as e:
        print(f"  ✗ Logging test failed: {e}")
        return False

def test_metrics():
    """Test metrics calculation."""
    print("Testing metrics...")
    try:
        from src.utils.metrics import calculate_risk_metrics, PerformanceMetrics
        import pandas as pd
        import numpy as np
        
        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Test risk metrics calculation
        metrics = calculate_risk_metrics(returns)
        print(f"  ✓ Risk metrics calculated: Sharpe ratio = {metrics['sharpe_ratio']:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Metrics test failed: {e}")
        return False

def test_technical_indicators():
    """Test technical indicators."""
    print("Testing technical indicators...")
    try:
        from src.features.technical_indicators import TechnicalIndicators, calculate_returns
        import pandas as pd
        import numpy as np
        
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
        
        data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test technical indicators
        from src.utils.config_loader import Settings
        settings = Settings.from_paths('configs/settings.yaml')
        indicators = TechnicalIndicators(settings)
        
        result = indicators.calculate_all_indicators(data)
        print(f"  ✓ Technical indicators calculated: {len(result.columns)} indicators")
        
        # Test returns calculation
        returns = calculate_returns(data['close'])
        print(f"  ✓ Returns calculated: {len(returns)} data points")
        
        return True
    except Exception as e:
        print(f"  ✗ Technical indicators test failed: {e}")
        return False

def test_simulation():
    """Test simulation environment."""
    print("Testing simulation...")
    try:
        from src.sim.env_intraday_rl import IntradayRLEnvironment
        from src.utils.config_loader import Settings
        
        # Create sample data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2023-01-02', periods=20, freq='D')
        prices = np.cumprod(1 + np.random.normal(0.001, 0.02, 20))
        
        data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 20)
        }, index=dates)
        
        # Test environment creation
        settings = Settings.from_paths('configs/settings.yaml')
        env = IntradayRLEnvironment(market_data=data, config=settings)
        
        # Test reset and step
        obs = env.reset()
        print(f"  ✓ Environment reset: observation shape = {obs.shape}")
        
        action = 0  # Hold
        obs, reward, done, info = env.step(action)
        print(f"  ✓ Environment step: reward = {reward:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Simulation test failed: {e}")
        return False

def test_trading():
    """Test trading components."""
    print("Testing trading...")
    try:
        from src.trading.paper_trading import PaperTradingEngine
        from src.utils.config_loader import Settings
        
        # Test paper trading engine
        config = {
            'initial_capital': 100000.0,
            'commission_per_trade': 2.5,
            'max_position_size': 5
        }
        
        engine = PaperTradingEngine(config=config)
        print("  ✓ Paper trading engine created")
        
        return True
    except Exception as e:
        print(f"  ✗ Trading test failed: {e}")
        return False

def test_data_modules():
    """Test data modules."""
    print("Testing data modules...")
    try:
        from src.data.vix_loader import VIXLoader
        from src.data.econ_calendar import EconomicCalendar
        
        # Test VIX loader
        vix_loader = VIXLoader()
        print("  ✓ VIX loader created")
        
        # Test economic calendar
        calendar = EconomicCalendar()
        print("  ✓ Economic calendar created")
        
        return True
    except Exception as e:
        print(f"  ✗ Data modules test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("RL Trading System - Component Test Suite")
    print("=" * 60)
    
    tests = [
        test_config_loader,
        test_logging,
        test_metrics,
        test_technical_indicators,
        test_simulation,
        test_trading,
        test_data_modules
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())