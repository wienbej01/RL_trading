#!/usr/bin/env python3
"""
Test script to verify composite reward functionality with both SPY and AAPL data.
"""

import pandas as pd
import numpy as np
from src.rl.composite_reward import CompositeReward
from src.data.data_loader import UnifiedDataLoader
from src.utils.config_loader import Settings

def test_composite_reward():
    """Test composite reward with sample data."""
    print("Testing CompositeReward class...")
    
    # Create composite reward with default parameters
    reward_fn = CompositeReward(
        w_ret=1.0,
        w_turnover=0.2,
        w_inv=0.05,
        w_dsr=0.0,
        include_costs=True
    )
    
    # Test with sample data
    print("Testing with sample PnL data...")
    positions = [0, 1, 1, 0, -1, -1, 0]
    pnls = [0.0, 10.0, 5.0, -2.0, -8.0, 3.0, 1.0]
    tx_costs = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    
    rewards = []
    for i in range(len(positions)):
        reward, breakdown = reward_fn.step(pnls[i], positions[i], tx_costs[i])
        rewards.append(reward)
        print(f"Step {i}: pos={positions[i]}, pnl={pnls[i]}, cost={tx_costs[i]}, reward={reward:.4f}")
    
    stats = reward_fn.get_stats()
    print(f"Reward stats: {stats}")
    
    return True

def test_data_availability():
    """Test data availability for SPY and AAPL."""
    print("\nTesting data availability...")
    
    # Load settings
    settings = Settings.from_paths('configs/settings.yaml')
    loader = UnifiedDataLoader(settings=settings)
    
    # Test date range that should have data for both tickers
    start_date = pd.Timestamp('2024-01-02')
    end_date = pd.Timestamp('2024-01-31')
    
    print(f"Testing data loading for period: {start_date} to {end_date}")
    
    tickers = ['SPY', 'AAPL']
    data_status = {}
    
    for ticker in tickers:
        try:
            print(f"Loading data for {ticker}...")
            data = loader.load_ohlcv(ticker, start_date, end_date)
            data_status[ticker] = {
                'available': not data.empty,
                'rows': len(data) if not data.empty else 0,
                'date_range': f"{data.index.min()} to {data.index.max()}" if not data.empty else "N/A"
            }
            print(f"  {ticker}: {'Available' if data_status[ticker]['available'] else 'Not available'}")
            if data_status[ticker]['available']:
                print(f"    Rows: {data_status[ticker]['rows']}")
                print(f"    Date range: {data_status[ticker]['date_range']}")
        except Exception as e:
            print(f"  {ticker}: Error loading data - {e}")
            data_status[ticker] = {'available': False, 'error': str(e)}
    
    return data_status

def main():
    """Main test function."""
    print("=" * 50)
    print("Composite Reward Test Suite")
    print("=" * 50)
    
    # Test composite reward class
    try:
        test_composite_reward()
        print("✓ CompositeReward test passed")
    except Exception as e:
        print(f"✗ CompositeReward test failed: {e}")
        return False
    
    # Test data availability
    try:
        data_status = test_data_availability()
        print("✓ Data availability test completed")
        
        # Check if both tickers have data
        both_available = all(status.get('available', False) for status in data_status.values())
        if both_available:
            print("✓ Both SPY and AAPL data available")
        else:
            print("⚠ Some tickers missing data:")
            for ticker, status in data_status.items():
                if not status.get('available', False):
                    print(f"  {ticker}: {status.get('error', 'No data')}")
    except Exception as e:
        print(f"✗ Data availability test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Test suite completed")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    main()