#!/usr/bin/env python3
"""
Test script to run reward ablation with both SPY and AAPL using composite reward.
"""

import subprocess
import sys
from pathlib import Path

def run_walk_forward_test():
    """Run walk-forward test with composite reward."""
    print("Running walk-forward test with composite reward...")
    
    # Command to run walk-forward with both SPY and AAPL for January 2024
    cmd = [
        "python", "scripts/run_wf.py",
        "--config", "configs/settings.yaml",
        "--run-name", "test_composite_reward",
        "--tickers", "SPY", "AAPL",
        "--train-start", "2024-01-02",
        "--train-end", "2024-01-31",
        "--wf-train-days", "10",
        "--wf-valid-days", "3",
        "--wf-test-days", "3",
        "--wf-step-days", "3",
        "--embargo-min", "15",
        "--feature-pack", "curated",
        "--timesteps", "1000",
        "--seed", "123",
        "--strict-test-window",
        "--reward-mix", "ret=1.0,turnover=0.2,inventory=0.05,dsr=0.0"
    ]
    
    print("Command:", " ".join(cmd))
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        print("Return code:", result.returncode)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 50)
    print("Reward Ablation Test")
    print("=" * 50)
    
    # Run walk-forward test
    success = run_walk_forward_test()
    
    if success:
        print("✓ Walk-forward test completed successfully")
    else:
        print("✗ Walk-forward test failed")
    
    print("\n" + "=" * 50)
    print("Test completed")
    print("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)