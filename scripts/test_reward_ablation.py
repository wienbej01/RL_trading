#!/usr/bin/env python3
"""
Test script to run reward ablation with both SPY and AAPL using composite reward.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_walk_forward_test():
    """Run walk-forward test with composite reward and real-time progress."""
    print("Running walk-forward test with composite reward...")
    print("=" * 60)
    
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
    print("=" * 60)
    print("Starting process... (this may take several minutes)")
    print("Progress: ", end="", flush=True)
    
    try:
        # Start the process with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=".",
            env=dict(os.environ, PYTHONPATH=os.getcwd())
        )
        
        # Read output in real-time
        progress_chars = ['|', '/', '-', '\\']
        progress_idx = 0
        line_count = 0
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line_count += 1
                # Show progress indicator every 10 lines
                if line_count % 10 == 0:
                    print(f"\rProgress: {progress_chars[progress_idx % 4]} ({line_count} lines processed)", end="", flush=True)
                    progress_idx += 1
                # Print important lines immediately
                if any(keyword in output.lower() for keyword in ['error', 'warn', 'fail', 'window', 'ticker']):
                    print(f"\n[LOG] {output.strip()}")
                    print("Progress: ", end="", flush=True)
        
        # Wait for process to complete
        rc = process.poll()
        print(f"\rProgress: ✓ Completed ({line_count} lines processed)")
        
        if rc == 0:
            print("✓ Walk-forward test completed successfully")
            return True
        else:
            print(f"✗ Walk-forward test failed with return code: {rc}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠ Test interrupted by user")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"\n✗ Error running command: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Reward Ablation Test with Real-time Progress")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run walk-forward test
    success = run_walk_forward_test()
    
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("✓ Overall test completed successfully")
    else:
        print("✗ Overall test failed")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)