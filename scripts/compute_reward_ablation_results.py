#!/usr/bin/env python3
"""
Script to compute reward ablation results for different reward mixes.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

def load_backtest_results(results_dir):
    """Load backtest results from directory."""
    try:
        # Look for summary.json or backtest_results.json
        summary_path = Path(results_dir) / "backtest" / "summary.json"
        if not summary_path.exists():
            summary_path = Path(results_dir) / "summary.json"
            
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            return data
        else:
            print(f"No summary file found in {results_dir}")
            return None
    except Exception as e:
        print(f"Error loading results from {results_dir}: {e}")
        return None

def extract_metrics(results_data):
    """Extract key metrics from backtest results."""
    if not results_data:
        return {}
    
    metrics = {}
    
    # Try to get metrics from different possible locations in the JSON
    portfolio_metrics = results_data.get('portfolio_metrics', {})
    if not portfolio_metrics:
        portfolio_metrics = results_data.get('metrics', {})
    
    # Extract key metrics
    metrics['pf'] = portfolio_metrics.get('profit_factor', 0)
    metrics['sharpe'] = portfolio_metrics.get('sharpe_ratio', 0)
    metrics['flips'] = portfolio_metrics.get('flips', 0)
    metrics['turnover'] = portfolio_metrics.get('turnover', 0)
    metrics['long_steps'] = portfolio_metrics.get('long_steps', 0)
    metrics['short_steps'] = portfolio_metrics.get('short_steps', 0)
    
    return metrics

def run_reward_mix(config_path, tickers, train_start, train_end, test_start, test_end, 
                   feature_pack, feature_screen_run, max_steps, seed, reward_mix, output_dir):
    """Run a single reward mix configuration."""
    import subprocess
    import os
    
    cmd = [
        "python", "scripts/run_multiticker_pipeline.py",
        "--config", config_path,
        "--tickers", tickers,
        "--train-start", train_start,
        "--train-end", train_end,
        "--test-start", test_start,
        "--test-end", test_end,
        "--feature-pack", feature_pack,
        "--feature-screen-run", feature_screen_run,
        "--max-steps", str(max_steps),
        "--seed", str(seed),
        "--reward-mix", reward_mix,
        "--strict-test-window",
        "--output-dir", output_dir,
        "--skip-training"  # Skip training if model already exists
    ]
    
    # Try to run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"Stderr: {result.stderr}")
            return None
        return output_dir
    except Exception as e:
        print(f"Error running command: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compute reward ablation results")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--tickers", default="SPY")
    parser.add_argument("--train-start", default="2024-05-01")
    parser.add_argument("--train-end", default="2024-06-30")
    parser.add_argument("--test-start", default="2024-06-15")
    parser.add_argument("--test-end", default="2024-06-30")
    parser.add_argument("--feature-pack", default="curated")
    parser.add_argument("--feature-screen-run", default="fs_spy_aapl_may_jun")
    parser.add_argument("--max-steps", type=int, default=7500)
    parser.add_argument("--seed", type=int, default=123)
    
    args = parser.parse_args()
    
    # Define reward mixes to test
    reward_mixes = {
        "M0 (ret=1.0,t=0,i=0,dsr=0)": "ret=1.0,turnover=0.0,inventory=0.0,dsr=0.0",
        "M1 (ret=1.0,t=0.2,i=0.05,dsr=0)": "ret=1.0,turnover=0.2,inventory=0.05,dsr=0.0",
        "M2 (ret=1.0,t=0.4,i=0.10,dsr=0)": "ret=1.0,turnover=0.4,inventory=0.10,dsr=0.0",
        "M3 (ret=1.0,t=0.2,i=0.05,dsr=0.05)": "ret=1.0,turnover=0.2,inventory=0.05,dsr=0.05"
    }
    
    # Store results
    results = []
    
    # Run each reward mix
    for name, mix in reward_mixes.items():
        print(f"Running {name}...")
        
        # Create output directory for this mix
        output_dir = f"results/dev/reward_ablation/{name.replace(' ', '_').replace('(', '').replace(')', '')}"
        
        # Run the pipeline
        result_dir = run_reward_mix(
            args.config, args.tickers, args.train_start, args.train_end,
            args.test_start, args.test_end, args.feature_pack, args.feature_screen_run,
            args.max_steps, args.seed, mix, output_dir
        )
        
        if result_dir:
            # Load results
            results_data = load_backtest_results(result_dir)
            metrics = extract_metrics(results_data)
            
            # Add to results
            row = {
                "Mix": name,
                "Test PF": metrics.get('pf', 0),
                "Test Sharpe": metrics.get('sharpe', 0),
                "Flips": metrics.get('flips', 0),
                "Turnover": metrics.get('turnover', 0),
                "Long steps": metrics.get('long_steps', 0),
                "Short steps": metrics.get('short_steps', 0)
            }
            results.append(row)
            print(f"  Completed: {row}")
        else:
            print(f"  Failed to run {name}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/dev/reward_ablation_results.csv", index=False)
    print("\nResults saved to results/dev/reward_ablation_results.csv")
    
    # Print markdown table
    print("\nResults Table:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()