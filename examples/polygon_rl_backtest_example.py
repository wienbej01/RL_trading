#!/usr/bin/env python3
"""
Example: RL Backtest with Polygon Data

This example demonstrates how to use collected Polygon data for RL training
and backtesting. It shows the complete workflow from data loading to
environment testing.

Usage:
    python examples/polygon_rl_backtest_example.py --symbol SPY --start-date 2024-01-01 --end-date 2024-03-31
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import UnifiedDataLoader
from src.features.pipeline import FeaturePipeline
from src.sim.env_intraday_rl import IntradayRLEnv
from src.utils.config_loader import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_polygon_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load Polygon data using the unified data loader.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading Polygon data for {symbol} from {start_date} to {end_date}")

    config_path = Path(__file__).parent.parent / 'configs' / 'settings.yaml'
    loader = UnifiedDataLoader(data_source='polygon', config_path=str(config_path))

    df = loader.load(
        symbol=symbol,
        start=start_date,
        end=end_date,
        timeframe='1min'
    )

    logger.info(f"Loaded {len(df)} records for {symbol}")
    return df


def create_feature_pipeline():
    """
    Create feature pipeline configuration for Polygon data.

    Returns:
        Configured FeaturePipeline
    """
    config = {
        'data_source': 'polygon',
        'technical': {
            'calculate_returns': True,
            'calculate_log_returns': True,
            'sma_windows': [5, 10, 20, 50],
            'ema_windows': [5, 10, 20, 50],
            'calculate_atr': True,
            'atr_window': 14,
            'calculate_rsi': True,
            'rsi_window': 14,
            'calculate_macd': True,
            'calculate_bollinger_bands': True,
            'bollinger_window': 20,
            'calculate_stochastic': True,
            'calculate_williams_r': True
        },
        'microstructure': {
            'calculate_spread': True,
            'calculate_microprice': True,
            'calculate_queue_imbalance': True,
            'calculate_vwap': True,
            'calculate_twap': True,
            'calculate_price_impact': True
        },
        'time': {
            'extract_time_of_day': True,
            'extract_day_of_week': True,
            'extract_session_features': True
        },
        'polygon': {
            'features': {
                'use_vwap_column': True
            },
            'quality_checks': {
                'enabled': True
            }
        }
    }

    return FeaturePipeline(config)


def run_backtest_simulation(env: IntradayRLEnv, num_episodes: int = 5) -> dict:
    """
    Run backtest simulation with random actions.

    Args:
        env: RL environment
        num_episodes: Number of episodes to run

    Returns:
        Simulation results
    """
    logger.info(f"Running backtest simulation with {num_episodes} episodes")

    results = {
        'episodes': [],
        'total_return': 0.0,
        'total_steps': 0,
        'avg_reward': 0.0,
        'win_rate': 0.0
    }

    total_reward = 0.0
    total_steps = 0
    winning_episodes = 0

    for episode in range(num_episodes):
        episode_reward = 0.0
        episode_steps = 0

        obs, info = env.reset()

        done = False
        while not done:
            # Random action for demonstration
            action = np.random.randint(0, 3)

            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        # Calculate episode return with NaN handling
        equity_curve = env.get_equity_curve()
        if len(equity_curve) >= 2:
            start_equity = equity_curve.iloc[0]
            end_equity = equity_curve.iloc[-1]
            
            # Handle NaN values
            if np.isnan(start_equity) or np.isnan(end_equity) or start_equity == 0:
                episode_return = 0.0
            else:
                episode_return = (end_equity - start_equity) / start_equity
        else:
            episode_return = 0.0

        if episode_return > 0:
            winning_episodes += 1

        total_reward += episode_reward

        results['episodes'].append({
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': episode_steps,
            'return': episode_return,
            'final_equity': equity_curve.iloc[-1] if len(equity_curve) > 0 and not np.isnan(equity_curve.iloc[-1]) else env.cash
        })

        logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                   f"Steps={episode_steps}, Return={episode_return:.4%}")

    results['total_return'] = sum(ep['return'] for ep in results['episodes']) / num_episodes
    results['total_steps'] = total_steps
    results['avg_reward'] = total_reward / total_steps if total_steps > 0 else 0.0
    results['win_rate'] = winning_episodes / num_episodes

    return results


def plot_results(results: dict, symbol: str):
    """
    Plot backtest results.

    Args:
        results: Simulation results
        symbol: Stock symbol
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'RL Backtest Results - {symbol}', fontsize=16)

        # Episode rewards
        episodes = [ep['episode'] for ep in results['episodes']]
        rewards = [ep['reward'] for ep in results['episodes']]
        ax1.bar(episodes, rewards, color='skyblue')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)

        # Episode returns
        returns = [ep['return'] * 100 for ep in results['episodes']]
        ax2.bar(episodes, returns, color='lightgreen')
        ax2.set_title('Episode Returns (%)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)

        # Final equity
        equity = [ep['final_equity'] for ep in results['episodes']]
        ax3.plot(episodes, equity, 'o-', color='orange', linewidth=2, markersize=8)
        ax3.set_title('Final Equity')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Equity ($)')
        ax3.grid(True, alpha=0.3)

        # Summary statistics
        ax4.axis('off')
        summary_text = f"""
SUMMARY STATISTICS

Total Episodes: {len(results['episodes'])}
Average Reward: {results['avg_reward']:.4f}
Win Rate: {results['win_rate']:.1%}
Average Return: {results['total_return']:.2%}
Total Steps: {results['total_steps']}
"""
        ax4.text(0.1, 0.8, summary_text, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()
        plt.savefig(f'backtest_results_{symbol}.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Results plot saved as backtest_results_{symbol}.png")

    except ImportError:
        logger.warning("Matplotlib not available for plotting")
    except Exception as e:
        logger.error(f"Error creating plot: {e}")


def main():
    """Main example function."""
    parser = argparse.ArgumentParser(description="RL Backtest with Polygon Data")
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock symbol')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-03-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--plot', action='store_true', help='Generate result plots')

    args = parser.parse_args()

    print(f"RL Backtest Example - {args.symbol}")
    print("=" * 50)
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Episodes: {args.episodes}")
    print()

    try:
        # Load data
        print("Loading data...")
        ohlcv_data = load_polygon_data(args.symbol, args.start_date, args.end_date)

        if ohlcv_data.empty:
            print(f"No data available for {args.symbol}")
            return

        print(f"Loaded {len(ohlcv_data)} records")
        print(f"Date range: {ohlcv_data.index.min()} to {ohlcv_data.index.max()}")
        print()

        # Create features
        print("Creating features...")
        pipeline = create_feature_pipeline()
        features = pipeline.fit_transform(ohlcv_data)
        print(f"Created {len(features.columns)} features")
        print()

        # Initialize environment
        print("Initializing RL environment...")
        env = IntradayRLEnv(
            ohlcv=ohlcv_data,
            features=features,
            cash=100000.0,
            point_value=5.0  # MES contract multiplier
        )
        print("Environment initialized successfully")
        print()

        # Run backtest
        print("Running backtest simulation...")
        results = run_backtest_simulation(env, args.episodes)
        print()

        # Print results
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Episodes: {len(results['episodes'])}")
        print(f"Average Reward: {results['avg_reward']:.4f}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Average Return: {results['total_return']:.2%}")
        print(f"Total Steps: {results['total_steps']}")
        print()

        # Detailed episode results
        print("Episode Details:")
        for ep in results['episodes']:
            print(f"Episode {ep['episode']:2d}: Reward={ep['reward']:8.2f}, Steps={ep['steps']:6d}, Return={ep['return']:7.2%}, Final Equity=${ep['final_equity']:9.0f}")

        # Generate plots if requested
        if args.plot:
            print("\nGenerating plots...")
            plot_results(results, args.symbol)

        print("\nBacktest completed successfully!")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()
