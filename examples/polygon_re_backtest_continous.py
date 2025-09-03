#!/usr/bin/env python3
"""
Example: RL Backtest with Polygon Data (Episodic or Continuous)

This script is based on the original polygon_rl_backtest_example.py, with the
minimum changes necessary to:
  1) Support a --continuous mode that runs a single episode across the entire
     date range by disabling end-of-day resets.
  2) Robustly handle parquet data that lacks a recognized timestamp column name
     by falling back to a manual RAW loader that reconstructs a proper
     timestamp index and returns a clean OHLCV dataframe.

Usage:
  python examples/polygon_re_backtest_continous.py \
    --symbol SPY \
    --start-date 2024-01-01 \
    --end-date 2024-06-30 \
    --model-path rl-intraday/models/trained_model.zip \
    --continuous \
    --plot
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is on sys.path (same as original)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import UnifiedDataLoader
from src.features.pipeline import FeaturePipeline
from src.sim.env_intraday_rl import IntradayRLEnv
from src.utils.config_loader import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ----------------------------- Helpers -----------------------------

def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a proper timestamp column and DateTimeIndex localized to
    America/New_York. Accepts common names or uses a DatetimeIndex if present.
    """
    if df is None or df.empty:
        return df

    # Candidate timestamp columns in decreasing preference
    candidates = ["timestamp", "datetime", "time", "dt", "ts", "t"]

    ts_col = None
    for c in candidates:
        if c in df.columns:
            ts_col = c
            break

    # If no candidate column found but index looks like datetime, use it
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            # Move index to a named column for uniform handling
            df = df.reset_index()
            df = df.rename(columns={"index": "timestamp"})
            ts_col = "timestamp"
        else:
            # Last resort: try to auto-detect any column with datetime-ish dtype
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    ts_col = c
                    break

    if ts_col is None:
        raise ValueError("No usable timestamp/datetime column or index found in DataFrame")

    # Coerce to UTC, then convert to America/New_York
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.loc[ts.notna()].copy()
    df[ts_col] = ts
    df = df.sort_values(ts_col).set_index(ts_col)

    try:
        # If already tz-aware in UTC, convert; if tz-naive, localize to UTC first
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
    except Exception:
        # Fallback: force UTC then convert
        df.index = df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT").tz_convert("America/New_York")

    # Drop duplicate timestamps if any
    df = df[~df.index.duplicated(keep="first")]

    return df


def _iter_polygon_raw_files(raw_dir: Path, symbol: str) -> List[Path]:
    """
    Find all partitioned RAW parquet files for the given symbol:
      raw_dir / symbol=SPY / year=YYYY / month=MM / day=DD / data.parquet
    """
    sym_dir = raw_dir / f"symbol={symbol.upper()}"
    if not sym_dir.exists():
        logger.debug("Symbol RAW dir does not exist: %s", sym_dir)
        return []
    return sorted(sym_dir.rglob("data.parquet"))


def _manual_load_polygon_raw(settings: Settings, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Manually reconstruct OHLCV from RAW partitioned parquet files when the
    UnifiedDataLoader cannot find a suitable timestamp column in cached/raw data.
    """
    paths = settings.paths
    raw_dir = Path(paths.get("polygon_raw_dir"))
    files = _iter_polygon_raw_files(raw_dir, symbol)
    if not files:
        raise FileNotFoundError(f"No RAW parquet files found under {raw_dir}/symbol={symbol.upper()}")

    # Load and concatenate, then filter by date
    dfs = []
    for p in files:
        try:
            part = pd.read_parquet(p)
            if part is None or part.empty:
                continue
            part = _normalize_timestamp_column(part)
            dfs.append(part)
        except Exception as ex:
            logger.debug("Skipping %s due to read/normalize error: %s", p, ex)

    if not dfs:
        raise ValueError("Unable to reconstruct RAW data; all parquet reads failed or lacked timestamps.")

    df = pd.concat(dfs, axis=0, ignore_index=False)
    # Filter by provided date range in local market time
    s = pd.Timestamp(start_date).tz_localize("America/New_York")
    e = pd.Timestamp(end_date)
    # Treat midnight 'end' as exclusive next-day to mirror loader's normalization
    if e.tzinfo is None:
        e = e.tz_localize("America/New_York")
    if e.hour == 0 and e.minute == 0 and e.second == 0:
        e = e + pd.Timedelta(days=1)

    df = df.loc[(df.index >= s) & (df.index < e)]

    # Basic column sanity: ensure standard OHLCV exist (rename if needed)
    rename_map = {}
    if "o" in df.columns and "open" not in df.columns:
        rename_map["o"] = "open"
    if "h" in df.columns and "high" not in df.columns:
        rename_map["h"] = "high"
    if "l" in df.columns and "low" not in df.columns:
        rename_map["l"] = "low"
    if "c" in df.columns and "close" not in df.columns:
        rename_map["c"] = "close"
    if "v" in df.columns and "volume" not in df.columns:
        rename_map["v"] = "volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Keep typical columns if present
    cols_pref = [c for c in ["open", "high", "low", "close", "volume", "vwap", "transactions"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols_pref]
    df = df[cols_pref + other_cols] if cols_pref else df

    logger.info("Manual RAW load complete: %d rows, %d cols", len(df), df.shape[1])
    return df


# -------------------------- Original-ish code --------------------------

def load_polygon_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load Polygon data using UnifiedDataLoader; if it raises the timestamp error,
    fall back to a manual RAW loader that reconstructs a proper datetime index.
    """
    logger.info(f"Loading Polygon data for {symbol} from {start_date} to {end_date}")

    config_path = Path(__file__).parent.parent / 'configs' / 'settings.yaml'
    settings = Settings(config_path=str(config_path))

    # Prefer bypassing cache to avoid stale parquet timestamp issues
    loader = UnifiedDataLoader(
        data_source='polygon',
        config_path=str(config_path),
        cache_enabled=False,            # <--- avoid cached parquet causing timestamp errors
        default_timeframe='1min'
    )

    try:
        df = loader.load(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe='1min'
        )
        # If loader returned something with index but no col, normalize anyway
        df = _normalize_timestamp_column(df)
        logger.info(f"Loaded {len(df)} records for {symbol}")
        return df

    except ValueError as e:
        msg = str(e).lower()
        if "no timestamp/datetime column" in msg:
            logger.warning("UnifiedDataLoader failed due to timestamp column detection; falling back to manual RAW load.")
            df = _manual_load_polygon_raw(settings, symbol, start_date, end_date)
            logger.info(f"Loaded {len(df)} records for {symbol} via RAW fallback")
            return df
        else:
            logger.error("Loader error: %s", e)
            raise


def create_feature_pipeline() -> FeaturePipeline:
    """
    Create a feature pipeline configuration for Polygon data.
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


def run_backtest_simulation(env: IntradayRLEnv, model, num_episodes: int = 5) -> dict:
    """
    Run backtest simulation with the provided environment and model.

    Args:
        env: RL environment
        model: Trained PPO model (stable-baselines3)
        num_episodes: Number of episodes (days) to run

    Returns:
        Dict of summary stats and per-episode metrics.
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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        # Compute episode return from equity curve
        equity_curve = env.get_equity_curve()
        if len(equity_curve) >= 2:
            start_equity = equity_curve.iloc[0]
            end_equity = equity_curve.iloc[-1]
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
            'final_equity': equity_curve.iloc[-1]
                if len(equity_curve) > 0 and not np.isnan(equity_curve.iloc[-1])
                else env.cash
        })

        logger.info(
            f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
            f"Steps={episode_steps}, Return={episode_return:.4%}"
        )

    # Aggregate stats
    results['total_return'] = sum(ep['return'] for ep in results['episodes']) / num_episodes
    results['total_steps'] = total_steps
    results['avg_reward'] = total_reward / total_steps if total_steps > 0 else 0.0
    results['win_rate'] = winning_episodes / num_episodes

    return results


def run_continuous_backtest(env: IntradayRLEnv, model) -> dict:
    """
    Run a single continuous episode across the entire date range.
    """
    logger.info("Running continuous backtest...")

    total_reward = 0.0
    total_steps = 0

    obs, info = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1

    equity_curve = env.get_equity_curve()
    if len(equity_curve) >= 2:
        start_equity = equity_curve.iloc[0]
        end_equity = equity_curve.iloc[-1]
        if np.isnan(start_equity) or np.isnan(end_equity) or start_equity == 0:
            episode_return = 0.0
        else:
            episode_return = (end_equity - start_equity) / start_equity
    else:
        episode_return = 0.0

    result = {
        'episodes': [{
            'episode': 1,
            'reward': total_reward,
            'steps': total_steps,
            'return': episode_return,
            'final_equity': equity_curve.iloc[-1]
                if len(equity_curve) > 0 and not np.isnan(equity_curve.iloc[-1])
                else env.cash
        }],
        'total_return': episode_return,
        'total_steps': total_steps,
        'avg_reward': total_reward / total_steps if total_steps > 0 else 0.0,
        'win_rate': 1.0 if episode_return > 0 else 0.0
    }

    logger.info(
        f"Continuous backtest: Reward={total_reward:.2f}, "
        f"Steps={total_steps}, Return={episode_return:.4%}"
    )
    return result


class ContinuousIntradayRLEnv(IntradayRLEnv):
    """
    Subclass of IntradayRLEnv that disables end-of-day termination so the episode
    spans the full data range. All other behavior is unchanged.
    """
    def _eod(self, ts):
        return False


def plot_results(results: dict, symbol: str):
    """
    Plot backtest results.
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'RL Backtest Results - {symbol}', fontsize=16)

        # Episode rewards
        episodes = [ep['episode'] for ep in results['episodes']]
        rewards = [ep['reward'] for ep in results['episodes']]
        ax1.bar(episodes, rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)

        # Episode returns
        returns = [ep['return'] * 100 for ep in results['episodes']]
        ax2.bar(episodes, returns)
        ax2.set_title('Episode Returns (%)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)

        # Final equity
        equity = [ep['final_equity'] for ep in results['episodes']]
        ax3.plot(episodes, equity, 'o-', linewidth=2, markersize=8)
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
        ax4.text(
            0.1, 0.8, summary_text, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray")
        )

        plt.tight_layout()
        plt.savefig(f'backtest_results_{symbol}.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Results plot saved as backtest_results_{symbol}.png")

    except ImportError:
        logger.warning("Matplotlib not available for plotting")
    except Exception as e:
        logger.error(f"Error creating plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="RL Backtest with Polygon Data")
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock symbol')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-03-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--plot', action='store_true', help='Generate result plots')
    parser.add_argument('--model-path', type=str, help='Path to the trained model')
    parser.add_argument('--continuous', action='store_true',
                        help='Run a single continuous backtest over the entire date range')
    args = parser.parse_args()

    print(f"RL Backtest Example - {args.symbol}")
    print("=" * 50)
    print(f"Date Range: {args.start_date} to {args.end_date}")
    if args.continuous:
        print("Mode: Continuous backtest over entire date range")
    else:
        print(f"Episodes: {args.episodes}")
    print()

    try:
        # Load data
        print("Loading data...")
        ohlcv_data = load_polygon_data(args.symbol, args.start_date, args.end_date)
        if ohlcv_data is None or ohlcv_data.empty:
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

        # Initialize environment (choose episodic or continuous)
        print("Initializing RL environment...")
        config_path = Path(__file__).parent.parent / 'configs' / 'settings.yaml'
        settings = Settings(config_path=str(config_path))
        EnvClass = ContinuousIntradayRLEnv if args.continuous else IntradayRLEnv
        env = EnvClass(
            ohlcv=ohlcv_data,
            features=features,
            cash=100000.0,
            point_value=5.0,  # MES contract multiplier
            config=settings.env
        )
        print("Environment initialized successfully")
        print()

        # Load model
        if not args.model_path:
            raise ValueError("A trained model must be provided via --model-path")

        from stable_baselines3 import PPO
        print(f"Loading model from {args.model_path}")
        model = PPO.load(args.model_path)

        # Run backtest
        print("Running backtest simulation...")
        if args.continuous:
            results = run_continuous_backtest(env, model)
        else:
            results = run_backtest_simulation(env, model, args.episodes)
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
            print(
                f"Episode {ep['episode']:2d}: "
                f"Reward={ep['reward']:8.2f}, "
                f"Steps={ep['steps']:6d}, "
                f"Return={ep['return']:7.2%}, "
                f"Final Equity=${ep['final_equity']:9.0f}"
            )

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
    # Configure logging similarly to the original
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
