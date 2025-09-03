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
logging.getLogger("src.data.data_loader").setLevel(logging.ERROR)

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
    Return df with a proper DateTimeIndex in America/New_York named 'timestamp'.
    Handles:
      - explicit columns: 'timestamp','datetime','time','dt','ts','t',
        'sip_timestamp','participant_timestamp','unix'
      - datetime-like index
      - numeric epoch in ns/ms/s (auto-detected)
    Deduplicates + sorts ascending.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="America/New_York", name="timestamp"))

    # 1) Find a likely timestamp column
    candidates = (
        "timestamp", "datetime", "time", "dt", "ts", "t",
        "sip_timestamp", "participant_timestamp", "unix"
    )
    ts_col = next((c for c in candidates if c in df.columns), None)

    # If not found, try the index
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
            ts_col = "timestamp"
        else:
            # If any column is already datetime dtype, use that
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    ts_col = c
                    break

    if ts_col is None:
        # Last attempt: is there any single numeric-like col that could be epoch?
        numeric_hint = next(
            (c for c in df.columns if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c])),
            None
        )
        if numeric_hint is None:
            raise ValueError("No timestamp-like column, datetime index, or numeric epoch candidate found.")
        ts_col = numeric_hint

    # 2) Build a UTC-aware Timestamp Series from the chosen column
    s = df[ts_col]

    # If it's already datetime-like, coerce with utc=True
    if pd.api.types.is_datetime64_any_dtype(s):
        ts = pd.to_datetime(s, utc=True, errors="coerce")

    else:
        # It may be epoch in ns/ms/s; try those units in order of likelihood for Polygon
        ts = pd.to_datetime(s, utc=True, errors="coerce", unit="ns")
        if ts.isna().all():
            ts = pd.to_datetime(s, utc=True, errors="coerce", unit="ms")
        if ts.isna().all():
            ts = pd.to_datetime(s, utc=True, errors="coerce", unit="s")

        # If still all NaT, try a generic parse (strings like '2024-01-02T...')
        if ts.isna().all():
            ts = pd.to_datetime(s, utc=True, errors="coerce")

    # Drop rows where we couldn't form a timestamp
    mask = ts.notna()
    if not mask.any():
        raise ValueError(f"Could not parse timestamps from column '{ts_col}'.")

    out = df.loc[mask].copy()
    out["timestamp"] = ts[mask]

    # 3) Sort, index, tz:America/New_York, dedupe
    out = out.sort_values("timestamp").set_index("timestamp")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index = out.index.tz_convert("America/New_York")
    out.index.name = "timestamp"

    # Deduplicate timestamps
    out = out[~out.index.duplicated(keep="first")]

    return out


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

def _manual_load_polygon_raw(
    settings: "Settings",
    symbol: str,
    start_date: "str | pd.Timestamp",
    end_date: "str | pd.Timestamp",
) -> pd.DataFrame:
    """
    Manual RAW loader for Polygon daily parquet partitions:
      data/polygon/historical/symbol=SYMBOL/year=YYYY/month=MM/day=DD/data.parquet

    - Reads each file, normalizes timestamps via _normalize_timestamp_column (robust).
    - Concatenates, sorts, de-dupes.
    - Filters to [start_date, end_date] in America/New_York.
    """
    import pandas as pd
    from pathlib import Path

    tz = "America/New_York"
    # Parse inputs and make them tz-aware in market time
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if s.tz is None:
        s = s.tz_localize(tz)
    else:
        s = s.tz_convert(tz)
    if e.tz is None:
        e = e.tz_localize(tz)
    else:
        e = e.tz_convert(tz)

    # Resolve RAW root from settings (with sane default)
    try:
        raw_root = Path(settings.get("paths", "polygon_raw_dir"))
    except Exception:
        raw_root = (Path(__file__).resolve().parents[1] / "data" / "polygon" / "historical").resolve()

    # Enumerate days (inclusive)
    days = pd.date_range(s.normalize(), e.normalize(), freq="D")
    parts: list[pd.DataFrame] = []
    total = 0
    bad = 0

    for d in days:
        total += 1
        y = d.year
        m = f"{d.month:02d}"
        dd = f"{d.day:02d}"
        path = raw_root / f"symbol={symbol}" / f"year={y}" / f"month={m}" / f"day={dd}" / "data.parquet"
        if not path.exists():
            continue
        try:
            df_part = pd.read_parquet(path)
            df_part = _normalize_timestamp_column(df_part)  # <-- robust: no 'timestamp' assumption
            # Keep only in-range rows (include end day; use < e+1 day to be safe)
            mask = (df_part.index >= s) & (df_part.index < (e + pd.Timedelta(days=1)))
            df_part = df_part.loc[mask]
            if not df_part.empty:
                parts.append(df_part)
        except Exception as ex:
            bad += 1
            logger.warning("RAW: failed to load %s: %s", path, ex)

    if not parts:
        logger.info("RAW: loaded 0/%d partitions (bad=%d) -> empty frame", total, bad)
        # Empty, but keep correct typed/tz index
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=tz, name="timestamp"))

    df = pd.concat(parts, axis=0, copy=False)
    # Safety: sort, dedupe, name index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = _normalize_timestamp_column(df)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "timestamp"

    first = df.index.min()
    last = df.index.max()
    logger.info("RAW: loaded %d/%d partitions (bad=%d); rows=%d range=[%s .. %s] tz=%s",
                len(parts), total, bad, len(df), first, last, df.index.tz)

    return df

def load_polygon_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Preferred path: UnifiedDataLoader with cache enabled.
    If the cache is empty or a timestamp/datetime error occurs, fall back to RAW,
    then rebuild the cache so subsequent runs are fast and stable.

    Always returns a DataFrame indexed by NY-tz 'timestamp'.
    """
    import pandas as pd
    from pathlib import Path

    logger.info(f"Loading Polygon data for {symbol} from {start_date} to {end_date}")

    # Settings/config path
    config_path = Path(__file__).parent.parent / 'configs' / 'settings.yaml'
    settings = Settings(config_path=str(config_path))

    # Build Unified loader (cache ON)
    loader = UnifiedDataLoader(
        data_source='polygon',
        config_path=str(config_path),
        cache_enabled=True,
        default_timeframe='1min'
    )

    # Convert to Timestamps
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)

    # Compute expected cache path (mirrors loader naming)
    try:
        cache_dir = Path(settings.get("paths", "cache_dir"))
    except Exception:
        cache_dir = (Path(__file__).resolve().parents[1] / "data" / "cache").resolve()
    end_plus = (e + pd.Timedelta(days=1)).strftime("%Y%m%d")  # loader uses exclusive end
    cache_name = f"{symbol.upper()}_{s.strftime('%Y%m%d')}_{end_plus}_ohlcv_1min.parquet"
    cache_path = cache_dir / cache_name

    def _fallback_from_raw_and_refresh_cache() -> pd.DataFrame:
        """RAW fallback + rebuild cache if we got data."""
        df_raw = _manual_load_polygon_raw(settings, symbol, s, e)
        if not df_raw.empty:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df_raw.to_parquet(cache_path)
                logger.info(f"Rebuilt cache at {cache_path} with {len(df_raw)} rows")
            except Exception as ce:
                logger.warning(f"Failed to rebuild cache {cache_path}: {ce}")
        else:
            logger.info("RAW fallback produced empty frame.")
        return df_raw

    try:
        df = loader.load_ohlcv(
            symbol=symbol,
            start=s,
            end=e,
            timeframe='1min',
            use_cache=True
        )
        # Normalize to canonical NY-tz index
        df = _normalize_timestamp_column(df)

        # If cache produced empty → purge + RAW fallback + rebuild cache
        if df is None or df.empty:
            logger.warning("UnifiedDataLoader returned empty frame; purging cache and falling back to RAW.")
            try:
                if cache_path.exists():
                    cache_path.unlink(missing_ok=True)
            except Exception as ue:
                logger.warning(f"Failed to unlink stale cache {cache_path}: {ue}")
            df = _fallback_from_raw_and_refresh_cache()

        logger.info(f"Loaded {len(df)} records for {symbol}")
        return df

    except Exception as ex:
        msg = (str(ex) or "").lower()
        # Timestamp/datetime-related → RAW fallback path
        if "timestamp" in msg or "datetime" in msg:
            logger.warning("UnifiedDataLoader failed due to timestamp/datetime detection; falling back to RAW.")
            # Best-effort purge of possibly-bad cache
            try:
                if cache_path.exists():
                    cache_path.unlink(missing_ok=True)
            except Exception as ue:
                logger.warning(f"Failed to unlink stale cache {cache_path}: {ue}")
            df = _fallback_from_raw_and_refresh_cache()
            logger.info(f"Loaded {len(df)} records for {symbol} via RAW fallback")
            return df

        # Other exceptions — surface them
        logger.error(f"Loader error (non-timestamp-related): {ex}")
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
def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure env-safe OHLCV:
      - Guarantee a 'close' column (fallback: vwap -> price -> open).
      - Drop rows where close is NaN or not finite.
      - If O/H/L exist, drop rows with any NaN in them (avoid synthetic OHLC).
      - Volume/vwap/transactions: coerce numeric; fill NaN volume with 0.
    Returns a copy.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # 1) Ensure we have a 'close' column
    if "close" not in out.columns:
        for alt in ("vwap", "price", "open"):
            if alt in out.columns:
                out["close"] = out[alt]
                break
    if "close" not in out.columns:
        # No usable price column at all -> empty (better than NaNs in env)
        return out.iloc[0:0]

    # 2) Coerce candidate numeric columns to numeric
    num_cols = [c for c in ("open", "high", "low", "close", "volume", "vwap", "transactions") if c in out.columns]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # 3) Drop rows with invalid close
    mask_close = out["close"].notna() & np.isfinite(out["close"]) & (out["close"] > 0)
    out = out.loc[mask_close]

    # 4) If OHLC exist, drop rows where any is NaN (safer than forward-filling)
    ohlc_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
    if ohlc_cols:
        out = out.dropna(subset=ohlc_cols, how="any")

    # 5) Volume: fill NaN with 0, clamp negatives to 0
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0)
        out.loc[out["volume"] < 0, "volume"] = 0

    # 6) Final tidy: sort, dedup, name index
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "timestamp"

    return out

def _apply_rth_resample_prune(df: pd.DataFrame, settings: "Settings") -> pd.DataFrame:
    """
    Enforce Regular Trading Hours and exact 1-minute bars, then prune sparse days.
    Finally sanitize to guarantee non-NaN close prices for the env.

    Config keys (defaults used if missing):
      data.session.tz:           "America/New_York"
      data.session.rth_start:    "09:30"
      data.session.rth_end:      "16:00"
      data.resample.strict:      True
      data.prune.min_minutes_per_day: 0   (disabled if 0)
    """
    if df is None or df.empty:
        logger.info("Usable records after RTH+resample+gap-prune: 0")
        return df

    # ----- config with defaults -----
    try:
        tz = settings.get("data", "session", "tz", default="America/New_York")
    except Exception:
        tz = "America/New_York"
    try:
        rth_start = settings.get("data", "session", "rth_start", default="09:30")
    except Exception:
        rth_start = "09:30"
    try:
        rth_end = settings.get("data", "session", "rth_end", default="16:00")
    except Exception:
        rth_end = "16:00"
    try:
        strict = bool(settings.get("data", "resample", "strict", default=True))
    except Exception:
        strict = True
    try:
        min_minutes = int(settings.get("data", "prune", "min_minutes_per_day", default=0))
    except Exception:
        min_minutes = 0

    out = df.copy()

    # ----- ensure NY-tz index -----
    if not isinstance(out.index, pd.DatetimeIndex):
        out = _normalize_timestamp_column(out)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index = out.index.tz_convert(tz)

    # ----- RTH filter -----
    out = out.between_time(rth_start, rth_end, inclusive="left")

    # ----- build resample aggregation map -----
    agg = {}
    if "open" in out.columns:
        agg["open"] = "first"
    if "high" in out.columns:
        agg["high"] = "max"
    if "low" in out.columns:
        agg["low"] = "min"
    if "close" in out.columns:
        agg["close"] = "last"
    if "volume" in out.columns:
        agg["volume"] = "sum"
    if "vwap" in out.columns:
        agg["vwap"] = "last"
    if "transactions" in out.columns:
        agg["transactions"] = "sum"

    if not agg:
        # No known fields -> keep row count as a last-resort metric
        out["rows"] = 1
        agg = {"rows": "sum"}

    # ----- strict 1-minute resample -----
    if strict:
        out = out.resample("1min").agg(agg)

    # ----- prune sparse days if configured -----
    if min_minutes and min_minutes > 0 and not out.empty:
        counts = out.groupby(out.index.normalize()).size()
        bad_days = counts[counts < min_minutes].index
        if len(bad_days):
            out = out[~out.index.normalize().isin(bad_days)]

    # ----- sanitize OHLCV to remove NaN prices before env sees it -----
    out = _sanitize_ohlcv(out)

    # ----- logging summary -----
    usable = len(out)
    if usable:
        first = out.index.min()
        last = out.index.max()
        logger.info(f"Usable records after RTH+resample+gap-prune: {usable}")
        logger.info(f"Usable date range: {first} to {last}")
    else:
        logger.info("Usable records after RTH+resample+gap-prune: 0")

    return out


def run_backtest_simulation(env, model, episodes: int = 5) -> dict:
    """
    Run an episodic backtest for `episodes` episodes.

    - Compatible with Gym (obs, reward, done, info) and Gymnasium
      (obs, reward, terminated, truncated, info).
    - Tracks per-episode rewards, steps, equity, and returns.
    - Avoids off-by-one by appending exactly once per environment step.
    """
    import numpy as np

    def _step_unpack(step_out):
        # Support both Gym and Gymnasium signatures
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
            return obs, reward, done, info
        elif len(step_out) == 4:
            obs, reward, done, info = step_out
            return obs, reward, bool(done), info
        else:
            raise RuntimeError("Unknown env.step(...) return signature")

    ep_summaries = []

    for ep in range(1, int(episodes) + 1):
        # Reset per episode
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, reset_info = reset_out
        else:
            obs, reset_info = reset_out, {}

        # Establish initial equity baseline
        initial_equity = getattr(env, "cash", None)
        if initial_equity is None:
            initial_equity = getattr(env, "initial_cash", 100_000.0)
        try:
            initial_equity = float(reset_info.get("equity", initial_equity))
        except Exception:
            pass

        rewards = []
        equities = []
        steps = 0
        done = False
        last_equity = float(initial_equity)

        while not done:
            # Model action (deterministic during evaluation)
            if hasattr(model, "predict"):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = 0  # hold/no-op fallback

            obs, reward, done, info = _step_unpack(env.step(action))
            steps += 1
            rewards.append(float(reward))

            eq = info.get("equity", last_equity)
            try:
                eq = float(eq)
            except Exception:
                eq = last_equity
            equities.append(eq)
            last_equity = eq

        total_reward = float(np.nansum(rewards)) if rewards else 0.0
        final_equity = float(equities[-1]) if equities else float(initial_equity)
        ep_return = (final_equity / float(initial_equity) - 1.0) if initial_equity else 0.0

        ep_summaries.append(
            {
                "episode": ep,
                "reward": total_reward,
                "steps": steps,
                "return": ep_return,
                "final_equity": final_equity,
                "equity_series": equities,  # optional for plotting
            }
        )

    # Aggregate metrics
    rewards_arr = np.array([e["reward"] for e in ep_summaries], dtype=float) if ep_summaries else np.array([0.0])
    returns_arr = np.array([e["return"] for e in ep_summaries], dtype=float) if ep_summaries else np.array([0.0])
    steps_sum = int(np.nansum([e["steps"] for e in ep_summaries])) if ep_summaries else 0

    results = {
        "episodes": ep_summaries,
        "avg_reward": float(np.nanmean(rewards_arr)),
        "win_rate": float(np.mean(returns_arr > 0.0)) if ep_summaries else 0.0,
        "total_return": float(np.nanmean(returns_arr)),  # average return across episodes
        "total_steps": steps_sum,
    }

    # Try to close env neatly
    try:
        env.close()
    except Exception:
        pass

    return results


def run_continuous_backtest(env, model) -> dict:
    """
    Run a single continuous backtest over the entire dataset exposed by `env`.

    - Works with Gym (obs, reward, done, info) and Gymnasium
      (obs, reward, terminated, truncated, info) step signatures.
    - Tracks rewards, steps, and equity (from info['equity'] if provided).
    - Avoids off-by-one by appending exactly once per environment step.
    - Returns a results dict compatible with the existing printing/plotting code.
    """
    import numpy as np

    def _step_unpack(step_out):
        # Support both Gym and Gymnasium signatures
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
            return obs, reward, done, info
        elif len(step_out) == 4:
            obs, reward, done, info = step_out
            return obs, reward, bool(done), info
        else:
            raise RuntimeError("Unknown env.step(...) return signature")

    # Reset env
    reset_out = env.reset()
    # Gymnasium returns (obs, info); Gym returns obs
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, reset_info = reset_out
    else:
        obs, reset_info = reset_out, {}

    # Pull initial cash if the env exposes it; default to 100_000
    initial_equity = getattr(env, "cash", None)
    if initial_equity is None:
        initial_equity = getattr(env, "initial_cash", 100_000.0)
    try:
        # Some envs track a current equity in info at reset
        initial_equity = float(reset_info.get("equity", initial_equity))
    except Exception:
        pass

    rewards = []
    steps = 0
    equities = []

    done = False
    last_equity = float(initial_equity)

    while not done:
        # Model action (deterministic for eval)
        if hasattr(model, "predict"):
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Fallback: hold/no-op if model interface is different
            action = 0

        obs, reward, done, info = _step_unpack(env.step(action))
        steps += 1
        rewards.append(float(reward))

        # Track equity if provided; otherwise carry forward last known value
        eq = info.get("equity", last_equity)
        try:
            eq = float(eq)
        except Exception:
            eq = last_equity
        equities.append(eq)
        last_equity = eq

    # Final stats
    total_reward = float(np.nansum(rewards)) if rewards else 0.0
    final_equity = float(equities[-1]) if equities else float(initial_equity)

    # Compute return robustly
    if initial_equity and initial_equity != 0:
        total_return = (final_equity / float(initial_equity)) - 1.0
    else:
        total_return = 0.0

    # Build "episodes" list compatible with existing reporting
    ep_summary = {
        "episode": 1,
        "reward": total_reward,
        "steps": steps,
        "return": total_return,
        "final_equity": final_equity,
        # optional: store equity series for plotting if your plotter uses it
        "equity_series": equities,
    }

    results = {
        "episodes": [ep_summary],
        "avg_reward": total_reward / 1.0,
        "win_rate": 1.0 if total_return > 0 else 0.0,
        "total_return": total_return,
        "total_steps": steps,
    }

    # Best effort to close the env if it exposes close()
    try:
        env.close()
    except Exception:
        pass

    return results


def get_clean_ohlcv(symbol: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, "Settings"]:
    """
    Main flow entry point for OHLCV:
      1) UnifiedDataLoader (cache ON) -> timestamp error fallback to manual RAW.
      2) RTH + strict 1-min resample + prune (from settings.yaml).
      3) Returns (df, settings) where df is NY-tz indexed by 'timestamp'.
    """
    from pathlib import Path
    import pandas as pd

    # Load settings once here; return it so downstream (features/env) use the same config instance.
    config_path = Path(__file__).parent.parent / 'configs' / 'settings.yaml'
    settings = Settings(config_path=str(config_path))

    # Step 1: load via Unified (or RAW fallback)
    df = load_polygon_data(symbol, start_date, end_date)

    if df is None or df.empty:
        logger.info(f"No data available for {symbol}")
        return df, settings

    # Step 2: enforce RTH+resample+prune using config
    df = _apply_rth_resample_prune(df, settings)

    # Final logging (redundant with helper, but keeps the main flow informative)
    if not df.empty:
        logger.info(f"Final OHLCV shape after cleaning: {df.shape}")
    else:
        logger.info("Final OHLCV is empty after cleaning")

    return df, settings


class ContinuousIntradayRLEnv(IntradayRLEnv):
    """
    Subclass of IntradayRLEnv that disables end-of-day termination so the episode
    spans the full data range. All other behavior is unchanged.
    """
    def _eod(self, ts):
        return False

def plot_results(results: dict, symbol: str) -> None:
    """
    Save a results figure:
      - If equity_series is present in episodes, plot the equity curve(s).
      - Otherwise, plot per-episode returns as bars.
      - Always annotate summary metrics.

    Saves to: backtest_results_{symbol}.png
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    episodes = results.get("episodes", []) or []
    avg_reward = results.get("avg_reward", 0.0)
    win_rate = results.get("win_rate", 0.0)
    total_return = results.get("total_return", 0.0)
    total_steps = results.get("total_steps", 0)

    # Determine if we have any equity series
    has_equity = any(isinstance(ep.get("equity_series", None), (list, tuple, np.ndarray)) and len(ep["equity_series"]) > 1
                     for ep in episodes)

    # Prepare figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    title = f"{symbol} Backtest"
    if len(episodes) == 1:
        title += f" — {len(episodes)} Episode"
    else:
        title += f" — {len(episodes)} Episodes"
    ax.set_title(title, pad=12)

    if has_equity:
        # Plot equity curves (one per episode)
        for ep in episodes:
            eq = ep.get("equity_series", None)
            if eq is None or len(eq) == 0:
                continue
            y = np.asarray(eq, dtype=float)
            x = np.arange(len(y), dtype=int)
            ax.plot(x, y, linewidth=1.2, alpha=0.85, label=f"Ep {ep.get('episode', '?')}")

        ax.set_xlabel("Step")
        ax.set_ylabel("Equity")
        ax.grid(True, linestyle="--", alpha=0.3)
        if len(episodes) > 1:
            ax.legend(loc="best", frameon=False)

    else:
        # No equity series — show per-episode return bars
        if len(episodes) == 0:
            ax.text(0.5, 0.5, "No results", ha="center", va="center", transform=ax.transAxes)
        else:
            rets = [float(ep.get("return", 0.0)) for ep in episodes]
            xs = np.arange(len(rets))
            ax.bar(xs, rets)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"Ep {ep.get('episode','?')}" for ep in episodes])
            ax.set_ylabel("Return")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Summary box (in axes coords)
    summary = (
        f"Avg Reward: {avg_reward:.4f}\n"
        f"Win Rate:   {win_rate:.1%}\n"
        f"Avg Return: {total_return:.2%}\n"
        f"Total Steps:{total_steps:,}"
    )
    ax.text(
        0.99, 0.01, summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9),
    )

    fig.tight_layout()

    out_name = f"backtest_results_{symbol}.png"
    try:
        fig.savefig(out_name, dpi=140)
        logger.info(f"Results plot saved as {out_name}")
    except Exception as e:
        logger.warning(f"Failed to save plot {out_name}: {e}")

    # In headless environments, plt.show() will warn; keep it but ignore warning.
    try:
        plt.show()
    except Exception:
        pass
    finally:
        plt.close(fig)


def main():
    """Main example function."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="RL Backtest with Polygon Data")
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock symbol')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-06-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run (episodic mode)')
    parser.add_argument('--plot', action='store_true', help='Generate result plots')
    parser.add_argument('--model-path', type=str, help='Path to the trained model (Stable-Baselines3)')
    parser.add_argument('--continuous', action='store_true',
                        help='Run a single continuous backtest over the entire date range')
    args = parser.parse_args()

    print(f"RL Backtest Example - {args.symbol}")
    print("=" * 50)
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Mode: {'Continuous backtest over entire date range' if args.continuous else f'Episodic ({args.episodes} episodes)'}")
    print()

    try:
        # ---------------------------
        # 1) Load & Clean OHLCV data
        # ---------------------------
        print("Loading data...")
        ohlcv_data, settings = get_clean_ohlcv(args.symbol, args.start_date, args.end_date)
        if ohlcv_data is None or ohlcv_data.empty:
            print(f"No data available for {args.symbol}")
            return

        print(f"Loaded {len(ohlcv_data):,} records")
        print(f"Date range: {ohlcv_data.index.min()} to {ohlcv_data.index.max()}")
        print()

        # ---------------------------
        # 2) Feature Engineering
        # ---------------------------
        print("Creating features...")
        pipeline = create_feature_pipeline()
        features = pipeline.fit_transform(ohlcv_data)
        print(f"Created {len(features.columns)} features")
        print()

        # --------------------------------
        # 3) Initialize RL Trading Environment
        # --------------------------------
        print("Initializing RL environment...")
        EnvClass = ContinuousIntradayRLEnv if args.continuous else IntradayRLEnv

        # If your YAML has env overrides, you can fetch here (optional)
        # pv = 1.0 for SPY; keep consistent with training
        try:
            pv = float(settings.get("env", "point_value", default=1.0))
        except Exception:
            pv = 1.0

        env = EnvClass(
            ohlcv=ohlcv_data,
            features=features,
            cash=100000.0,
            point_value=pv
        )
        print("Environment initialized successfully")
        print(f"Usable bars in env: {len(ohlcv_data)}")
        print(f"First/last: {ohlcv_data.index.min()} {ohlcv_data.index.max()}")
        print(f"Env class: {EnvClass.__name__}")
        print()

        # ---------------------------
        # 4) Load Trained Model
        # ---------------------------
        model = None
        if args.model_path:
            print(f"Loading model from {args.model_path}")
            from stable_baselines3 import PPO
            model = PPO.load(args.model_path)

        if model is None:
            raise ValueError("A trained model must be provided via --model-path")

        # ---------------------------
        # 5) Run Backtest
        # ---------------------------
        print("Running backtest simulation...")
        if args.continuous:
            results = run_continuous_backtest(env, model)
        else:
            results = run_backtest_simulation(env, model, args.episodes)
        print()

        # ---------------------------
        # 6) Print Results
        # ---------------------------
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Episodes: {len(results['episodes'])}")
        print(f"Average Reward: {results['avg_reward']:.4f}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Average Return: {results['total_return']:.2%}")
        print(f"Total Steps: {results['total_steps']}")
        print()

        print("Episode Details:")
        for ep in results['episodes']:
            print(f"Episode {ep['episode']:2d}: "
                  f"Reward={ep['reward']:8.2f}, "
                  f"Steps={ep['steps']:6d}, "
                  f"Return={ep['return']:7.2%}, "
                  f"Final Equity=${ep['final_equity']:9.0f}")

        # ---------------------------
        # 7) Plots
        # ---------------------------
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
