#!/usr/bin/env python3
"""
Out-of-Sample (OOS) evaluation across rolling windows and held-out tickers.

Runs backtests over three disjoint windows in the last ~18 months and reports
cadence, action mix, expectancy (after costs), drawdowns, and Sharpe/MAR-style metrics.

Usage (example):
  PYTHONPATH=. python scripts/oos_eval.py \
    --config configs/settings.yaml \
    --data results/multiticker_pipeline/data/multiticker_data_2020-09-01_to_2025-06-30.parquet \
    --features results/multiticker_pipeline/features/multiticker_features_2020-09-01_to_2025-06-30.parquet \
    --model results/multiticker_pipeline/models/model \
    --test-tickers AAPL MSFT NVDA \
    --output results/oos_eval/demo
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logging import get_logger
from src.rl.multiticker_trainer import MultiTickerRLTrainer


logger = get_logger(__name__)


@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp


def infer_windows(idx: pd.DatetimeIndex, n_windows: int = 3) -> List[Window]:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.sort_values().tz_convert("America/New_York")
    if idx.empty:
        return []
    last = idx.max()
    # ~18 months back, then split into n_windows equal windows
    start_all = last - pd.DateOffset(months=18)
    rng = idx[(idx >= start_all) & (idx <= last)]
    if rng.empty:
        # Fallback: last 180 calendar days
        start_all = last - pd.Timedelta(days=180)
        rng = idx[(idx >= start_all) & (idx <= last)]
    if rng.empty:
        return []
    # Split rng into n_windows contiguous chunks
    boundaries = np.linspace(0, len(rng) - 1, n_windows + 1, dtype=int)
    wins: List[Window] = []
    for i in range(n_windows):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        # Ensure non-empty and advancing
        if hi <= lo:
            continue
        wins.append(Window(start=rng[lo], end=rng[hi - 1]))
    return wins


def slice_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = (df.index >= start) & (df.index <= end)
    return df.loc[m]


def trades_per_day(total_trades: int, idx: pd.DatetimeIndex) -> float:
    if idx.empty:
        return 0.0
    n_days = len(pd.Index(pd.to_datetime(idx.date)).unique())
    return float(total_trades) / float(max(1, n_days))


def evaluate_window(
    config: Dict[str, Any],
    model_path: str,
    data_win: pd.DataFrame,
    feat_win: pd.DataFrame,
    tickers: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = MultiTickerRLTrainer(config)
    # Hint the trainer to enforce obs shape parity for portfolio env
    try:
        trainer._train_tickers = list(tickers)
    except Exception:
        pass
    # Backtest
    from sb3_contrib import RecurrentPPO
    try:
        model = RecurrentPPO.load(model_path)
    except Exception as e:
        return {"status": "error", "error": f"failed to load model: {e}"}

    try:
        summary = trainer.backtest(
            model=model,
            data=data_win,
            features=feat_win,
            output_dir=out_dir,
            allowed_tickers=tickers,
        )
    except Exception as e:
        return {"status": "error", "error": f"backtest failed: {e}"}

    # Augment with cadence and action mix if possible
    metrics = {}
    try:
        # Portfolio metrics path
        m = summary.get("portfolio_metrics", {})
        total_tr = int(m.get("total_trades", 0))
        metrics.update(m)
        metrics["trades_per_day"] = trades_per_day(total_tr, data_win.index)
        # Action mix (if present)
        long_tr = int(m.get("long_trades", 0))
        short_tr = int(m.get("short_trades", 0))
        if total_tr > 0:
            metrics["action_mix"] = {
                "long_share": long_tr / total_tr,
                "short_share": short_tr / total_tr,
                "hold_share_proxy": max(0.0, 1.0 - (long_tr + short_tr) / max(1, total_tr)),
            }
    except Exception:
        pass

    # Simple gates (illustrative; tweak per project needs)
    gates = {}
    try:
        gates = {
            "cadence_min_1": metrics.get("trades_per_day", 0.0) >= 1.0,
            "action_diversity": (metrics.get("action_mix", {}).get("long_share", 0) > 0)
                                and (metrics.get("action_mix", {}).get("short_share", 0) > 0),
            "expectancy_pos": float(metrics.get("avg_pnl", 0.0)) > 0.0,
            "max_dd_lt_20pct": float(abs(metrics.get("max_drawdown", 0.0))) < 0.2,
        }
    except Exception:
        pass

    return {
        "status": "ok",
        "metrics": metrics,
        "gates": gates,
        "summary": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Out-of-sample evaluation across windows and tickers")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True, help="Path to SB3 model to evaluate")
    ap.add_argument("--test-tickers", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Load frames
    data = pd.read_parquet(args.data)
    feats = pd.read_parquet(args.features)
    # Ensure DatetimeIndex and ticker column when present
    if "timestamp" in data.columns:
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        data = data.loc[data["timestamp"].notna()].set_index("timestamp").sort_index()
    if "timestamp" in feats.columns:
        feats = feats.copy()
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True, errors="coerce")
        feats = feats.loc[feats["timestamp"].notna()].set_index("timestamp").sort_index()

    # Build windows off combined index
    idx = data.index
    wins = infer_windows(idx, n_windows=max(1, int(args.windows)))
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    results = {"windows": []}
    # Evaluate each window and aggregate
    for i, w in enumerate(wins):
        d_w = slice_range(data, w.start, w.end)
        f_w = slice_range(feats, w.start, w.end)
        # Keep only requested tickers if present
        if "ticker" in d_w.columns:
            d_w = d_w[d_w["ticker"].isin(args.test_tickers)]
        if "ticker" in f_w.columns:
            f_w = f_w[f_w["ticker"].isin(args.test_tickers)]
        if d_w.empty or f_w.empty:
            results["windows"].append({
                "window": {"start": str(w.start), "end": str(w.end)},
                "status": "skip_empty",
            })
            continue
        win_dir = out_root / f"win_{i+1:02d}"
        res = evaluate_window(cfg, args.model, d_w, f_w, args.test_tickers, win_dir)
        res["window"] = {"start": str(w.start), "end": str(w.end)}
        results["windows"].append(res)

    # Aggregate metrics across windows where available
    def collect(k: str) -> List[float]:
        vals: List[float] = []
        for w in results["windows"]:
            m = w.get("metrics", {})
            if k in m and isinstance(m[k], (int, float)):
                vals.append(float(m[k]))
        return vals

    agg = {
        "median_sharpe": np.median(collect("sharpe_ratio")) if collect("sharpe_ratio") else None,
        "median_trades_per_day": np.median(collect("trades_per_day")) if collect("trades_per_day") else None,
        "worst_quintile_avg_pnl": (np.quantile(collect("avg_pnl"), 0.2) if collect("avg_pnl") else None),
        "max_drawdown_worst": (np.max(np.abs(collect("max_drawdown"))) if collect("max_drawdown") else None),
    }
    results["aggregate"] = agg

    with (out_root / "oos_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("OOS evaluation complete. Results at %s", out_root / "oos_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

