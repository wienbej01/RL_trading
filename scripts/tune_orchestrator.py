#!/usr/bin/env python3
"""
Lightweight tuning orchestrator that reuses the existing trainer and OOS evaluator.

For each trial, samples hyperparameters, trains a small model, evaluates OOS across
windows and held-out tickers, and ranks trials by gate pass count then median Sharpe.

Example:
  PYTHONPATH=. python scripts/tune_orchestrator.py \
    --config configs/settings.yaml \
    --data data/cache/SPY_20240101_20240701_ohlcv_1min.parquet \
    --features results/oos_eval_dryrun/spy_features.parquet \
    --output results/tuning_demo \
    --test-tickers SPY \
    --trials 4 --total-steps 5000
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.config_loader import Settings, load_config
from src.utils.logging import get_logger
from src.rl.train import TrainingConfig, train_ppo_lstm
from scripts.oos_eval import infer_windows, slice_range, evaluate_window


logger = get_logger(__name__)


@dataclass
class TrialSpec:
    learning_rate: float
    ent_coef: float
    clip_range: float
    gamma: float
    gae_lambda: float
    seed: int


def sample_trial(seed: int) -> TrialSpec:
    rnd = random.Random(seed)
    lr = 10 ** rnd.uniform(-5.5, -3.5)         # ~[3e-6, 3e-4]
    ent = 10 ** rnd.uniform(-4.0, -1.7)        # ~[1e-4, 2e-2]
    clip = rnd.uniform(0.1, 0.3)
    gamma = rnd.uniform(0.95, 0.999)
    lam = rnd.uniform(0.9, 0.98)
    return TrialSpec(learning_rate=lr, ent_coef=ent, clip_range=clip, gamma=gamma, gae_lambda=lam, seed=seed)


def run_trial(
    cfg_path: str,
    data_path: str,
    feat_path: str,
    out_dir: Path,
    trial: TrialSpec,
    total_steps: int,
    test_tickers: List[str],
    windows: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Settings (use file; training overrides via TrainingConfig)
    settings = Settings.from_yaml(cfg_path)
    # Build training config
    tc = TrainingConfig()
    tc.learning_rate = float(trial.learning_rate)
    tc.ent_coef = float(trial.ent_coef)
    tc.clip_range = float(trial.clip_range)
    tc.gamma = float(trial.gamma)
    tc.gae_lambda = float(trial.gae_lambda)
    tc.total_steps = int(total_steps)
    tc.seed = int(trial.seed)
    # Train
    model_path = str(out_dir / "model")
    try:
        train_ppo_lstm(
            settings=settings,
            data_path=data_path,
            features_path=feat_path,
            model_path=model_path,
            training_config=tc,
        )
    except Exception as e:
        return {"status": "error", "stage": "train", "error": str(e)}

    # OOS windows
    import pandas as pd
    data = pd.read_parquet(data_path)
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        data = data.loc[data["timestamp"].notna()].set_index("timestamp").sort_index()
    feats = pd.read_parquet(feat_path)
    if "timestamp" in feats.columns:
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True, errors="coerce")
        feats = feats.loc[feats["timestamp"].notna()].set_index("timestamp").sort_index()

    wins = infer_windows(data.index, n_windows=max(1, int(windows)))
    cfg = load_config(cfg_path)
    per_win: List[Dict[str, Any]] = []
    for i, w in enumerate(wins):
        d_w = slice_range(data, w.start, w.end)
        f_w = slice_range(feats, w.start, w.end)
        if "ticker" in d_w.columns:
            d_w = d_w[d_w["ticker"].isin(test_tickers)]
        if "ticker" in f_w.columns:
            f_w = f_w[f_w["ticker"].isin(test_tickers)]
        res = evaluate_window(cfg, model_path, d_w, f_w, test_tickers, out_dir / f"win_{i+1:02d}")
        per_win.append(res)

    # Aggregate quick scoring
    def metric_list(k: str) -> List[float]:
        vals: List[float] = []
        for r in per_win:
            m = r.get("metrics", {})
            if k in m and isinstance(m[k], (int, float)):
                vals.append(float(m[k]))
        return vals
    median_sharpe = (np.median(metric_list("sharpe_ratio")) if metric_list("sharpe_ratio") else None)
    median_tpd = (np.median(metric_list("trades_per_day")) if metric_list("trades_per_day") else None)
    gates_pass = sum(int(all(r.get("gates", {}).values())) for r in per_win if r.get("gates"))

    trial_json = {
        "status": "ok",
        "trial": asdict(trial),
        "model_path": model_path,
        "windows": per_win,
        "score": {
            "gates_pass": gates_pass,
            "median_sharpe": median_sharpe,
            "median_trades_per_day": median_tpd,
        },
    }
    with (out_dir / "oos.json").open("w") as f:
        json.dump(trial_json, f, indent=2)
    return trial_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Tuning orchestrator using existing trainer + OOS evaluator")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--test-tickers", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--total-steps", type=int, default=5000)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    results: List[Dict[str, Any]] = []
    for k in range(int(args.trials)):
        spec = sample_trial(seed=rng.randint(1, 10_000_000))
        tdir = out_root / f"trial_{k+1:02d}"
        logger.info("Running trial %d: %s", k + 1, spec)
        res = run_trial(
            cfg_path=args.config,
            data_path=args.data,
            feat_path=args.features,
            out_dir=tdir,
            trial=spec,
            total_steps=int(args.total_steps),
            test_tickers=list(args.test_tickers),
            windows=int(args.windows),
        )
        results.append(res)

    # Rank trials
    def rank_key(r: Dict[str, Any]):
        s = r.get("score", {})
        return (
            int(s.get("gates_pass", 0)),
            float(s.get("median_sharpe", -1e9)) if s.get("median_sharpe") is not None else -1e9,
            float(s.get("median_trades_per_day", -1e9)) if s.get("median_trades_per_day") is not None else -1e9,
        )

    ranked = sorted(results, key=rank_key, reverse=True)
    summary = {
        "trials": results,
        "best": ranked[0] if ranked else None,
    }
    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Tuning complete. Summary at %s", out_root / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

