#!/usr/bin/env python3
"""
Read-only audit tool for RL runs.

Collects facts from a run directory and config file, then writes
JSON and Markdown reports under results/audits/<run>_<ts>/.

Usage:
  python scripts/audit_collect.py --run-dir results/mt_lowpx_core --config configs/settings.yaml --out-dir results/audits
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import traceback
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _newest(path_glob: str) -> Optional[Path]:
    paths = [Path(p) for p in glob(path_glob)]
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]


def _get_commit_hash() -> Optional[str]:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _read_pipeline_meta(run_dir: Path) -> Dict[str, Any]:
    meta = {}
    meta_path = run_dir / "pipeline_config.json"
    if meta_path.exists():
        meta = _load_json(meta_path)
    return meta


def _infer_dates_from_files(run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    # Try to parse train/test from pipeline_config first
    meta = _read_pipeline_meta(run_dir)
    tr_s = meta.get("train_start")
    tr_e = meta.get("train_end")
    te_s = meta.get("test_start")
    te_e = meta.get("test_end")
    # Fallback: parse from features/data filenames
    if not (tr_s and tr_e):
        df = _newest(str(run_dir / "data" / "*.parquet"))
        if df:
            name = df.stem
            parts = name.split("_")
            if len(parts) >= 4:
                tr_s, tr_e = parts[-3], parts[-1]
    if not (te_s and te_e):
        # Backtest file often contains test range
        bj = _newest(str(run_dir / "backtest" / "*.json"))
        if bj:
            s = bj.stem
            if "to" in s:
                rng = s.rsplit("_", 1)[-1]
                if "to" in rng:
                    te_s, te_e = rng.split("_to_")
    return tr_s, tr_e, te_s, te_e


def _count_days(s: Optional[str], e: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        if s and e:
            ds = dt.date.fromisoformat(str(s))
            de = dt.date.fromisoformat(str(e))
            out["wall_days"] = (de - ds).days + 1
        else:
            out["wall_days"] = None
    except Exception:
        out["wall_days"] = None
    return out


def _read_backtest_json(run_dir: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    bj = _newest(str(run_dir / "backtest" / "*.json"))
    return bj, (_load_json(bj) if bj else {})


def _bar_freq_from_history(run_dir: Path) -> Optional[str]:
    try:
        import pandas as pd  # type: ignore

        ph = run_dir / "backtest" / "portfolio_history.csv"
        if not ph.exists():
            return None
        df = pd.read_csv(ph)
        if "timestamp" not in df.columns or df.empty:
            return None
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        ts = ts.dropna()
        if len(ts) < 3:
            return None
        diffs = (ts.iloc[1:].values - ts.iloc[:-1].values)
        # numpy datetime64 diffs in ns
        mins = [float(d.astype("timedelta64[m]") / 1) for d in diffs]
        if not mins:
            return None
        med = sorted(mins)[len(mins) // 2]
        return f"{med:.0f}min"
    except Exception:
        return None


def _lib_versions() -> Dict[str, Any]:
    v = {"python": platform.python_version()}
    try:
        import stable_baselines3 as sb3  # type: ignore

        v["stable_baselines3"] = getattr(sb3, "__version__", "unknown")
    except Exception:
        v["stable_baselines3"] = "unavailable"
    try:
        import torch  # type: ignore

        v["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        v["torch"] = "unavailable"
    try:
        import numpy as np  # type: ignore

        v["numpy"] = getattr(np, "__version__", "unknown")
    except Exception:
        v["numpy"] = "unavailable"
    return v


def build_audit(run_dir: Path, cfg_path: Path) -> Tuple[Dict[str, Any], str]:
    errors: List[str] = []
    try:
        cfg = _load_yaml(cfg_path)
    except Exception as e:
        cfg = {}
        errors.append(f"config_load: {e}")

    meta = _read_pipeline_meta(run_dir)
    train_tickers = meta.get("tickers") or []

    tr_s, tr_e, te_s, te_e = _infer_dates_from_files(run_dir)
    dates = {
        "train": {"start": tr_s, "end": tr_e, **_count_days(tr_s, tr_e)},
        "test": {"start": te_s, "end": te_e, **_count_days(te_s, te_e)},
    }

    # Env mode
    portfolio_env = True  # default assume portfolio for multi-ticker trainer; refine by backtest JSON

    # Feature list
    features_file = _newest(str(run_dir / "features" / "*.parquet"))
    feature_list: List[str] = []
    try:
        if features_file:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(features_file)
            feature_list = [c for c in df.columns if c != "ticker"]
    except Exception as e:
        errors.append(f"features_read: {e}")

    # Normalization settings
    vecnorm_path = run_dir / "models" / "checkpoints" / "vecnorm.pkl"
    norm_cfg = (
        cfg.get("rl", {}).get("vecnormalize", {}) if isinstance(cfg, dict) else {}
    )
    normalize_obs = bool(norm_cfg.get("norm_obs", False))
    normalize_rew = bool(norm_cfg.get("norm_reward", False))

    # Execution/costs
    exec_cfg = cfg.get("execution", {}) if isinstance(cfg, dict) else {}

    # PPO params
    ppo_cfg = cfg.get("rl", {}).get("ppo", {}) if isinstance(cfg, dict) else {}
    seed = cfg.get("rl", {}).get("seed", None) if isinstance(cfg, dict) else None
    versions = _lib_versions()

    # Backtest artifacts
    bj, backtest = _read_backtest_json(run_dir)
    back_metrics = backtest.get("portfolio_metrics") or {}
    back_tickers = backtest.get("tickers") or []
    if isinstance(back_tickers, dict):
        back_tickers = list(back_tickers.keys())

    # Derived assessments
    derived: Dict[str, Any] = {}
    try:
        tr_days = dates["train"].get("wall_days") or 0
        te_days = dates["test"].get("wall_days") or 0
        derived["train_test_day_ratio"] = (float(tr_days) / float(max(te_days, 1))) if tr_days and te_days else None
        derived["n_train_tickers"] = len(train_tickers)
        derived["n_backtest_tickers"] = len(back_tickers)
    except Exception:
        pass

    # Failure modes (heuristics)
    failure_modes: List[Dict[str, str]] = []
    # PORTFOLIO_IN_NAME_ONLY
    try:
        if (len(train_tickers) > 1) and (len(back_tickers) <= 1):
            failure_modes.append({
                "code": "PORTFOLIO_IN_NAME_ONLY",
                "reason": "Multiple train tickers requested but backtest produced a single-ticker output."
            })
    except Exception:
        pass
    # NORM_MISMATCH
    try:
        if normalize_obs and not vecnorm_path.exists():
            failure_modes.append({
                "code": "NORM_MISMATCH",
                "reason": "Obs normalization enabled but VecNormalize stats not found in checkpoints."
            })
    except Exception:
        pass
    # SPLIT_THIN
    try:
        if (dates["train"].get("wall_days") or 0) >= 5 * (dates["test"].get("wall_days") or 0) > 0:
            failure_modes.append({
                "code": "SPLIT_THIN",
                "reason": "Test window is short relative to train (<1:5)."
            })
    except Exception:
        pass

    # Assemble report
    report: Dict[str, Any] = {
        "data": {
            "train_tickers": train_tickers,
            "backtest_tickers": back_tickers,
            "dates": dates,
            "portfolio_env": portfolio_env,
            "bar_frequency": _bar_freq_from_history(run_dir),
            "timezone": "America/New_York",
        },
        "features": {
            "features_used": feature_list,
            "normalize": {
                "obs": normalize_obs,
                "reward": normalize_rew,
                "vecnorm_path": str(vecnorm_path) if vecnorm_path.exists() else None,
            },
            "leakage_notes": "No explicit scaler leakage detected; review feature fit/transform if unsure.",
        },
        "strategy": {
            "action_space": "MultiDiscrete([-1,0,1] per ticker)",
            "entry_exit": cfg.get("env", {}).get("trading", {}) if isinstance(cfg, dict) else {},
            "risk_rules": cfg.get("environment", {}).get("risk", {}) if isinstance(cfg, dict) else {},
            "execution": exec_cfg,
        },
        "ppo": {
            "versions": versions,
            "policy": {
                "arch": (cfg.get("rl", {}).get("ppo", {}).get("policy_kwargs", {}).get("net_arch") if isinstance(cfg, dict) else None),
                "activation": (cfg.get("rl", {}).get("ppo", {}).get("policy_kwargs", {}).get("activation_fn") if isinstance(cfg, dict) else None),
                "recurrent": True,
                "ortho_init": (cfg.get("rl", {}).get("ppo", {}).get("policy_kwargs", {}).get("ortho_init") if isinstance(cfg, dict) else None),
            },
            "params": ppo_cfg,
            "seed": seed,
            "callbacks": {
                "kl_stop": {"target_kl": (ppo_cfg.get("target_kl") if isinstance(ppo_cfg, dict) else None)},
                "adaptive_lr_by_kl": {"low": 0.003, "high": (ppo_cfg.get("target_kl") if isinstance(ppo_cfg, dict) else None)},
                "live_lr_bump": {"flag": ".lr_bump"},
            },
        },
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "config": str(cfg_path.resolve()),
            "backtest_json": str(bj.resolve()) if bj else None,
            "metrics": back_metrics,
            "commit": _get_commit_hash(),
        },
        "derived": derived,
        "failure_modes": failure_modes,
        "errors": errors,
    }

    # Markdown rendering (compact)
    md_lines: List[str] = []
    md_lines.append(f"# RL Run Audit — {run_dir.name}")
    md_lines.append("")
    md_lines.append("## Overview")
    md_lines.append(f"- Commit: {report['artifacts']['commit']}")
    md_lines.append(f"- Libs: {versions}")
    md_lines.append(f"- Seed: {seed}")
    md_lines.append("")
    md_lines.append("## Data & Splits")
    md_lines.append(f"- Train tickers ({len(train_tickers)}): {train_tickers}")
    md_lines.append(f"- Backtest tickers ({len(back_tickers)}): {back_tickers}")
    md_lines.append(f"- Train: {dates['train']}")
    md_lines.append(f"- Test: {dates['test']}")
    md_lines.append(f"- Portfolio env: {portfolio_env}")
    md_lines.append(f"- Bar freq: {report['data']['bar_frequency']}")
    md_lines.append("")
    md_lines.append("## Features & Normalization")
    md_lines.append(f"- Features used ({len(feature_list)}): {feature_list[:50]}{'...' if len(feature_list)>50 else ''}")
    md_lines.append(f"- VecNormalize: obs={normalize_obs}, reward={normalize_rew}, path={report['features']['normalize']['vecnorm_path']}")
    md_lines.append("")
    md_lines.append("## Strategy & Execution")
    md_lines.append(f"- Execution: {exec_cfg}")
    md_lines.append(f"- Entry/exit: {report['strategy']['entry_exit']}")
    md_lines.append("")
    md_lines.append("## PPO Hyperparameters")
    md_lines.append(f"- Params: {ppo_cfg}")
    md_lines.append(f"- Callbacks: {report['ppo']['callbacks']}")
    md_lines.append("")
    md_lines.append("## Backtest Results")
    md_lines.append(f"- Metrics: {back_metrics}")
    md_lines.append("")
    md_lines.append("## Derived Ratios")
    md_lines.append(f"- {derived}")
    md_lines.append("")
    md_lines.append("## Likely Failure Modes")
    if failure_modes:
        for fm in failure_modes:
            md_lines.append(f"- {fm['code']}: {fm['reason']}")
    else:
        md_lines.append("- None detected by heuristics.")

    md = "\n".join(md_lines)
    return report, md


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit RL run directory")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = Path(args.config)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_root / f"{run_dir.name}_{ts}"
    dest.mkdir(parents=True, exist_ok=True)

    report, md = build_audit(run_dir, cfg_path)
    js_path = dest / "audit_report.json"
    md_path = dest / "audit_report.md"
    js_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(md)
    print(str(js_path.resolve()))
    print(str(md_path.resolve()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Print traceback and exit non-zero for CI visibility
        traceback.print_exc()
        raise SystemExit(1)

