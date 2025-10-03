from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


@dataclass
class BacktestResult:
    trades: pd.DataFrame                      # required
    equity: pd.DataFrame                      # must contain 'equity' indexed by ts
    steps: Optional[pd.DataFrame] = None      # must have ['action','pos','price', maybe 'ofi_proxy'/'signed_vol_delta']
    metrics: Optional[Dict[str, Any]] = None  # summary.json payload
    feature_names: Optional[list[str]] = None # for features_used.txt
    baselines: Optional[Dict[str, Dict[str, float]]] = None
    reward_breakdown: Optional[pd.DataFrame] = None  # per-episode reward component stats


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _coerce_float(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def write_backtest_artifacts(base_dir: Path, ticker: str, res: BacktestResult) -> None:
    """
    Best-effort writer. Writes ALL the things or a clear FAIL.txt if something goes wrong.
    Never silently skips. Costs are reconciled from trades.
    """
    tdir = base_dir / ticker
    _ensure_dir(tdir)

    failures: list[str] = []

    # 1) trades.csv (required)
    try:
        trades = res.trades.copy()
        cost_cols = [
            "commission_cost",
            "spread_cost",
            "slippage_cost",
            "impact_cost",
            "total_cost",
            "total_cost_est",
        ]
        for c in cost_cols:
            if c not in trades.columns:
                trades[c] = 0.0
        trades["total_cost_final"] = trades[["total_cost", "total_cost_est"]].max(axis=1)
        # prefer gross_pnl if present; else assume pnl already net
        if "gross_pnl" in trades.columns:
            trades["net_pnl"] = pd.to_numeric(trades["gross_pnl"], errors="coerce").fillna(0.0) - trades["total_cost_final"]
        else:
            pnl = trades.get("pnl", 0.0)
            trades["net_pnl"] = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
        _coerce_float(trades).to_csv(tdir / "trades.csv", index=False)
    except Exception as e:
        failures.append(f"trades.csv write failed: {e!r}")

    # 2) portfolio_history.csv
    try:
        eq = res.equity.copy()
        if "equity" not in eq.columns:
            raise ValueError("equity df missing 'equity' column")
        eq.to_csv(tdir / "portfolio_history.csv")
    except Exception as e:
        failures.append(f"portfolio_history.csv write failed: {e!r}")

    # 2b) reward_breakdown.csv (optional)
    try:
        if res.reward_breakdown is not None and not res.reward_breakdown.empty:
            res.reward_breakdown.to_csv(tdir / "reward_breakdown.csv", index=False)
    except Exception as e:
        failures.append(f"reward_breakdown.csv write failed: {e!r}")

    # 3) steps.parquet: always write or fail loudly
    try:
        steps = res.steps
        if steps is None or len(steps) == 0:
            raise ValueError("steps empty or None")
        must = ["action", "pos", "price"]
        missing = [m for m in must if m not in steps.columns]
        if missing:
            raise ValueError(f"steps missing columns: {missing}")
        steps = steps.copy()
        steps.index.name = "ts"
        # enforce dtypes
        steps["action"] = pd.to_numeric(steps["action"], errors="coerce").fillna(0).astype("int8")
        steps["pos"] = pd.to_numeric(steps["pos"], errors="coerce").fillna(0).astype("int8")
        steps["price"] = pd.to_numeric(steps["price"], errors="coerce").astype("float32")
        steps.to_parquet(tdir / "steps.parquet", engine="pyarrow")
    except Exception as e:
        failures.append(f"steps.parquet write failed: {e!r}")

    # 4) diagnostics.csv computed from steps if available
    try:
        dpath = tdir / "steps.parquet"
        if dpath.exists():
            s = pd.read_parquet(dpath)
            a = pd.to_numeric(s.get("action", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int).to_numpy()
            vals, cnts = np.unique(a, return_counts=True)
            probs = cnts / cnts.sum() if cnts.sum() > 0 else np.array([1.0])
            action_entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
            flow_col = "ofi_proxy" if "ofi_proxy" in s.columns else ("signed_vol_delta" if "signed_vol_delta" in s.columns else None)
            if flow_col and s[flow_col].notna().sum() > 50:
                corr_action_flow = float(np.corrcoef(np.sign(a), np.sign(pd.to_numeric(s[flow_col], errors="coerce").to_numpy()))[0, 1])
            else:
                corr_action_flow = None
            pd.DataFrame(
                {"metric": ["action_entropy", "corr_action_flow"], "value": [action_entropy, corr_action_flow]}
            ).to_csv(tdir / "diagnostics.csv", index=False)
        else:
            failures.append("diagnostics skipped: steps.parquet missing")
    except Exception as e:
        failures.append(f"diagnostics.csv write failed: {e!r}")

    # 5) summary.json (costs reconciled to trades)
    try:
        metrics: Dict[str, Any] = dict(res.metrics or {})
        if (tdir / "trades.csv").exists():
            t = pd.read_csv(tdir / "trades.csv")
            costs_sum = float(pd.to_numeric(t.get("total_cost_final", 0.0), errors="coerce").fillna(0.0).sum())
            metrics["tx_costs_total"] = costs_sum
        # deterministic parity flag if not provided
        for k in ["long_steps", "short_steps", "long_trades", "short_trades"]:
            metrics.setdefault(k, 0)
        if (metrics["long_steps"] > 0 or metrics["long_trades"] > 0) and (metrics["short_steps"] > 0 or metrics["short_trades"] > 0):
            metrics.setdefault("parity_flag", "BOTH")
        elif metrics["long_steps"] > 0 or metrics["long_trades"] > 0:
            metrics.setdefault("parity_flag", "ONLY_LONG")
        elif metrics["short_steps"] > 0 or metrics["short_trades"] > 0:
            metrics.setdefault("parity_flag", "ONLY_SHORT")
        else:
            metrics.setdefault("parity_flag", "NONE")
        if res.baselines:
            metrics["baselines"] = res.baselines
        (tdir / "summary.json").write_text(json.dumps(metrics, indent=2))
    except Exception as e:
        failures.append(f"summary.json write failed: {e!r}")

    # 6) features_used.txt at window root if provided
    try:
        if res.feature_names:
            (base_dir / "features_used.txt").write_text("\n".join(res.feature_names))
    except Exception as e:
        failures.append(f"features_used.txt write failed: {e!r}")

    # 7) FAIL marker
    if failures:
        (tdir / "FAIL.txt").write_text("\n".join(failures))
