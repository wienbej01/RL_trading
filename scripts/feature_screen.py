#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _resolve_exec_costs(cfg: Dict[str, Any]) -> Dict[str, float]:
    ex = (cfg.get("execution", {}) if isinstance(cfg, dict) else {}) or {}
    out = {
        "tick_value": float(ex.get("tick_value", 0.01)),
        "spread_ticks": float(ex.get("spread_ticks", 1)),
        "impact_bps": float(ex.get("impact_bps", 0.0)),
        "slippage_bps": float(ex.get("slippage_bps", 0.0)),
        "commission_per_contract": float(ex.get("commission_per_contract", 0.0)),
    }
    return out


def _build_dataset(cfg: Dict[str, Any], horizon_min: int) -> Tuple["np.ndarray", "np.ndarray", List[str], "np.ndarray"]:
    """
    Use existing loader + feature pipeline to build X (features), y (forward return net of costs), feature_names, and timestamps.
    """
    from src.utils.config_loader import load_config  # type: ignore
    from src.data.data_loader import UnifiedDataLoader  # type: ignore
    from src.features.pipeline import FeaturePipeline  # type: ignore
    import pandas as pd  # type: ignore

    settings = load_config("configs/settings.yaml") if not cfg else cfg

    # Resolve tickers and dates heuristically from config
    tickers = []
    try:
        # Prefer pipeline_config in results; otherwise fall back to default universe
        tickers = settings.get("data", {}).get("multiticker", {}).get("universe", []) or []
    except Exception:
        tickers = []
    # If not specified, keep it simple: use a small safe set or pull from CLI env var
    if not tickers:
        tickers = ["SPY"]

    # Dates — try rl pipeline defaults; else reasonable defaults
    train_start = settings.get("data", {}).get("start_date", "2024-04-01")
    train_end = settings.get("data", {}).get("end_date", "2024-08-15")

    loader = UnifiedDataLoader(config_path="configs/settings.yaml")
    frames = []
    for t in tickers:
        try:
            df = loader.load_ohlcv(symbol=t, start=pd.Timestamp(train_start), end=pd.Timestamp(train_end))
            if df.empty:
                continue
            df["ticker"] = t
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("No data available for screening.")
    ohlcv = pd.concat(frames, axis=0)

    pipe = FeaturePipeline(settings.get("features", {}))
    feats = pipe.fit_transform(ohlcv)

    # Build forward return net of costs per ticker
    # y = (close[t+h] - close[t]) / close[t] - costs (in return fraction)
    costs = _resolve_exec_costs(settings)
    def _net_ret(group: pd.DataFrame) -> pd.Series:
        close = group["close"] if "close" in group.columns else ohlcv.loc[group.index, "close"]
        fwd = close.shift(-horizon_min)
        raw = (fwd - close) / close
        price = close.replace(0, np.nan)
        cost_frac = (
            (costs["spread_ticks"] * costs["tick_value"]) / price
            + (costs["slippage_bps"] + costs["impact_bps"]) / 10000.0
            + (costs["commission_per_contract"]) / (price)
        )
        return (raw - cost_frac.fillna(0.0)).astype(float)

    # We need close in feats index: join from ohlcv
    if "close" not in feats.columns:
        try:
            close_align = ohlcv["close"].reindex(feats.index)
            feats = feats.copy()
            feats["close"] = close_align
        except Exception:
            pass
    by_t = feats.copy()
    if "ticker" not in by_t.columns:
        # derive ticker from ohlcv if missing
        try:
            feats = feats.join(ohlcv[["ticker"]], how="left")
        except Exception:
            feats["ticker"] = "UNKNOWN"
    y = by_t.groupby("ticker", group_keys=False).apply(_net_ret)

    # Final X: drop non-numeric and labels
    X = feats.select_dtypes(include=[np.number]).copy()
    if "close" in X.columns:
        X = X.drop(columns=["close"], errors="ignore")
    feature_names = list(X.columns)
    ts_idx = feats.index.values
    return X.to_numpy(dtype=float), y.reindex(X.index).to_numpy(dtype=float), feature_names, ts_idx


def _ridge(alpha: float, seed: int):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=float(alpha), random_state=int(seed))


def _rf(n: int, depth: int, seed: int):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=int(n), max_depth=int(depth), n_jobs=-1, random_state=int(seed))


def _score_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).sum()
    den = (y_true - y_true.mean()) ** 2
    den2 = (y_pred - y_pred.mean()) ** 2
    d = float(np.sqrt(den.sum() * den2.sum()) + 1e-12)
    return float(num / d) if d > 0 else 0.0


def _perm_importance_block(model, X_test: np.ndarray, y_test: np.ndarray, block: int = 100, metric=_score_r2) -> np.ndarray:
    rng = np.random.RandomState(123)
    base = float(metric(y_test, model.predict(X_test)))
    n_feat = X_test.shape[1]
    imps = np.zeros(n_feat, dtype=float)
    n = X_test.shape[0]
    # build contiguous blocks
    starts = list(range(0, n, max(1, block)))
    for j in range(n_feat):
        Xp = X_test.copy()
        for s in starts:
            e = min(n, s + block)
            rng.shuffle(Xp[s:e, j])
        sc = float(metric(y_test, model.predict(Xp)))
        imps[j] = base - sc
    return imps


def main() -> int:
    ap = argparse.ArgumentParser(description="Leak‑proof feature screening")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--horizon-min", type=int, default=5)
    ap.add_argument("--embargo-min", type=int, default=15)
    ap.add_argument("--n_slices", type=int, default=6)
    ap.add_argument("--rf-n", type=int, default=200)
    ap.add_argument("--rf-depth", type=int, default=7)
    ap.add_argument("--ridge-alpha", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))

    # Build dataset
    X, y, feature_names, ts_idx = _build_dataset(cfg, args.horizon_min)

    # CV splitter
    from src.utils.wf_cv import PurgedExpandingSplit  # type: ignore

    splitter = PurgedExpandingSplit(n_slices=int(args.n_slices), embargo_minutes=int(args.embargo_min))

    # Models
    mdl_specs = {
        "ridge": _ridge(args.ridge_alpha, args.seed),
        "rf": _rf(args.rf_n, args.rf_depth, args.seed),
    }

    out_root = Path("results/features") / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Save screen config
    scfg = {
        "config": args.config,
        "horizon_min": args.horizon_min,
        "embargo_min": args.embargo_min,
        "n_slices": args.n_slices,
        "rf": {"n": args.rf_n, "depth": args.rf_depth},
        "ridge": {"alpha": args.ridge_alpha},
        "seed": args.seed,
        "n_features": len(feature_names),
    }
    (out_root / "screen_config.json").write_text(json.dumps(scfg, indent=2))

    # Fit per slice
    per_slice_scores: Dict[str, List[np.ndarray]] = defaultdict(list)
    for i, (tr_idx, te_idx) in enumerate(splitter.split(pd.DatetimeIndex(ts_idx))):
        if len(tr_idx) < 100 or len(te_idx) < 100:
            continue
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        for name, mdl in mdl_specs.items():
            try:
                mdl.fit(Xtr, ytr)
                imps = _perm_importance_block(mdl, Xte, yte, block=100)
            except Exception:
                imps = np.zeros(len(feature_names), dtype=float)
            per_slice_scores[name].append(imps)
            # Write per-slice importance parquet
            try:
                import pandas as pd  # type: ignore

                df_imp = pd.DataFrame({"feature": feature_names, "importance": imps})
                df_imp.to_parquet(out_root / f"importance_slice{i+1}_{name}.parquet")
            except Exception:
                pass

    # Consensus aggregation
    try:
        import pandas as pd  # type: ignore

        stacks: List[pd.DataFrame] = []
        for name, mats in per_slice_scores.items():
            if not mats:
                continue
            mat = np.vstack(mats)
            mean_imp = mat.mean(axis=0)
            stacks.append(pd.DataFrame({"feature": feature_names, f"imp_{name}": mean_imp}))
        if stacks:
            df = stacks[0]
            for s in stacks[1:]:
                df = df.merge(s, on="feature", how="outer")
            # Consensus score: mean of z-scored importances across models
            for c in df.columns:
                if c.startswith("imp_"):
                    v = df[c].to_numpy()
                    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
                    df[c.replace("imp_", "z_")] = z
            zcols = [c for c in df.columns if c.startswith("z_")]
            df["consensus"] = df[zcols].mean(axis=1)
            df.sort_values("consensus", ascending=False, inplace=True)
            df.to_parquet(out_root / "consensus_importance.parquet")
            # Summary MD
            top = df.head(50)[["feature", "consensus"]]
            lines = ["# Feature Screening Summary", "", f"Run: {args.run_name}", "", "## Top 50 (consensus)"]
            for _, row in top.iterrows():
                lines.append(f"- {row['feature']}: {row['consensus']:.3f}")
            (out_root / "summary.md").write_text("\n".join(lines))
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

