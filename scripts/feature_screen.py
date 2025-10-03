#!/usr/bin/env python3
from __future__ import annotations

"""CPCV feature screening runner.

Provides a CLI and a callable API `run_cpcv_feature_screen` that compute a
consensus importance across Ridge and RandomForest models using a
Combinatorial Purged Cross‑Validation scheme.

Outputs under results/features/<run_name>:
  - consensus_importance.parquet (feature, consensus, ridge, rf, support_count)
  - curated_topN.txt (top N by consensus)
  - features_used.txt (curated plus forced microstructure, if available)
  - corr_heatmap.png (corr matrix for topN features)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src.utils.wf_cv import CombinatorialPurgedKFold
from src.features.packs import MICROSTRUCTURE_MIN


log = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    try:
        import random as _rd
        import torch as _torch
        _rd.seed(seed)
        np.random.seed(seed)
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        np.random.seed(seed)


def _zscore(v: np.ndarray) -> np.ndarray:
    m = float(np.mean(v))
    s = float(np.std(v))
    if not np.isfinite(s) or s == 0.0:
        return np.zeros_like(v, dtype=float)
    return (v - m) / s


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cpcv_feature_screen(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    *,
    run_name: str,
    out_root: Path | str = "results/features",
    n_slices: int = 10,
    n_test_groups: int = 2,
    embargo_bars: int = 15,
    rf_n: int = 200,
    rf_depth: int = 7,
    ridge_alpha: float = 5.0,
    top_n: int = 40,
    seed: int = 123,
) -> Path:
    """Screen features with CPCV and save artifacts under out_root/run_name.

    Returns the output directory path.
    """
    _set_seeds(seed)
    out_dir = Path(out_root) / run_name
    _ensure_dir(out_dir)

    # Align inputs and keep numeric columns; drop label NaNs
    X = X.copy()
    if 'ticker' in X.columns:
        X = X.drop(columns=['ticker'])
    X = X.select_dtypes(include=[np.number])
    y = pd.to_numeric(y, errors='coerce')
    idx = X.index.intersection(y.index)
    X = X.loc[idx]
    y = y.loc[idx]
    mask = y.notna().values
    X = X.loc[mask]
    y = y.loc[mask]
    groups = np.asarray(groups)[mask]
    feats = list(X.columns)
    n_features = len(feats)
    if n_features == 0:
        raise ValueError("No numeric features provided to screen")

    log.info("Screen run=%s | rows=%d cols=%d | slices=%d", run_name, len(X), n_features, n_slices)
    # CPCV splitter over integer group ids
    # Effective number of unique groups after masking
    uniq_g = int(np.unique(groups).size)
    eff_splits = int(min(max(2, n_slices), uniq_g))
    eff_test_groups = int(min(max(1, n_test_groups), max(1, eff_splits - 1)))
    splitter = CombinatorialPurgedKFold(n_splits=eff_splits, n_test_groups=eff_test_groups, embargo_bars=embargo_bars)

    ridge_imps: List[np.ndarray] = []
    rf_imps: List[np.ndarray] = []
    support_counts = np.zeros(n_features, dtype=int)

    fold_ct = 0
    for tr_idx, te_idx in splitter.split(groups):
        fold_ct += 1
        Xtr = X.iloc[tr_idx]
        ytr = y.iloc[tr_idx]
        # Drop rows with any NaN/Inf in training fold to keep linear models happy
        Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
        ytr = pd.to_numeric(ytr, errors='coerce')
        good_mask = Xtr.notna().all(axis=1) & ytr.notna()
        if not bool(np.any(good_mask.values)):
            # Skip this fold if no valid rows
            continue
        Xtr = Xtr.loc[good_mask]
        ytr = ytr.loc[good_mask]
        # Standardize for Ridge
        ss = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = ss.fit_transform(Xtr)
        # Ridge (regression)
        r = Ridge(alpha=float(ridge_alpha), random_state=seed)
        r.fit(Xtr_s, ytr.values)
        coefs = np.abs(np.asarray(r.coef_, dtype=float)).ravel()
        ridge_imps.append(coefs)
        # RF (regression)
        rf = RandomForestRegressor(n_estimators=int(rf_n), max_depth=int(rf_depth), random_state=seed, n_jobs=-1)
        rf.fit(Xtr.values, ytr.values)
        rf_imp = np.asarray(rf.feature_importances_, dtype=float)
        rf_imps.append(rf_imp)
        # Support: any positive importance on either model counts as support for this fold
        support_counts += ((coefs > 0) | (rf_imp > 0)).astype(int)

    ridge_arr = np.vstack(ridge_imps) if ridge_imps else np.zeros((1, n_features))
    rf_arr = np.vstack(rf_imps) if rf_imps else np.zeros((1, n_features))

    # Model-wise normalized means (sum to 1 across features)
    ridge_mean = ridge_arr.mean(axis=0)
    rf_mean = rf_arr.mean(axis=0)
    ridge_norm = ridge_mean / (ridge_mean.sum() + 1e-12)
    rf_norm = rf_mean / (rf_mean.sum() + 1e-12)

    # Consensus: mean z-score across models and folds per feature
    z_all: List[np.ndarray] = []
    for row in ridge_arr:
        z_all.append(_zscore(row))
    for row in rf_arr:
        z_all.append(_zscore(row))
    if not z_all:
        consensus = np.zeros(n_features)
    else:
        consensus = np.vstack(z_all).mean(axis=0)

    df = pd.DataFrame({
        'feature': feats,
        'consensus': consensus,
        'ridge': ridge_norm,
        'rf': rf_norm,
        'support_count': support_counts,
    }).sort_values('consensus', ascending=False)

    df.to_parquet(out_dir / 'consensus_importance.parquet', engine='pyarrow', index=False)

    top_feats = df['feature'].head(int(top_n)).tolist()
    # Ensure microstructure presence if available
    try:
        # append available MICROSTRUCTURE_MIN if missing
        for m in MICROSTRUCTURE_MIN:
            for c in feats:
                if c == m or c.startswith(m) or c.endswith(m):
                    if c not in top_feats:
                        top_feats.append(c)
                    break
    except Exception:
        pass
    (out_dir / 'curated_topN.txt').write_text("\n".join(top_feats) + "\n")
    (out_dir / 'features_used.txt').write_text("\n".join(top_feats) + "\n")

    # Corr heatmap for topN (guard against zero-variance columns)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # noqa: F401
        sel = X[top_feats].copy()
        var = sel.var(numeric_only=True)
        keep = var[var > 0].index
        sel = sel[keep]
        c = sel.corr().clip(-1, 1).fillna(0.0)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(c.values, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(c.columns)))
        ax.set_xticklabels(list(c.columns), rotation=90, fontsize=6)
        ax.set_yticks(range(len(c.index)))
        ax.set_yticklabels(list(c.index), fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.savefig(out_dir / 'corr_heatmap.png', dpi=150)
        plt.close(fig)
    except Exception:
        pass

    print(f"[screen] outputs -> {out_dir}")
    return out_dir


def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='CPCV Feature Screening')
    ap.add_argument('--config', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--tickers', nargs='+', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--horizon-min', type=int, default=10)
    ap.add_argument('--embargo-min', type=int, default=15)
    ap.add_argument('--n_slices', type=int, default=10)
    ap.add_argument('--rf-n', type=int, default=200)
    ap.add_argument('--rf-depth', type=int, default=7)
    ap.add_argument('--ridge-alpha', type=float, default=5.0)
    ap.add_argument('--topN', type=int, default=40)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--label-kind', choices=['triple_barrier', 'signed_ret'], default='signed_ret')
    return ap.parse_args()


def _cli_main() -> int:
    args = _parse_cli()
    _set_seeds(int(args.seed))
    try:
        from src.data.data_loader import UnifiedDataLoader
        from src.features.pipeline import FeaturePipeline
        from src.utils.config_loader import load_config
    except Exception as e:
        print(f"[screen] Failed to import project modules: {e}")
        return 2

    cfg = load_config(args.config)
    loader = UnifiedDataLoader(config_path=args.config)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    frames: List[pd.DataFrame] = []
    for t in args.tickers:
        try:
            df = loader.load_ohlcv(t, start, end)
            if not df.empty:
                df['ticker'] = t
                frames.append(df)
        except Exception:
            continue
    if not frames:
        print("[screen] No data loaded for tickers in range")
        return 3
    ohlcv = pd.concat(frames, axis=0)
    pipe = FeaturePipeline(cfg.get('features', {}))
    feats = pipe.fit_transform(ohlcv)
    try:
        pipe.write_reports(out_dir, prefix=args.run_name, features=feats)
    except Exception as exc:
        print(f"[screen] feature report export failed: {exc}")
    # Label: signed_ret forward horizon
    by_t = []
    for t in sorted(pd.unique(ohlcv['ticker'])):
        df_t = ohlcv[ohlcv['ticker'] == t]
        y_t = df_t['close'].pct_change(periods=int(args.horizon_min)).shift(-int(args.horizon_min))
        y_t.name = 'y'
        y_t = y_t.reindex(feats.index)
        y_t = y_t.where(feats.get('ticker', pd.Series(index=feats.index)).eq(t))
        by_t.append(y_t)
    y = pd.concat(by_t, axis=0).sort_index()
    # Groups: contiguous slices
    idx = feats.index
    n = len(idx)
    n_slices = max(2, int(args.n_slices))
    group_ids = np.floor(np.linspace(0, n_slices - 1, num=n)).astype(int)
    out_dir = run_cpcv_feature_screen(
        feats, y, group_ids,
        run_name=args.run_name,
        n_slices=n_slices,
        n_test_groups=2,
        embargo_bars=int(args.embargo_min),
        rf_n=int(args.rf_n),
        rf_depth=int(args.rf_depth),
        ridge_alpha=float(args.ridge_alpha),
        top_n=int(args.topN),
        seed=int(args.seed),
    )
    print(json.dumps({'ok': True, 'out_dir': str(out_dir)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli_main())
