#!/usr/bin/env python3
from __future__ import annotations

"""Embargoed Walk-Forward driver with CPCV feature screening and PPO training.

Usage (dry-run):
  python scripts/run_wf.py --config configs/settings.yaml --run-name demo --tickers AAPL MSFT \
    --train-start 2024-01-01 --train-end 2024-03-31 --dry-run

Usage (full):
  python scripts/run_wf.py --config configs/settings.yaml --run-name demo --tickers AAPL MSFT \
    --train-start 2024-01-01 --train-end 2024-03-31 --wf-train-days 60 --wf-valid-days 10 \
    --wf-test-days 10 --wf-step-days 10 --embargo-min 15 --feature-pack curated --timesteps 100000
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.wf_cv import EmbargoedWalkForward
from src.data.data_loader import UnifiedDataLoader
from src.features.pipeline import FeaturePipeline, build_features_for_window
from src.features.packs import get_features_for_pack, MICROSTRUCTURE_OHLCV
from src.rl.multiticker_trainer import MultiTickerRLTrainer
from src.utils.feat_cache import feature_cache_path, load_features, save_features
from src.utils.artifacts import write_backtest_artifacts, BacktestResult


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Embargoed Walk-Forward runner with CPCV screening")
    ap.add_argument('--config', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--tickers', nargs='+', required=True)
    ap.add_argument('--train-start', required=True)
    ap.add_argument('--train-end', required=True)
    ap.add_argument('--wf-train-days', type=int, default=None)
    ap.add_argument('--wf-valid-days', type=int, default=None)
    ap.add_argument('--wf-test-days', type=int, default=None)
    ap.add_argument('--wf-step-days', type=int, default=None)
    ap.add_argument('--embargo-min', type=int, default=None)
    ap.add_argument('--timesteps', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--feature-pack', type=str, default=None)
    ap.add_argument('--feature-list-path', type=str, default=None)
    ap.add_argument('--feature-screen-run', type=str, default=None, help='Use curated list from results/features/<name>/ if pack starts with curated*')
    ap.add_argument('--reward-mix', type=str, default=None, help='Reward mix parameters in format ret=1.0,turnover=0.2,inventory=0.05,dsr=0.0')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--strict-test-window', action='store_true', help='Skip tickers with empty test slice; skip window if all empty')
    ap.add_argument('--fast-smoke', action='store_true', help='Fast PPO smoke mode: small MLP policy, fewer steps, no TB/eval, env max 2500 bars')
    ap.add_argument('--entropy', type=float, default=0.01, help='Entropy coeff for PPO in fast-smoke (exploration). Defaults to 0.01')
    ap.add_argument('--early-stop', type=str, default=None, help='Early stop spec: "check_freq=10,min_delta=0.001,patience=5" (ignored in --fast-smoke)')
    ap.add_argument('--no-cache', action='store_true', help='Force recompute features (skip cache LOAD) but still SAVE to cache path')
    return ap.parse_args()


def _compute_target_returns(ohlcv: pd.DataFrame) -> pd.Series:
    # Next-bar return per ticker for screening
    df = ohlcv.copy()
    if 'ticker' not in df.columns:
        raise ValueError("OHLCV must have a 'ticker' column for screening")
    df = df.sort_values(['ticker', df.index.name])
    ret = df.groupby('ticker')['close'].apply(lambda s: s.pct_change().shift(-1))
    ret.name = 'target_ret'
    return ret


def main() -> int:
    args = parse_args()
    # Default to WARNING to trim training-time noise; keep cache/window INFO
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("wf.features")
    log.setLevel(logging.INFO)
    cfg = load_config(args.config)
    wf_cfg = cfg.get('wf', {}) if isinstance(cfg, dict) else {}
    tz = str(wf_cfg.get('timezone', 'America/New_York'))
    # Build windows using EmbargoedWalkForward(start, end, ...)
    start = pd.Timestamp(args.train_start, tz=tz)
    end = pd.Timestamp(args.train_end, tz=tz)
    wf_iter = EmbargoedWalkForward(
        start=start,
        end=end,
        train_days=int(args.wf_train_days or wf_cfg.get('train_days', 60)),
        valid_days=int(args.wf_valid_days or wf_cfg.get('valid_days', 10)),
        test_days=int(args.wf_test_days or wf_cfg.get('test_days', 10)),
        step_days=int(args.wf_step_days or wf_cfg.get('step_days', 10)),
        embargo_min=int(args.embargo_min or wf_cfg.get('embargo_min', 15)),
        tz=tz,
    )
    # Default strict-test-window from config if not provided on CLI
    if not getattr(args, 'strict_test_window', False):
        try:
            args.strict_test_window = bool(wf_cfg.get('strict_test_window', False))
        except Exception:
            pass

    run_root = Path('results/wf') / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    # Load data for outer bound [train_start, train_end]
    loader = UnifiedDataLoader(config_path=args.config)
    frames: List[pd.DataFrame] = []
    for t in args.tickers:
        try:
            df = loader.load_ohlcv(t, start, end)
            if df.empty:
                continue
            df['ticker'] = t
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("No OHLCV data available for requested outer window and tickers.")
    ohlcv_all = pd.concat(frames, axis=0)

    # Build a deterministic cfg hash from feature-pack, feature config, curated seed, and loader params
    feat_cfg = cfg.get('features', {}) if isinstance(cfg, dict) else {}
    # Resolve screen-run name early if available
    screen_run_name: Optional[str] = None
    try:
        if getattr(args, 'feature_screen_run', None):
            screen_run_name = str(args.feature_screen_run)
        elif isinstance(cfg, dict):
            screen_run_name = str((feat_cfg or {}).get('screen_run', '') or '')
        if screen_run_name == '':
            screen_run_name = None
    except Exception:
        screen_run_name = None
    # Curated seed list to affect cache key (union with MICROSTRUCTURE_OHLCV)
    curated_seed: List[str] = []
    try:
        if screen_run_name and str(getattr(args, 'feature_pack', '')).startswith('curated'):
            base = (Path('results/features') / screen_run_name).resolve()
            src = base / 'features_used.txt'
            if not src.exists():
                src = base / 'curated_topN.txt'
            if src.exists():
                curated_seed = [l.strip() for l in src.read_text().splitlines() if l.strip()]
        # Always union with MICROSTRUCTURE_OHLCV to reflect enforced microstructure union
        for m in MICROSTRUCTURE_OHLCV:
            if m not in curated_seed:
                curated_seed.append(m)
    except Exception:
        curated_seed = []
    cfg_key: Dict[str, Any] = {
        'feature_pack': args.feature_pack or '',
        'features_cfg': feat_cfg,
        'feature_screen_run': screen_run_name or '',
        'curated_hash': hashlib.sha1("|".join(sorted(curated_seed)).encode('utf-8')).hexdigest()[:12] if curated_seed else '',
        'loader': {
            'session_tz': getattr(loader, 'session_tz', None),
            'apply_rth_resample': getattr(loader, 'apply_rth_resample', None),
            'strict_resample': getattr(loader, 'strict_resample', None),
            'min_minutes_per_day': getattr(loader, 'min_minutes_per_day', None),
            'data_source': getattr(loader, 'data_source', None),
            'default_timeframe': getattr(loader, 'default_timeframe', None),
        },
    }
    try:
        import json as _json
        cfg_bytes = _json.dumps(cfg_key, sort_keys=True, default=str).encode('utf-8')
    except Exception:
        cfg_bytes = repr(cfg_key).encode('utf-8')
    cfg_hash = hashlib.sha1(cfg_bytes).hexdigest()[:12]

    # Feature cache: attempt to load per-ticker cached features for the OUTER span
    tickers_available = sorted(pd.unique(ohlcv_all['ticker']))
    cache_paths = {
        t: feature_cache_path(
            args.run_name,
            t,
            split='outer',
            start=args.train_start,
            end=args.train_end,
            cfg_hash=cfg_hash,
        )
        for t in tickers_available
    }

    if (not getattr(args, 'no_cache', False)) and all(p.exists() for p in cache_paths.values()):
        log.info(
            "Feature cache HIT for run=%s span=[%s..%s] cfg=%s (tickers=%d)",
            args.run_name, args.train_start, args.train_end, cfg_hash, len(cache_paths),
        )
        parts = []
        for t, p in cache_paths.items():
            f = load_features(p)
            if not f.empty:
                parts.append(f)
        feats_all = pd.concat(parts, axis=0) if parts else pd.DataFrame(index=ohlcv_all.index)
    else:
        if getattr(args, 'no_cache', False):
            log.info(
                "Feature cache BYPASS (--no-cache) for run=%s span=[%s..%s] cfg=%s → computing",
                args.run_name, args.train_start, args.train_end, cfg_hash,
            )
        else:
            log.info(
                "Feature cache MISS for run=%s span=[%s..%s] cfg=%s → computing",
                args.run_name, args.train_start, args.train_end, cfg_hash,
            )
        # Build features once and reuse slicing per window
        pipe = FeaturePipeline(feat_cfg)
        feats_all = pipe.fit_transform(ohlcv_all)
        # Persist per-ticker caches for full outer span
        for t in tickers_available:
            f_t = feats_all[feats_all.get('ticker', pd.Series(index=feats_all.index)).eq(t)]
            save_features(f_t, cache_paths[t])

    # Optionally reduce features by pack or explicit list before screening
    available = [c for c in feats_all.columns if c != 'ticker']
    selected_base: Optional[List[str]] = None
    if args.feature_list_path:
        p = Path(args.feature_list_path)
        if p.exists():
            selected_base = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    elif args.feature_pack:
        selected_base = get_features_for_pack(available, [args.feature_pack])

    # Resolve screen-run name from CLI or config
    screen_run_name: Optional[str] = None
    try:
        if args.feature_screen_run:
            screen_run_name = str(args.feature_screen_run)
        elif isinstance(cfg, dict):
            screen_run_name = str((cfg.get('features', {}) or {}).get('screen_run', '') or '')
        if screen_run_name == '':
            screen_run_name = None
    except Exception:
        screen_run_name = None
    warn_missing_screen_once = False

    # If early-stop specified on CLI, attach to cfg for trainer unless fast-smoke
    if args.early_stop and not args.fast_smoke and isinstance(cfg, dict):
        # Parse "k=v" comma list into types
        es_cfg: Dict[str, Any] = {}
        try:
            for pair in str(args.early_stop).split(','):
                if not pair.strip():
                    continue
                k, v = pair.split('=', 1)
                k = k.strip()
                v = v.strip()
                # Cast to int/float when possible
                if v.isdigit():
                    es_cfg[k] = int(v)
                else:
                    try:
                        es_cfg[k] = float(v)
                    except Exception:
                        es_cfg[k] = v
        except Exception:
            es_cfg = {}
        cfg.setdefault('rl', {})
        cfg['rl']['early_stop'] = es_cfg

    # Enumerate windows
    windows = [(w.train_start, w.train_end, w.valid_start, w.valid_end, w.test_start, w.test_end) for w in wf_iter]
    print(f"[WF] Computed {len(windows)} windows between {start} and {end}")
    if not windows:
        raise SystemExit("No windows produced by EmbargoedWalkForward. Check date bounds and step sizes.")
    if args.dry_run:
        print("k | train_start              | train_end                | valid_start              | valid_end                | test_start               | test_end                 | embargo_min")
        for i, (ts, te, vs, ve, ts2, te2) in enumerate(windows, 0):
            print(f"{i:02d} | {ts} | {te} | {vs} | {ve} | {ts2} | {te2} | {int(args.embargo_min or wf_cfg.get('embargo_min', 15))}")
        return 0

    # Train/backtest per window
    per_win: List[Dict[str, Any]] = []
    skipped_windows: List[int] = []
    from scripts.feature_screen import run_cpcv_feature_screen
    for wi, (tr_s, tr_e, va_s, va_e, te_s, te_e) in enumerate(windows, 0):
        wdir = run_root / f"window_{wi:02d}"
        wdir.mkdir(parents=True, exist_ok=True)
        # Slice frames
        train_mask_d = (ohlcv_all.index >= tr_s) & (ohlcv_all.index <= tr_e)
        test_mask_d = (ohlcv_all.index >= te_s) & (ohlcv_all.index <= te_e)
        train_mask_f = (feats_all.index >= tr_s) & (feats_all.index <= tr_e)
        test_mask_f = (feats_all.index >= te_s) & (feats_all.index <= te_e)
        o_tr_full = ohlcv_all[train_mask_d]
        o_te_full = ohlcv_all[test_mask_d]
        X_tr_full = feats_all[train_mask_f]
        X_te_full = feats_all[test_mask_f]

        # Split per ticker; enforce strict-test-window if requested
        tickers = sorted(pd.unique(o_tr_full['ticker']))
        per_ticker_rows: List[Dict[str, Any]] = []
        any_ticker_kept = False
        window_pool_metrics = []
        for tck in tickers:
            print(f"[WF] window {wi:02d} ticker {tck}: start")
            o_tr = o_tr_full[o_tr_full['ticker'] == tck]
            o_te = o_te_full[o_te_full['ticker'] == tck]
            X_tr = X_tr_full[X_tr_full.get('ticker', pd.Series(index=X_tr_full.index)).eq(tck)]
            X_te = X_te_full[X_te_full.get('ticker', pd.Series(index=X_te_full.index)).eq(tck)]
            print(f"[WF] window {wi:02d} ticker {tck}: sliced train={len(o_tr)} test={len(o_te)} Xtr={len(X_tr)} Xte={len(X_te)}")
            if args.strict_test_window and o_te.empty:
                print(f"[WARN] window {wi:02d} ticker {tck}: empty TEST slice; skipping ticker")
                continue
            if o_te.empty:
                continue
            any_ticker_kept = True
            # Preselect by pack if provided
            if selected_base:
                keep_tr = [c for c in X_tr.columns if (c in selected_base or c == 'ticker')]
                keep_te = [c for c in X_te.columns if (c in selected_base or c == 'ticker')]
                X_tr = X_tr[keep_tr]
                X_te = X_te[keep_te]

            # Target for screening: next bar return per ticker
            y_tr = _compute_target_returns(o_tr).reindex(X_tr.index).fillna(0.0)

            # CPCV feature screening or use a provided screen run
            curated: List[str] = []
            screen_dir: Optional[Path] = None
            if screen_run_name and str(args.feature_pack or '').startswith('curated'):
                base = (Path('results/features') / screen_run_name).resolve()
                if base.exists():
                    screen_dir = base
                    used = base / 'features_used.txt'
                    topn = base / 'curated_topN.txt'
                    src = used if used.exists() else (topn if topn.exists() else None)
                    if src is not None:
                        curated = [l.strip() for l in src.read_text().splitlines() if l.strip()]
                    else:
                        if not warn_missing_screen_once:
                            print(f"[WARN] no curated list found in {base}; will run screening")
                            warn_missing_screen_once = True
                        screen_dir = None
                else:
                    if not warn_missing_screen_once:
                        print(f"[WARN] feature-screen-run '{screen_run_name}' not found in results/features; will run screening")
                        warn_missing_screen_once = True
            if screen_dir is None:
                screen_run = f"{args.run_name}_w{wi:02d}_{tck}"
                try:
                    cap = 50000
                    if len(X_tr) > cap:
                        Xs = X_tr.tail(cap)
                        ys = y_tr.reindex(Xs.index)
                    else:
                        Xs = X_tr
                        ys = y_tr
                    groups = np.arange(len(Xs))
                    print(f"[WF] window {wi:02d} ticker {tck}: screening rows={len(Xs)} cols={Xs.shape[1]-('ticker' in Xs.columns)}")
                    ns = 5 if len(Xs) <= 25000 else 10
                    _ = run_cpcv_feature_screen(
                        Xs, ys, groups, run_name=screen_run, n_slices=ns, n_test_groups=2, embargo_bars=int(args.embargo_min or 15), top_n=40
                    )
                    screen_dir = Path('results/features') / screen_run
                except Exception as e:
                    print(f"[WARN] window {wi:02d} ticker {tck}: screening failed ({e}); fallback to base pack")
                    cols = [c for c in X_tr.columns if c != 'ticker']
                    screen_dir = Path('results/features') / screen_run
                    screen_dir.mkdir(parents=True, exist_ok=True)
                    (screen_dir / 'curated_topN.txt').write_text("\n".join(cols[:40]) + "\n")
            # Resolve curated feature list and build training matrix for this window/ticker
            if screen_dir is None:
                screen_dir = Path('results/features') / f"{args.run_name}_w{wi:02d}_{tck}"
            try:
                feats_window, curated = build_features_for_window(
                    ohlcv=o_tr,
                    base_features=X_tr,
                    run_name=args.run_name,
                    ticker=tck,
                    window_k=wi,
                    window_dir=wdir,
                    screen_dir=screen_dir,
                    feature_pack=args.feature_pack,
                    start=tr_s,
                    end=te_e,
                    use_cache=not getattr(args, 'no_cache', False),
                    logger=log,
                )
            except AssertionError as err:
                raise
            except Exception as err:
                log.warning(
                    "[features] window=%02d ticker=%s curated resolution failed (%s); falling back to raw features",
                    wi,
                    tck,
                    err,
                )
                curated = [c for c in X_tr.columns if c != 'ticker']
                feats_window = X_tr[['ticker'] + curated] if 'ticker' in X_tr.columns else X_tr[curated]

            curated = [c for c in curated if c != 'ticker']
            ticker_col = 'ticker'
            if ticker_col not in feats_window.columns:
                feats_window = feats_window.copy()
                feats_window[ticker_col] = tck
            ordered_cols = [ticker_col] + [c for c in curated]
            feats_window = feats_window.loc[:, ordered_cols]
            X_tr2 = feats_window

            X_te2 = X_te.copy()
            if ticker_col not in X_te2.columns:
                X_te2.insert(0, ticker_col, tck)
            else:
                X_te2[ticker_col] = tck
            for col in curated:
                if col not in X_te2.columns:
                    X_te2[col] = 0.0
            X_te2 = X_te2.loc[:, [ticker_col] + curated]

            # Diagnostics on curated set
            try:
                includes_micro = any(m in curated for m in MICROSTRUCTURE_OHLCV)
                missing = [m for m in MICROSTRUCTURE_OHLCV if m not in curated]
                log.info(
                    "[features] window=%02d ticker=%s curated count=%d includes_micro=%s missing=%s",
                    wi,
                    tck,
                    len(curated),
                    includes_micro,
                    missing[:3],
                )
                if not includes_micro:
                    print(f"[WARN] window {wi:02d} ticker {tck}: microstructure proxies absent; shorts may underperform")
            except Exception:
                pass

            # Train PPO (single-ticker per env via trainer)
            # Fast-smoke overrides: light MLP, fewer steps, no TB/eval, env cap
            if isinstance(cfg, dict):
                cfg.setdefault('rl', {})
                if args.fast_smoke:
                    cfg['rl']['fast_smoke'] = True
            trainer = MultiTickerRLTrainer(cfg)
            # Timesteps + fast-smoke hyperparams
            trainer.hp.total_steps = int(args.timesteps)
            if args.fast_smoke:
                trainer.hp.n_steps = 512
                trainer.hp.batch_size = 512
                # Encourage exploration to avoid mode collapse
                trainer.hp.ent_coef = float(max(0.0, args.entropy))
                trainer.hp.clip_range = 0.2
                trainer.hp.gae_lambda = 0.9
                trainer.hp.gamma = 0.99
                trainer.hp.target_kl = 0.02
            print(f"[WF] window {wi:02d} ticker {tck}: training timesteps={trainer.hp.total_steps}")
            # Train + backtest, then write artifacts robustly
            print(f"[WF] window {wi:02d} ticker {tck}: train+backtest start")
            window_dir = wdir
            (wdir / tck).mkdir(parents=True, exist_ok=True)
            try:
                res: BacktestResult = trainer.train_and_backtest(
                    data=o_tr,
                    features=X_tr2,
                    output_dir=wdir / tck / 'backtest',
                    eval_episodes=1,
                )
            except Exception as e:
                # On failure, emit FAIL artifacts with minimal context
                import pandas as _pd
                empty = BacktestResult(
                    trades=_pd.DataFrame(),
                    equity=_pd.DataFrame({'equity': []}),
                    steps=_pd.DataFrame(),
                    metrics={"error": str(e)},
                    feature_names=curated,
                    baselines={},
                )
                write_backtest_artifacts(window_dir, tck, empty)
                raise
            else:
                write_backtest_artifacts(window_dir, tck, res)
            print(f"[WF] window {wi:02d} ticker {tck}: train+backtest done")

            # Pull per-ticker metrics
            pm = dict(res.metrics or {})
            # Handle nested portfolio_metrics structure
            if 'portfolio_metrics' in pm and isinstance(pm['portfolio_metrics'], dict):
                # Merge nested metrics into top level
                pm.update(pm['portfolio_metrics'])
            # Verify features include microstructure signals used for parity/shorts
            try:
                feats_used = set(res.feature_names or curated)
                if not any(m in feats_used for m in MICROSTRUCTURE_OHLCV):
                    print(f"[WARN] window {wi:02d} ticker {tck}: feature set lacks OHLCV micro proxies; shorts may underperform")
            except Exception:
                pass
            curated = list(res.feature_names or curated)
            # Save action mix
            mix = {
                'long_steps': int(pm.get('long_steps', 0)),
                'short_steps': int(pm.get('short_steps', 0)),
                'flat_steps': int(pm.get('flat_steps', 0)),
                'flips': int(pm.get('flips', 0)),
                'avg_hold_min_long': float(pm.get('avg_duration_minutes_long', 0.0)),
                'avg_hold_min_short': float(pm.get('avg_duration_minutes_short', 0.0)),
                'exposure_pct': float(pm.get('exposure_pct', 0.0)),
                'turnover': float(pm.get('turnover', 0.0)),
                'entries': int(pm.get('entries', 0)),
                'exits': int(pm.get('exits', 0)),
                'long_pf': float(pm.get('long_pf', 0.0)),
                'short_pf': float(pm.get('short_pf', 0.0)),
                'long_ret': float(pm.get('long_ret', 0.0)),
                'short_ret': float(pm.get('short_ret', 0.0)),
                'avg_abs_pos': float(pm.get('avg_abs_pos', 0.0)),
                'avg_abs_dpos': float(pm.get('avg_abs_dpos', 0.0)),
            }
            import csv
            with (wdir / tck / 'action_mix.csv').open('w', newline='') as f:
                wcsv = csv.DictWriter(f, fieldnames=list(mix.keys()))
                wcsv.writeheader(); wcsv.writerow(mix)

            # Per-ticker KPIs (basic, side PF/ret set to NaN placeholders here; can be refined if trades available)
            row_t = {
                'ticker': tck,
                'window': wi,
                'sharpe': float(pm.get('sharpe_ratio', 0.0)),
                'pf': float(pm.get('profit_factor', 0.0)),
                'ret': float(pm.get('total_return', 0.0)),
                'maxdd': float(pm.get('max_drawdown', 0.0)),
                'trades': int(pm.get('total_trades', 0)),
                'tx_costs_total': float(pm.get('tx_costs_total', 0.0)),
                'long_trades': int(pm.get('long_trades', 0)),
                'short_trades': int(pm.get('short_trades', 0)),
                'long_pf': float(pm.get('long_pf', 0.0)) if pm.get('long_pf') is not None else None,
                'short_pf': float(pm.get('short_pf', 0.0)) if pm.get('short_pf') is not None else None,
                'long_ret': float(pm.get('long_ret', 0.0)) if pm.get('long_ret') is not None else None,
                'short_ret': float(pm.get('short_ret', 0.0)) if pm.get('short_ret') is not None else None,
                'parity_flag': pm.get('parity_flag', None),
            }
            # Parity flag refinement
            try:
                ls = int(pm.get('long_steps', 0)); ss = int(pm.get('short_steps', 0))
                if ls > 0 and ss > 0:
                    row_t['parity_flag'] = 'BOTH'
                elif ls > 0 and ss == 0:
                    row_t['parity_flag'] = 'ONLY_LONG'
                elif ss > 0 and ls == 0:
                    row_t['parity_flag'] = 'ONLY_SHORT'
            except Exception:
                pass
            per_ticker_rows.append(row_t)
            window_pool_metrics.append(pm)
            # Persist per-ticker summary.json
            (wdir / tck / 'summary.json').write_text(json.dumps({'portfolio_metrics': pm}, indent=2))

        if not any_ticker_kept:
            skipped_windows.append(wi)
            continue

        # Pooled per-window summary across tickers kept
        if window_pool_metrics:
            dfp = pd.DataFrame(window_pool_metrics)
            pooled = {
                'window': wi,
                'sharpe_ratio': float(dfp.get('sharpe_ratio', pd.Series(dtype=float)).mean(skipna=True) if not dfp.empty else 0.0),
                'profit_factor': float(dfp.get('profit_factor', pd.Series(dtype=float)).mean(skipna=True) if not dfp.empty else 0.0),
                'total_return': float(dfp.get('total_return', pd.Series(dtype=float)).mean(skipna=True) if not dfp.empty else 0.0),
                'max_drawdown': float(dfp.get('max_drawdown', pd.Series(dtype=float)).min(skipna=True) if not dfp.empty else 0.0),
                'total_trades': int(dfp.get('total_trades', pd.Series(dtype=float)).sum(skipna=True) if not dfp.empty else 0),
                'long_trades': int(dfp.get('long_trades', pd.Series(dtype=float)).sum(skipna=True) if not dfp.empty else 0),
                'short_trades': int(dfp.get('short_trades', pd.Series(dtype=float)).sum(skipna=True) if not dfp.empty else 0),
                'tx_costs_total': float(dfp.get('tx_costs_total', pd.Series(dtype=float)).sum(skipna=True) if not dfp.empty else 0.0),
                'no_shorts_tickers': int(((dfp.get('short_trades', pd.Series(dtype=float)) == 0) | (dfp.get('short_steps', pd.Series(dtype=float)) == 0)).sum()) if not dfp.empty else 0,
            }
        else:
            pooled = {'window': wi}
        # Save window-level pooled summary and per-ticker table
        (wdir / 'summary_window.json').write_text(json.dumps({'pooled': pooled, 'per_ticker': per_ticker_rows}, indent=2))
        # Accumulate for global aggregate
        per_win.append(pooled)

    # Aggregate
    if not per_win:
        raise SystemExit("No windows evaluated.")
    dfw = pd.DataFrame(per_win)
    # Aggregate
    # Build parity violations list across windows/tickers
    parity_violations = []
    try:
        for wi_idx in range(len(per_win)):
            # Load per-window summary files to inspect per-ticker parity flags
            wdir = run_root / f"window_{wi_idx:02d}"
            if not wdir.exists():
                continue
            for tck in args.tickers:
                summ_path = wdir / tck / 'summary.json'
                if summ_path.exists():
                    try:
                        import json as _json
                        sd = _json.loads(summ_path.read_text())
                        pm = sd.get('portfolio_metrics', {}) or {}
                        flag = sd.get('parity_flag') or pm.get('parity_flag')
                        if flag in ('ONLY_SHORT', 'ONLY_LONG', 'NO_SHORTS', 'NO_LONGS'):
                            reason = 'NO_SHORTS' if flag in ('ONLY_LONG', 'NO_SHORTS') else 'NO_LONGS'
                            parity_violations.append({'window': wi_idx, 'ticker': tck, 'reason': reason})
                    except Exception:
                        pass
    except Exception:
        parity_violations = []

    agg = {
        'agg_sharpe': float(dfw.get('sharpe_ratio', pd.Series(dtype=float)).mean(skipna=True) if not dfw.empty else 0.0),
        'agg_pf': float(dfw.get('profit_factor', pd.Series(dtype=float)).mean(skipna=True) if not dfw.empty else 0.0),
        'agg_ret': float(dfw.get('total_return', pd.Series(dtype=float)).mean(skipna=True) if not dfw.empty else 0.0),
        'agg_maxdd': float(dfw.get('max_drawdown', pd.Series(dtype=float)).min(skipna=True) if not dfw.empty else 0.0),
        'trades_total': int(dfw.get('total_trades', pd.Series(dtype=float)).sum(skipna=True) if not dfw.empty else 0),
        'long_total': int(dfw.get('long_trades', pd.Series(dtype=float)).sum(skipna=True) if not dfw.empty else 0),
        'short_total': int(dfw.get('short_trades', pd.Series(dtype=float)).sum(skipna=True) if not dfw.empty else 0),
        'costs_total': float(dfw.get('tx_costs_total', pd.Series(dtype=float)).sum(skipna=True) if not dfw.empty else 0.0),
        'windows': per_win,
        'tickers': args.tickers,
        'skipped_windows': skipped_windows,
        'no_shorts_windows': int(dfw.get('no_shorts_tickers', pd.Series(dtype=float)).gt(0).sum()) if 'no_shorts_tickers' in dfw.columns else 0,
        'no_shorts_tickers_total': int(dfw.get('no_shorts_tickers', pd.Series(dtype=float)).sum()) if 'no_shorts_tickers' in dfw.columns else 0,
        'parity_violations': parity_violations,
        'parity_ok': bool(len(parity_violations) == 0),
    }
    if parity_violations:
        print(f"[PARITY] Violations detected: {len(parity_violations)} → {parity_violations}")
    (run_root / 'aggregate.json').write_text(json.dumps(agg, indent=2))

    # Active end-of-run checks per window/ticker
    try:
        print("\n[CHECKS] — Per window/ticker diagnostics")
        for wi_idx in range(len(per_win)):
            wdir = run_root / f"window_{wi_idx:02d}"
            if not wdir.exists():
                continue
            for tck in args.tickers:
                tdir = wdir / tck
                if not tdir.exists():
                    continue
                print(f"- window={wi_idx:02d} ticker={tck}")
                # 1) Microstructure inclusion
                try:
                    used_path = tdir / 'features_used.txt'
                    if not used_path.exists():
                        used_path = wdir / 'features_used.txt'
                    curated_list = [l.strip() for l in used_path.read_text().splitlines() if l.strip()] if used_path.exists() else []
                    includes_micro = any(m in curated_list for m in MICROSTRUCTURE_OHLCV)
                    missing = [m for m in MICROSTRUCTURE_OHLCV if m not in curated_list]
                    print(f"  features: includes_micro={includes_micro} missing={missing if missing else []} count={len(curated_list)}")
                except Exception as e:
                    print(f"  features: ERROR reading features_used.txt ({e})")
                # 2) diagnostics.csv
                try:
                    import csv as _csv
                    diag_csv = tdir / 'diagnostics.csv'
                    diag = {}
                    if diag_csv.exists():
                        with diag_csv.open('r') as f:
                            r = _csv.DictReader(f)
                            for row in r:
                                diag[row['metric']] = row['value']
                    ae = float(diag.get('action_entropy', 0.0) or 0.0)
                    ca = diag.get('corr_action_flow')
                    ca_val = None
                    try:
                        ca_val = float(ca)
                    except Exception:
                        ca_val = None
                    ok_diag = (ae > 0.0) and (ca_val is not None) and (abs(ca_val) > 0.0)
                    print(f"  diagnostics: action_entropy={ae:.4f} corr_action_flow={ca_val} ok={ok_diag}")
                except Exception as e:
                    print(f"  diagnostics: ERROR ({e})")
                # 3) Costs: summary vs trades
                try:
                    import pandas as _pd
                    sum_path = tdir / 'summary.json'
                    with sum_path.open('r') as f:
                        sdat = json.load(f)
                    pm = (sdat.get('portfolio_metrics') if isinstance(sdat, dict) else {}) or {}
                    tx_sum = float(pm.get('tx_costs_total', 0.0) or 0.0)
                    trades_path = tdir / 'trades.csv'
                    tdf = _pd.read_csv(trades_path)
                    # Compute per row used cost = max(total_cost, total_cost_est)
                    if 'total_cost' not in tdf.columns:
                        tdf['total_cost'] = 0.0
                    if 'total_cost_est' not in tdf.columns:
                        tdf['total_cost_est'] = 0.0
                    used = _pd.concat([
                        _pd.to_numeric(tdf['total_cost'], errors='coerce').fillna(0.0),
                        _pd.to_numeric(tdf['total_cost_est'], errors='coerce').fillna(0.0)
                    ], axis=1).max(axis=1)
                    sum_costs = float(used.sum())
                    ok_costs = abs(tx_sum - sum_costs) < 1e-6
                    print(f"  costs: summary={tx_sum:.4f} trades_sum={sum_costs:.4f} ok={ok_costs}")
                except Exception as e:
                    print(f"  costs: ERROR ({e})")
                # 4) Parity
                try:
                    flag = sdat.get('parity_flag') or pm.get('parity_flag')
                    ls = int(pm.get('long_steps', 0) or 0)
                    ss = int(pm.get('short_steps', 0) or 0)
                    lt = int(pm.get('long_trades', 0) or 0)
                    st = int(pm.get('short_trades', 0) or 0)
                    ok_parity = (flag == 'BOTH') and (ls > 0) and (ss > 0) and (lt > 0) and (st > 0)
                    print(f"  parity: flag={flag} long_steps={ls} short_steps={ss} long_trades={lt} short_trades={st} ok={ok_parity}")
                except Exception as e:
                    print(f"  parity: ERROR ({e})")
    except Exception as e:
        print(f"[CHECKS] ERROR while running validations: {e}")

    # Append registry row
    try:
        import csv, time
        reg_dir = Path('results/_registry')
        reg_dir.mkdir(parents=True, exist_ok=True)
        reg_csv = reg_dir / 'runs.csv'
        row = {
            'run_name': args.run_name,
            'wf_windows': int(len(per_win)),
            'agg_sharpe': float(agg['agg_sharpe']),
            'agg_pf': float(agg['agg_pf']),
            'agg_ret': float(agg['agg_ret']),
            'trades_total': int(agg['trades_total']),
            'long_total': int(agg['long_total']),
            'short_total': int(agg['short_total']),
            'costs_total': float(agg['costs_total']),
            'seed': int(args.seed),
            'timestamp': int(time.time()),
        }
        header = list(row.keys())
        write_header = not reg_csv.exists()
        with reg_csv.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
