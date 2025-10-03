# Codex Prompts — Steps 4 through 9 (inclusive)

These are copy‑paste prompts for Codex to apply, in order. They assume you already created the OHLCV microstructure module and resolver from earlier steps.

---

## 4) Pipeline: compute OHLCV proxies, never try L1, write `features_used`, build cache key

**To Codex:** Edit `src/features/pipeline.py` to:

- Detect L1 once: `HAS_L1 = all(c in df.columns for c in ("bid_price","ask_price","bid_size","ask_size"))`
- Resolve curated with `resolve_curated_pack(...)`
- Compute proxies with `OHLCV_MICRO`
- Write `features_used.txt` per window
- Build cache hash with resolved features + curated token + micro module path
- Never log “ofi_best missing.” L1 remap happens upstream.

Insert or replace the core with something like:

```python
# inside build_features_for_window(...)
from pathlib import Path
import pandas as pd
from src.features.packs import resolve_curated_pack, LAST_CURATED_CACHE_TOKEN
from src.features.micro_ohlcv import OHLCV_MICRO, MICROSTRUCTURE_OHLCV
from src.utils.feat_cache import feature_cache_path, save_features, load_features, augment_cfg_hash

HAS_L1 = all(c in ohlcv.columns for c in ("bid_price","ask_price","bid_size","ask_size"))
final_features = resolve_curated_pack(Path(screen_dir), has_l1=HAS_L1, logger=logger, window_k=window_k, ticker=ticker)

# cache key
base_hash = f"{run_name}|{ticker}|{start:%Y%m%d}-{end:%Y%m%d}|{feature_pack}|{screen_dir}"
cfg_hash = augment_cfg_hash(
    base_hash,
    run_name=run_name,
    features=final_features,
    micro_module_path=Path("src/features/micro_ohlcv.py"),
    curated_token=LAST_CURATED_CACHE_TOKEN,
)
cache_path = feature_cache_path(run_name, ticker, f"{start:%Y-%m-%d}_{end:%Y-%m-%d}", cfg_hash)

if use_cache and cache_path.exists():
    logger.info(f"Feature cache HIT -> {cache_path}")
    feats = load_features(cache_path)
else:
    logger.info(f"Feature cache {'BYPASS' if not use_cache else 'MISS'} -> computing {len(final_features)} features")
    feats = pd.DataFrame(index=ohlcv.index)
    # compute requested features here, including classic TA ones...
    for name in final_features:
        if name in OHLCV_MICRO:
            feats[name] = OHLCV_MICRO[name](ohlcv)
        # else: compute other packs as you already do
    save_features(feats, cache_path)

# persist features_used.txt at window root
(Path(window_dir) / "features_used.txt").write_text("\n".join(final_features))
return feats, final_features
```

If any requested proxy in `MICROSTRUCTURE_OHLCV` fails to compute, raise:
```python
missing = [f for f in MICROSTRUCTURE_OHLCV if f in final_features and f not in feats.columns]
if missing:
    raise AssertionError(f"Requested micro proxies missing from features: {missing}")
```

---

## 5) Artifacts writer (always emits files, reconciles costs, computes diagnostics)

**To Codex:** Create or replace `src/utils/artifacts.py` with:

```python
# src/utils/artifacts.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, numpy as np, pandas as pd
from typing import Dict, Any, Optional

@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    steps: pd.DataFrame
    metrics: Dict[str, Any]
    feature_names: Optional[list] = None
    baselines: Optional[Dict[str, Dict[str, float]]] = None

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def write_backtest_artifacts(base_dir: Path, ticker: str, res: BacktestResult) -> None:
    tdir = Path(base_dir) / ticker
    _ensure_dir(tdir)
    fails = []

    # trades.csv
    try:
        t = res.trades.copy()
        for c in ["commission_cost","spread_cost","slippage_cost","impact_cost","total_cost","total_cost_est","gross_pnl"]:
            if c not in t: t[c] = 0.0
        t["total_cost_final"] = t[["total_cost","total_cost_est"]].max(axis=1)
        t["net_pnl"] = t["gross_pnl"] - t["total_cost_final"]
        t.to_csv(tdir/"trades.csv", index=False)
    except Exception as e:
        fails.append(f"trades.csv: {e!r}")

    # portfolio_history.csv
    try:
        eq = res.equity.copy()
        if "equity" not in eq.columns: raise ValueError("equity df missing 'equity'")
        eq.to_csv(tdir/"portfolio_history.csv")
    except Exception as e:
        fails.append(f"portfolio_history.csv: {e!r}")

    # steps.parquet
    try:
        s = res.steps.copy()
        must = ["action","pos","price"]
        miss = [m for m in must if m not in s.columns]
        if miss: raise ValueError(f"steps missing {miss}")
        s.index.name = "ts"
        s["action"] = s["action"].astype("int8")
        s["pos"]    = s["pos"].astype("int8")
        s["price"]  = s["price"].astype("float32")
        s.to_parquet(tdir/"steps.parquet", engine="pyarrow")
    except Exception as e:
        fails.append(f"steps.parquet: {e!r}")

    # diagnostics.csv (from steps)
    try:
        steps = pd.read_parquet(tdir/"steps.parquet")
        a = steps["action"].astype("int8").to_numpy()
        vals, cnts = np.unique(a, return_counts=True)
        probs = cnts / cnts.sum()
        action_entropy = float(-(probs * np.log2(probs)).sum())
        flow_col = "ofi_proxy" if "ofi_proxy" in steps.columns else ("signed_vol_delta" if "signed_vol_delta" in steps.columns else None)
        if flow_col and steps[flow_col].notna().sum() > 50:
            corr_action_flow = float(np.corrcoef(np.sign(a), np.sign(steps[flow_col].to_numpy()))[0,1])
        else:
            corr_action_flow = None
        pd.DataFrame({"metric":["action_entropy","corr_action_flow"],"value":[action_entropy, corr_action_flow]}).to_csv(tdir/"diagnostics.csv", index=False)
    except Exception as e:
        fails.append(f"diagnostics.csv: {e!r}")

    # summary.json (reconcile costs, deterministic parity)
    try:
        metrics = dict(res.metrics or {})
        if (tdir/"trades.csv").exists():
            costs_sum = float(pd.read_csv(tdir/"trades.csv")["total_cost_final"].sum())
            metrics["tx_costs_total"] = costs_sum
        for k in ["long_steps","short_steps","long_trades","short_trades"]:
            metrics.setdefault(k, 0)
        if (metrics["long_steps"]>0 or metrics["long_trades"]>0) and (metrics["short_steps"]>0 or metrics["short_trades"]>0):
            metrics["parity_flag"] = "BOTH"
        elif (metrics["long_steps"]>0 or metrics["long_trades"]>0):
            metrics["parity_flag"] = "ONLY_LONG"
        elif (metrics["short_steps"]>0 or metrics["short_trades"]>0):
            metrics["parity_flag"] = "ONLY_SHORT"
        else:
            metrics["parity_flag"] = "NONE"
        if res.baselines: metrics["baselines"] = res.baselines
        (tdir/"summary.json").write_text(json.dumps(metrics, indent=2))
    except Exception as e:
        fails.append(f"summary.json: {e!r}")

    # features_used.txt at window root
    try:
        if res.feature_names:
            (Path(base_dir)/"features_used.txt").write_text("\n".join(res.feature_names))
    except Exception as e:
        fails.append(f"features_used.txt: {e!r}")

    if fails:
        (tdir/"FAIL.txt").write_text("\n".join(fails))
```

---

## 6) Trainer: return `BacktestResult`, attach a flow proxy to steps

**To Codex:** Edit `src/rl/multiticker_trainer.py`:

- Import types:
```python
from src.utils.artifacts import BacktestResult
```

- After backtest, build `steps_df` aligned to timestamps and include a flow proxy:

```python
# assume ts_index, action_series, pos_series, price_series, volume_series and live_feature_frame exist
steps_df = pd.DataFrame({
    "ts": ts_index, "action": action_series, "pos": pos_series, "price": price_series
}).set_index("ts")

for col in ("ofi_proxy","signed_vol_delta"):
    if col in live_feature_frame.columns:
        steps_df[col] = live_feature_frame[col].reindex(steps_df.index).fillna(0.0).astype("float32")
        break

if not any(c in steps_df.columns for c in ("ofi_proxy","signed_vol_delta")):
    import numpy as np
    sgn  = np.sign(price_series.diff().fillna(0.0))
    dvol = volume_series.reindex(ts_index).fillna(0.0)
    ofi  = (sgn * dvol).astype("float32")
    by_day = ofi.groupby(ts_index.date)
    steps_df["ofi_proxy"] = ((ofi - by_day.transform("mean"))/(by_day.transform("std")+1e-12)).fillna(0.0)

# trades_df, equity_df, metrics_dict, baselines_dict are your existing outputs
return BacktestResult(
    trades=trades_df,
    equity=equity_df,
    steps=steps_df,
    metrics=metrics_dict,
    feature_names=resolved_feature_names,
    baselines=baselines_dict,
)
```

Remove any old ad‑hoc file writes from inside the trainer.

---

## 7) WF runner: call the writer, don’t invent its own “checks”

**To Codex:** Edit `scripts/run_wf.py` at the per window/ticker write point:

```python
from pathlib import Path
from src.utils.artifacts import write_backtest_artifacts

window_dir = Path(f"results/wf/{args.run_name}/window_{k:02d}")
window_dir.mkdir(parents=True, exist_ok=True)

res = trainer.train_and_backtest(...)  # returns BacktestResult
write_backtest_artifacts(window_dir, ticker, res)

# keep your aggregate.json creation, but read per-ticker summary.json that was just written
```

Delete any legacy code paths that wrote partial files or “checks” off stale in‑memory data.

---

## 8) Feature screen heatmap: drop zero‑variance before corr

**To Codex:** Edit `scripts/feature_screen.py` where you render `corr_heatmap.png`:

```python
sel = X[topN]
var = sel.var()
keep = var[var > 0].index
sel = sel[keep]
corr = sel.corr().clip(-1, 1).fillna(0.0)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.imshow(corr, interpolation="nearest")
plt.title("Top-N Feature Correlation (zero-variance dropped)")
plt.colorbar()
plt.tight_layout()
plt.savefig(out_dir/"corr_heatmap.png", dpi=120)
plt.close()
```

---

## 9) Audit script

No edits required if you already added `scripts/audit_pipeline.py`. Run it after a WF window to hard‑fail on missing artifacts.
