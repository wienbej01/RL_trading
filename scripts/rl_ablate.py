#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _select_features(cfg: Dict[str, Any], variant: str, available: List[str], run_name: str | None = None) -> List[str]:
    from src.features.packs import get_features_for_pack, propose_curated_from_consensus  # type: ignore

    # Build curated from screen outputs if requested
    curated: List[str] | None = None
    if variant.startswith("curated") and run_name:
        cons = Path("results/features") / run_name / "consensus_importance.parquet"
        if cons.exists():
            top_n = int((cfg.get("features", {}) or {}).get("curated_top_n", 40))
            curated = propose_curated_from_consensus(cons, top_n)
    # Base packs from config
    packs = (cfg.get("features", {}) or {}).get("packs", ["price_vol", "microstructure", "context"])  # type: ignore
    selected = get_features_for_pack(available, packs, curated)

    # Minus variants: curated_minus:<pack>
    if variant.startswith("curated_minus:"):
        minus = variant.split(":", 1)[1]
        drop = set(get_features_for_pack(available, [minus]))
        selected = [f for f in selected if f not in drop]
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick RL ablation runner (feature packs)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--feature-pack", required=True, help="curated or curated_minus:<pack>")
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))

    # Build dataset
    from src.utils.config_loader import load_config  # type: ignore
    from src.data.data_loader import UnifiedDataLoader  # type: ignore
    from src.features.pipeline import FeaturePipeline  # type: ignore
    from src.rl.multiticker_trainer import MultiTickerRLTrainer  # type: ignore
    import pandas as pd  # type: ignore

    settings = load_config(args.config)
    meta = cfg
    tickers = meta.get("data", {}).get("multiticker", {}).get("universe", []) or ["SPY"]
    tr_s = meta.get("data", {}).get("start_date", "2024-04-01")
    tr_e = meta.get("data", {}).get("end_date", "2024-08-15")
    te_s = meta.get("walkforward", {}).get("test_start", "2024-08-16") if "walkforward" in meta else tr_e
    te_e = meta.get("walkforward", {}).get("test_end", "2024-09-30") if "walkforward" in meta else tr_e

    loader = UnifiedDataLoader(config_path=args.config)
    dframes = []
    for t in tickers:
        try:
            df = loader.load_ohlcv(symbol=t, start=pd.Timestamp(tr_s), end=pd.Timestamp(te_e))
            if df.empty:
                continue
            df["ticker"] = t
            dframes.append(df)
        except Exception:
            continue
    if not dframes:
        raise SystemExit("No data available for ablation.")
    ohlcv = pd.concat(dframes, axis=0)

    pipe = FeaturePipeline(settings.get("features", {}))
    feats = pipe.fit_transform(ohlcv)
    avail = [c for c in feats.columns if c != "ticker"]
    selected = _select_features(cfg, args.feature_pack, avail, args.run_name)
    feats = feats[[*(selected), "ticker"]]

    # Train+backtest
    out_root = Path("results/ablations") / args.run_name / args.feature_pack.replace(":", "-")
    out_root.mkdir(parents=True, exist_ok=True)
    trainer = MultiTickerRLTrainer(settings)
    trainer.hp.total_steps = int(args.timesteps)
    model = trainer.train(data=ohlcv[(ohlcv.index >= tr_s) & (ohlcv.index <= tr_e)],
                          features=feats[(feats.index >= tr_s) & (feats.index <= tr_e)],
                          output_dir=out_root / "models")
    summ = trainer.backtest(model=model,
                            data=ohlcv[(ohlcv.index >= te_s) & (ohlcv.index <= te_e)],
                            features=feats[(feats.index >= te_s) & (feats.index <= te_e)],
                            output_dir=out_root / "backtest")
    # Baselines (placeholders)
    baselines = {
        "no_trade": {"total_trades": 0, "total_return": 0.0, "sharpe_ratio": 0.0},
        "vwap_fade": {"total_trades": None, "total_return": None, "sharpe_ratio": None},
        "ofi_follow": {"total_trades": None, "total_return": None, "sharpe_ratio": None},
    }
    # Summary
    kpis = {
        "ablation": args.feature_pack,
        "selected_features": selected,
        "backtest_summary": summ,
        "baselines": baselines,
    }
    (out_root / "summary.json").write_text(json.dumps(kpis, indent=2))
    print(str((out_root / "summary.json").resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

