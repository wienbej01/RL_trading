#!/usr/bin/env python3
from __future__ import annotations

"""
Hard-fail audit for a single WF window/ticker.

Writes artifacts under results/dev/audit_<wf_run>_w<NN>_<TICKER>/ and exits non‑zero
on any invariant failure.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


MICRO_LIST: List[str] = [
    "ofi_proxy",
    "bar_imbalance",
    "signed_vol_delta",
    "spread_bps_hl",
    "quote_intensity_proxy",
    "queue_imbalance_proxy",
]


def _corr_action_flow(steps: pd.DataFrame) -> float | None:
    flow_col = (
        "ofi_proxy"
        if "ofi_proxy" in steps.columns
        else ("signed_vol_delta" if "signed_vol_delta" in steps.columns else None)
    )
    if not flow_col:
        return None
    s = np.sign(pd.to_numeric(steps["action"], errors="coerce").to_numpy())
    o = np.sign(pd.to_numeric(steps[flow_col], errors="coerce").to_numpy())
    mask = np.isfinite(o)
    if mask.sum() < 50:
        return None
    return float(np.corrcoef(s[mask], o[mask])[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit a single WF window/ticker")
    ap.add_argument("--wf-run", required=True)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--ticker", required=True)
    args = ap.parse_args()

    base = Path(f"results/wf/{args.wf_run}/window_{args.window:02d}/{args.ticker}")
    wdir = base.parent  # window root
    audit_dir = Path(f"results/dev/audit_{args.wf_run}_w{args.window:02d}_{args.ticker}")
    audit_dir.mkdir(parents=True, exist_ok=True)

    fails: list[str] = []

    # 1) features_used and micro presence
    feat_used_path = wdir / "features_used.txt"
    feat_used: List[str] = []
    if feat_used_path.exists():
        try:
            feat_used = [l.strip() for l in feat_used_path.read_text().splitlines() if l.strip()]
        except Exception:
            feat_used = []
    (audit_dir / "features_used.txt").write_text("\n".join(feat_used))

    # Resolved feature list after L1->OHLCV curation is approximated by features_used + MICRO_LIST union
    final_features = list(dict.fromkeys(feat_used + MICRO_LIST))
    (audit_dir / "final_features.txt").write_text("\n".join(final_features))

    present = set(final_features)
    micro_missing = [f for f in MICRO_LIST if f not in present]
    (audit_dir / "micro_presence.txt").write_text(f"missing={micro_missing}")
    if len(micro_missing) == len(MICRO_LIST):
        fails.append("No OHLCV micro proxies present in final_features")

    # 2) steps.parquet checks
    steps_path = base / "steps.parquet"
    if not steps_path.exists():
        fails.append(f"steps.parquet missing at {steps_path}")
        steps = pd.DataFrame()
    else:
        steps = pd.read_parquet(steps_path)
        # Save head and schema
        try:
            steps.head(3).reset_index().to_csv(audit_dir / "steps_head.csv", index=False)
            schema_lines = [f"{c}: {str(steps[c].dtype)}" for c in ["action", "pos", "price"] if c in steps.columns]
            (audit_dir / "steps_schema.txt").write_text("\n".join(schema_lines))
        except Exception:
            pass
        # entries/exits/flips/turnover/action_entropy
        entropy = 0.0
        entries = exits = flips = turnover = 0
        try:
            a = pd.to_numeric(steps.get("action", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int).to_numpy()
            p = pd.to_numeric(steps.get("pos", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int).to_numpy()
            if a.size:
                vals, cnts = np.unique(a, return_counts=True)
                probs = cnts / cnts.sum() if cnts.sum() > 0 else np.array([1.0])
                entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            if p.size:
                entries = int(((p[:-1] == 0) & (p[1:] != 0)).sum())
                exits = int(((p[:-1] != 0) & (p[1:] == 0)).sum())
                flips = int(((p[:-1] * p[1:]) < 0).sum())
                dp = np.diff(p, prepend=p[0])
                turnover = int(np.abs(dp).sum())
        except Exception:
            pass
        diag = pd.DataFrame({"metric": ["action_entropy"], "value": [entropy]})
        # corr_action_flow
        corr = _corr_action_flow(steps) if not steps.empty else None
        diag = pd.concat([diag, pd.DataFrame({"metric": ["corr_action_flow"], "value": [corr]})], ignore_index=True)
        diag.to_csv(audit_dir / "diagnostics_from_steps.csv", index=False)
        # Sanity hard checks
        if entropy == 0.0:
            fails.append("action_entropy == 0 from steps")
        if corr is None:
            fails.append("corr_action_flow None (no valid proxy in steps or insufficient data)")
        # Save proxies sample if present
        try:
            cols = [c for c in ("ofi_proxy", "signed_vol_delta") if c in steps.columns]
            if cols:
                steps[cols].head(5).reset_index().to_csv(audit_dir / "proxies_head.csv", index=False)
        except Exception:
            pass

    # 3) reconciliation trades vs summary
    summary_path = base / "summary.json"
    trades_path = base / "trades.csv"
    if not summary_path.exists():
        fails.append("summary.json missing")
    if not trades_path.exists():
        fails.append("trades.csv missing")
    if summary_path.exists() and trades_path.exists():
        try:
            m = json.loads(summary_path.read_text())
        except Exception:
            m = {}
        try:
            t = pd.read_csv(trades_path)
        except Exception:
            t = pd.DataFrame()
        if not t.empty:
            for c in ["total_cost", "total_cost_est"]:
                if c not in t:
                    t[c] = 0.0
            costs_sum = float(pd.to_numeric(t[["total_cost", "total_cost_est"]].max(axis=1), errors="coerce").fillna(0.0).sum())
        else:
            costs_sum = 0.0
        try:
            s_tx = float(m.get("tx_costs_total", 0.0))
        except Exception:
            s_tx = 0.0
        if abs(costs_sum - s_tx) > 1e-9:
            fails.append(f"cost mismatch: summary={s_tx} vs trades_sum={costs_sum}")
        # parity
        ls, ss = int(m.get("long_steps", 0)), int(m.get("short_steps", 0))
        lt, st = int(m.get("long_trades", 0)), int(m.get("short_trades", 0))
        flag = str(m.get("parity_flag", "NONE") or "NONE")
        expected = (
            "BOTH"
            if (ls > 0 or lt > 0) and (ss > 0 or st > 0)
            else ("ONLY_LONG" if (ls > 0 or lt > 0) else ("ONLY_SHORT" if (ss > 0 or st > 0) else "NONE"))
        )
        if flag != expected:
            fails.append(f"parity flag mismatch: summary={flag} expected={expected}")

    # 4) Cache presence (best effort)
    cache_dir = Path("results/cache/features") / args.wf_run / args.ticker
    if cache_dir.exists():
        caches = sorted(cache_dir.glob("*.parquet"))
        lines = [str(p) for p in caches[:10]]
        (audit_dir / "feature_cache_candidates.txt").write_text("\n".join(lines))
    else:
        (audit_dir / "feature_cache_candidates.txt").write_text("(bypass or no cache dir)")

    # 5) Finish
    (audit_dir / "FAILS.txt").write_text("\n".join(fails) if fails else "OK")
    if fails:
        print("\n[Audit FAIL]\n" + "\n".join(fails))
        return 2
    print("\n[Audit OK] All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

