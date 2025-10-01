import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _compute_metrics_from_steps(steps: pd.DataFrame) -> dict:
    s = steps.dropna(subset=["action", "pos"]).copy()
    a = s["action"].to_numpy(dtype=float)
    p = s["pos"].to_numpy(dtype=float)
    # Entropy over {-1,0,1} (base-2)
    vals, cnts = np.unique(a, return_counts=True)
    probs = cnts / cnts.sum() if cnts.sum() > 0 else np.array([1.0])
    action_entropy = float(-np.sum(probs * np.log2(probs + 1e-12))) if probs.size else 0.0
    # Entries/exits/flips using previous/next comparison
    if p.size >= 2:
        p_prev = np.concatenate(([p[0]], p[:-1]))
        entries = int(np.logical_and(p_prev == 0, p != 0).sum())
        exits = int(np.logical_and(p_prev != 0, p == 0).sum())
        flips = int(((np.sign(p_prev) * np.sign(p)) < 0).sum())
    else:
        entries = exits = flips = 0
    # Turnover sum|Δpos|
    turnover = float(np.abs(np.diff(p, prepend=p[0])).sum()) if p.size else 0.0
    # corr(sign(action), sign(flow_proxy)) without threshold for test purposes
    corr_action_flow = None
    flow_col = "ofi_proxy" if "ofi_proxy" in s.columns else ("signed_vol_delta" if "signed_vol_delta" in s.columns else None)
    if flow_col is not None:
        o = s[flow_col].to_numpy(dtype=float)
        mask = np.isfinite(o)
        if mask.any():
            sgn_a = np.sign(a[mask])
            sgn_o = np.sign(o[mask])
            corr_action_flow = float(spearmanr(sgn_a, sgn_o).correlation)
    return {
        "action_entropy": action_entropy,
        "entries": entries,
        "exits": exits,
        "flips": flips,
        "turnover": turnover,
        "corr_action_flow": corr_action_flow,
    }


def test_metrics_from_known_sequence():
    # Sequence: [0,1,1,0,-1,-1,0]
    a = np.array([0, 1, 1, 0, -1, -1, 0], dtype=float)
    # Construct a plausible cumulative position series matching actions with unit size
    # pos follows last non-zero action until zero (flat) desired; simplistic for test
    p = np.array([0, 1, 1, 1, -1, -1, 0], dtype=float)
    idx = pd.date_range("2024-01-01 09:30", periods=len(a), freq="T", tz="America/New_York")
    # Synthetic OFI matches actions half the time (alternate indices), opposite on others
    flow = np.array([0, 1, -1, 0, -1, 1, 0], dtype=float)
    steps = pd.DataFrame({"action": a, "pos": p, "ofi_proxy": flow}, index=idx)

    m = _compute_metrics_from_steps(steps)

    assert m["entries"] == 2, f"expected entries=2, got {m['entries']}"
    assert m["exits"] == 2, f"expected exits=2, got {m['exits']}"
    assert m["flips"] == 1, f"expected flips=1, got {m['flips']}"
    assert abs(m["turnover"] - 4.0) < 1e-9, f"expected turnover=4, got {m['turnover']}"
    assert m["action_entropy"] > 0.0, "action_entropy should be > 0 for mixed actions"
    assert m["corr_action_flow"] is not None, "corr_action_flow should not be None"
    assert -1.0 <= m["corr_action_flow"] <= 1.0, "corr_action_flow must be within [-1, 1]"
