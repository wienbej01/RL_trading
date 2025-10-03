import numpy as np
import pandas as pd


def compute_corr_action_flow(steps: pd.DataFrame) -> float | None:
    # Mirror the trainer logic: prefer ofi_proxy, else signed_vol_delta
    flow_col = "ofi_proxy" if "ofi_proxy" in steps.columns else ("signed_vol_delta" if "signed_vol_delta" in steps.columns else None)
    if flow_col is None or steps[flow_col].notna().sum() <= 50:
        return None
    s = np.sign(steps["action"].to_numpy())
    o = np.sign(steps[flow_col].to_numpy())
    return float(np.corrcoef(s, o)[0, 1])


def test_corr_action_flow_uses_ofi_proxy_and_is_nonzero():
    rs = np.random.RandomState(0)
    n = 200
    ofi = rs.normal(size=n)
    # Actions follow ofi proxy 70% of the time, flipped otherwise; include some flats
    follow = rs.rand(n) < 0.7
    act = np.sign(ofi)
    act[~follow] *= -1
    flats = rs.rand(n) < 0.1
    act[flats] = 0
    steps = pd.DataFrame({
        'action': act.astype(int),
        'ofi_proxy': ofi.astype(float),
    })
    corr = compute_corr_action_flow(steps)
    assert corr is not None
    assert np.isfinite(corr)
    assert abs(corr) > 0.1, f"expected noticeable alignment, got {corr}"

