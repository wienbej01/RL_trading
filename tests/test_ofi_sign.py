import pandas as pd
import numpy as np

from src.features.microstructure_features import calculate_ofi_best, calculate_order_flow_imbalance


def test_ofi_best_sign_buy_pressure():
    # Build a tiny synthetic tape where bid steps up and ask size depletes
    idx = pd.date_range("2024-05-01 09:30", periods=5, freq="1min", tz="America/New_York")
    bid_p = pd.Series([100.00, 100.01, 100.01, 100.01, 100.02], index=idx)
    ask_p = pd.Series([100.02, 100.02, 100.02, 100.01, 100.01], index=idx)
    bid_q = pd.Series([500, 600, 620, 610, 650], index=idx)   # Δq_b > 0 on step-up
    ask_q = pd.Series([700, 650, 640, 630, 620], index=idx)   # ask queue depletes

    ofi_raw = calculate_ofi_best(bid_p, bid_q, ask_p, ask_q)
    # Check sign at t=1..end (skip first NaN)
    assert ofi_raw[1:].iloc[-1] > 0, "OFI_best should be positive under buy pressure (bid up, ask deplete)"

    # Z-scored version should preserve sign at the event window
    ofi_z = calculate_order_flow_imbalance(bid_p, bid_q, ask_p, ask_q)
    # last finite entry should retain sign
    finite = ofi_z.dropna()
    assert not finite.empty
    assert np.sign(float(finite.iloc[-1])) >= 0, "Z-scored OFI should preserve sign for buy pressure"

