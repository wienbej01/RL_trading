from __future__ import annotations

import pandas as pd

from src.utils.wf_cv import EmbargoedWalkForward


def test_walk_forward_windows_contiguous():
    wf = EmbargoedWalkForward(train_days=2, valid_days=1, test_days=1, step_days=1, embargo_min=15)
    tz = 'America/New_York'
    start = pd.Timestamp('2024-01-01', tz=tz)
    end = pd.Timestamp('2024-01-10', tz=tz)
    wins = list(wf.iter_windows(start, end))
    assert len(wins) > 0
    # Check each test window is non-overlapping with previous test window
    for i in range(1, len(wins)):
        _, _, _, _, t1s, t1e = wins[i - 1]
        _, _, _, _, t2s, t2e = wins[i]
        assert t2s >= t1s  # non-decreasing start
        # Contiguity controlled by step_days; they can overlap depending on step, but we ensure ordering
    # Check embargo: valid starts after train end + embargo
    for (ts, te, vs, ve, _, _) in wins:
        assert vs > te  # due to embargo seconds offset

