import pandas as pd
from src.utils.wf_cv import EmbargoedWalkForward


def test_embargo_and_non_overlap():
    wf = list(EmbargoedWalkForward(
        start="2024-01-02", end="2024-03-29",
        train_days=10, valid_days=3, test_days=3, step_days=5,
        embargo_min=15, tz="America/New_York"
    ))
    # Non-empty and ordered
    assert len(wf) > 1
    # embargo gaps
    for w in wf:
        assert w.valid_start > w.train_end
        assert (w.valid_start - w.train_end).total_seconds() >= 15*60
        assert w.test_start > w.valid_end
        assert (w.test_start - w.valid_end).total_seconds() >= 15*60
    # test ranges don’t overlap
    tests = [(w.test_start, w.test_end) for w in wf]
    for i in range(1, len(tests)):
        assert tests[i][0] > tests[i-1][1]

