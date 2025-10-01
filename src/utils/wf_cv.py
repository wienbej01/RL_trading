from __future__ import annotations
from dataclasses import dataclass
from typing import Generator, Iterable, List, Tuple, Optional
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class WFWindow:
    k: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_minutes: int
    tz: str

def _tz(ts: pd.Timestamp | str, tz: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(tz)
    return t.tz_convert(tz)

def _add_days(ts: pd.Timestamp, days: int) -> pd.Timestamp:
    # business days are safer for intraday equities
    return (ts.tz_convert(None) + pd.tseries.offsets.BDay(days)).tz_localize(ts.tz)

def EmbargoedWalkForward(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    train_days: int,
    valid_days: int,
    test_days: int,
    step_days: int,
    embargo_min: int,
    tz: str = "America/New_York",
) -> Generator[WFWindow, None, None]:
    """
    Produces time windows with purging/embargo:
      [train] [embargo] [valid] [embargo] [test]
    Windows advance by step_days (BDay). Test slices never overlap.
    """
    s = _tz(start, tz)
    e = _tz(end, tz)

    k = 0
    cur_train_start = s
    while True:
        train_start = cur_train_start
        train_end = _add_days(train_start, train_days) - pd.Timedelta(minutes=1)
        emb1_end = train_end + pd.Timedelta(minutes=embargo_min)

        valid_start = emb1_end + pd.Timedelta(minutes=1)
        valid_end = _add_days(valid_start, valid_days) - pd.Timedelta(minutes=1)
        emb2_end = valid_end + pd.Timedelta(minutes=embargo_min)

        test_start = emb2_end + pd.Timedelta(minutes=1)
        test_end = _add_days(test_start, test_days) - pd.Timedelta(minutes=1)

        # Break when test pushes beyond end bound
        if test_start > e or test_end > e:
            break

        yield WFWindow(
            k=k,
            train_start=train_start, train_end=train_end,
            valid_start=valid_start, valid_end=valid_end,
            test_start=test_start, test_end=test_end,
            embargo_minutes=embargo_min, tz=tz
        )

        k += 1
        next_train_start = _add_days(train_start, step_days)
        # Guard against non-advancing iterator (avoid infinite loop)
        if next_train_start <= cur_train_start:
            break
        cur_train_start = next_train_start

class CombinatorialPurgedKFold:
    """
    Minimal CPCV splitter with purging/embargo between contiguous time groups.

    - `groups` is a 1D array of integer group ids aligned 1:1 with samples
    - Selects each contiguous block of `n_test_groups` groups as test
    - Purges `embargo_bars` observations on both sides of the test block from train
    - Yields `(train_idx, test_idx)` as numpy arrays

    Example
    -------
    >>> import numpy as np
    >>> g = np.repeat(np.arange(5), 3)  # 5 groups, 3 samples each => 15 samples
    >>> cpcv = CombinatorialPurgedKFold(n_splits=5, n_test_groups=2, embargo_bars=1)
    >>> splits = list(cpcv.split(g))
    >>> len(splits)
    4
    >>> tr, te = splits[0]
    >>> te.size  # two groups * 3 samples = 6
    6
    """
    def __init__(self, n_splits: int, n_test_groups: int = 2, embargo_bars: int = 0):
        assert n_test_groups >= 1
        assert n_splits >= n_test_groups + 1
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_bars = embargo_bars

    def split(self, groups: np.ndarray) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        g = np.asarray(groups)
        if g.ndim != 1:
            raise ValueError("groups must be a 1D array aligned 1:1 with samples")
        n_samples = g.shape[0]
        if n_samples == 0:
            raise ValueError("Empty groups array")
        uniq = np.unique(g)
        if len(uniq) < self.n_splits:
            raise ValueError("Not enough groups for CPCV")
        # pick contiguous test blocks; simple and leak-safe
        for start in range(0, len(uniq) - self.n_test_groups + 1):
            test_g = uniq[start:start + self.n_test_groups]
            test_mask = np.isin(g, test_g)
            train_mask = ~test_mask
            # purge embargo neighbors
            if self.embargo_bars > 0:
                test_idx = np.where(test_mask)[0]
                lo = max(0, test_idx.min() - self.embargo_bars)
                hi = min(len(g) - 1, test_idx.max() + self.embargo_bars)
                train_mask[lo:hi + 1] = False
            yield np.where(train_mask)[0], np.where(test_mask)[0]
