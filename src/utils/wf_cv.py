from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class PurgedExpandingSplit:
    """
    Expanding window CV with purge + embargo between train and test.

    - Splits an ordered DatetimeIndex (optionally per-ticker) into `n_slices`.
    - For slice i, train uses [start .. boundary_i) and test uses
      [boundary_i + embargo .. boundary_{i+1}), where boundaries are
      evenly spaced by index position.
    - Purge is implicit (no overlap); embargo specified in minutes.
    """

    n_slices: int = 6
    embargo_minutes: int = 15

    def split(self, index: pd.DatetimeIndex) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("PurgedExpandingSplit expects a DatetimeIndex")
        n = len(index)
        if n < self.n_slices + 2:
            # Degenerate: yield single split using 70/30 with embargo
            cut = int(0.7 * n)
            yield self._make_slice(index, 0, cut, n)
            return
        bounds = np.linspace(0, n, self.n_slices + 1, dtype=int)
        # produce pairs (train_end, test_end)
        for i in range(1, len(bounds)):
            tr_end = bounds[i]
            te_end = bounds[i] + (bounds[1] - bounds[0])
            tr_end = min(tr_end, n)
            te_end = min(te_end, n)
            if tr_end <= 1 or te_end <= tr_end:
                continue
            tr_idx, te_idx = self._make_slice(index, 0, tr_end, te_end)
            if len(tr_idx) > 0 and len(te_idx) > 0:
                yield tr_idx, te_idx

    def _make_slice(self, idx: pd.DatetimeIndex, start: int, tr_end: int, te_end: int) -> Tuple[np.ndarray, np.ndarray]:
        tr_idx = np.arange(start, tr_end)
        # Embargo window in minutes
        if self.embargo_minutes > 0 and tr_end < len(idx):
            t_tr_end = idx[tr_end - 1]
            embargo_cut = np.searchsorted(idx, t_tr_end + pd.Timedelta(minutes=self.embargo_minutes))
        else:
            embargo_cut = tr_end
        te_start = max(embargo_cut, tr_end)
        te_idx = np.arange(te_start, te_end)
        return tr_idx, te_idx

