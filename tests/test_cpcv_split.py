import numpy as np

from src.utils.wf_cv import CombinatorialPurgedKFold


def test_cpcv_contiguous_blocks_with_embargo():
    # 6 groups, 3 samples per group => 18 samples, groups aligned 1:1 with samples
    groups = np.repeat(np.arange(6), 3)
    n = len(groups)
    splitter = CombinatorialPurgedKFold(n_splits=6, n_test_groups=2, embargo_bars=1)

    for train_idx, test_idx in splitter.split(groups):
        # No overlap
        assert np.intersect1d(train_idx, test_idx).size == 0
        # Determine test span and check embargo removal on both sides
        tmin, tmax = test_idx.min(), test_idx.max()
        left_lo = max(0, tmin - 1)
        right_hi = min(n - 1, tmax + 1)
        # All indices in [left_lo, tmin-1] and [tmax+1, right_hi] must be excluded from train
        left_embargo = np.arange(left_lo, tmin) if left_lo < tmin else np.array([], dtype=int)
        right_embargo = np.arange(tmax + 1, right_hi + 1) if tmax + 1 <= right_hi else np.array([], dtype=int)
        if left_embargo.size:
            assert np.intersect1d(train_idx, left_embargo).size == 0
        if right_embargo.size:
            assert np.intersect1d(train_idx, right_embargo).size == 0

