"""Extensions to the feature pipeline for cross‑asset and multi‑horizon features.

This module contains an updated version of the ``prepare_features`` method used in
``src/features/pipeline.py``.  It adds:

1. Multi‑horizon return calculations for horizons specified in the configuration (default
   [1, 5, 15] minutes).
2. Cross‑asset volatility features (VIX and CVOL) by joining external CSV files.
3. Macro‑economic event flag features by joining a CSV file containing binary
   indicators for events such as NFP, CPI and FOMC announcements.

To apply these changes, locate the ``prepare_features`` method in your
``FeaturePipeline`` class and insert or replace the corresponding sections.
"""

from typing import List
import pandas as pd


def extended_prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Enhance the DataFrame with technical, microstructure and cross‑asset features.

    This function builds on the existing feature engineering pipeline by computing
    multi‑horizon returns and joining optional cross‑asset data series (VIX, CVOL,
    macro flags).  It assumes that other indicators (SMA, EMA, ATR, RSI, etc.) are
    calculated elsewhere in your pipeline.  Use this as a mixin or copy the
    relevant code blocks into your own implementation.

    Parameters
    ----------
    self : FeaturePipeline
        An instance containing configuration attributes.  The ``self.config``
        dictionary should have ``features``, ``regime`` and ``data`` keys.
    df : pd.DataFrame
        DataFrame indexed by datetime with at least a ``close`` column.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with additional feature columns joined.
    """
    # Multi‑horizon returns (percentage change)
    returns_horizons: List[int] = self.config.features.get("returns_horizons", [1, 5, 15])
    for horizon in returns_horizons:
        col_name = f"return_{horizon}"
        df[col_name] = df["close"].pct_change(periods=horizon)

    # Join VIX data
    if self.config.regime.get("vix", False):
        vix_file = self.config.data.get("vix_file")
        if vix_file:
            vix_df = pd.read_csv(vix_file, parse_dates=["date"])
            vix_df.set_index("date", inplace=True)
            df = df.join(vix_df, how="left")

    # Join CVOL data
    if self.config.regime.get("cvol", False):
        cvol_file = self.config.data.get("cvol_file")
        if cvol_file:
            cvol_df = pd.read_csv(cvol_file, parse_dates=["date"])
            cvol_df.set_index("date", inplace=True)
            df = df.join(cvol_df, how="left")

    # Join macro event flags
    if self.config.regime.get("macro_flags", False):
        macro_file = self.config.data.get("macro_flags_file")
        if macro_file:
            macro_df = pd.read_csv(macro_file, parse_dates=["date"])
            macro_df.set_index("date", inplace=True)
            df = df.join(macro_df, how="left")

    return df