"""Improved daily PnL calculation for the IBKR paper trading client.

The original ``_calculate_daily_pnl`` method in ``src/trading/ibkr_client.py``
returned zero and thus disabled the daily risk checks in paper trading.  This
module provides a replacement implementation that iterates through the trade
history and computes the realised profit or loss for trades executed on the
current day.

To use this code, copy the body of the ``_calculate_daily_pnl`` method below
into your ``IBKRClient`` class.
"""

import datetime
from typing import Any, Iterable


def calculate_daily_pnl(self) -> float:
    """Calculate the realised PnL for the current trading day.

    This function assumes ``self.trade_history`` is an iterable of trade records.
    Each record should provide at least the attributes ``timestamp`` (a
    ``datetime``), ``quantity`` (int), ``entry_price`` (float), ``exit_price`` (float)
    and optionally ``contract_size`` (int).  Trades that are still open (i.e.
    missing an exit price) are ignored in the realised PnL calculation.

    Returns
    -------
    float
        The sum of realised profits and losses for all completed trades on the
        current date.
    """
    today = datetime.datetime.now().date()
    pnl_total: float = 0.0
    trade_history: Iterable[Any] = getattr(self, "trade_history", [])
    for trade in trade_history:
        timestamp = getattr(trade, "timestamp", None)
        if timestamp is None or not hasattr(timestamp, "date"):
            continue
        if timestamp.date() != today:
            continue
        qty = getattr(trade, "quantity", 0)
        entry_price = getattr(trade, "entry_price", None)
        exit_price = getattr(trade, "exit_price", None)
        if entry_price is None or exit_price is None:
            # Skip trades that have not been closed yet
            continue
        contract_size = getattr(trade, "contract_size", 1)
        pnl_total += qty * (exit_price - entry_price) * contract_size
    return pnl_total