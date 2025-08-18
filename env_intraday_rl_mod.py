"""Modified logic for the IntradayRLEnv step function.

This module provides an updated implementation of the reward calculation and daily
kill‑switch logic for the `IntradayRLEnv` class used in the RL_Trading system.
Copy the contents of the `step` method below into your existing
`src/sim/env_intraday_rl.py` file, replacing the old implementation of the
`step` method.  The changes implement:

1. Risk penalties that scale with drawdown and realised daily loss.
2. A correct kill‑switch condition that triggers when realised drawdown exceeds
   the configured daily loss threshold.
3. Reward computation that subtracts the penalty terms from the profit and
   loss (PnL).

This module is standalone for illustration; it does not define the full
environment.  Integrate the logic into your existing class.
"""
from typing import Any, Dict, Tuple
import numpy as np


def modified_step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """Execute one time step within the environment with improved risk logic.

    Parameters
    ----------
    self : IntradayRLEnv
        The environment instance on which to operate.
    action : int
        Discrete action selected by the agent (e.g., 0=hold, 1=long, 2=short).

    Returns
    -------
    observation : np.ndarray
        Next observation vector.
    reward : float
        Reward obtained after taking the action.
    done : bool
        Whether the episode has terminated.
    info : dict
        Additional diagnostic information.

    Notes
    -----
    This function must be bound to an instance of `IntradayRLEnv` (e.g., by
    assignment ``IntradayRLEnv.step = modified_step``).  It assumes that
    attributes such as ``self.current_price``, ``self.position``, ``self.cash``,
    ``self.equity``, ``self.drawdown``, ``self.max_drawdown`` and
    ``self.realized_drawdown`` exist and are updated elsewhere in the class.
    """
    # Perform the original step logic: execute order, update positions and compute PnL.
    # This example assumes there is a helper method `_execute_action` that returns the
    # new observation and the raw profit and loss.  Replace this with your existing
    # environment logic.
    observation, pnl = self._execute_action(action)

    # Compute risk penalties.
    # Drawdown penalty scales with the ratio of current drawdown to maximum drawdown.
    if getattr(self, "max_drawdown", 0.0) > 0.0:
        drawdown_penalty = getattr(self, "drawdown", 0.0) / self.max_drawdown
    else:
        drawdown_penalty = 0.0
    # Realised daily loss penalty scales with realised drawdown relative to the maximum
    # allowed daily loss (expressed as a fraction of account size).  Use the
    # `max_daily_loss_r` parameter from the risk manager (R‑units) converted to
    # percentage.
    max_daily_pct = self.risk_manager.max_daily_loss_r * 0.01
    realised_fraction = getattr(self, "realized_drawdown", 0.0) / max_daily_pct if max_daily_pct > 0 else 0.0
    risk_penalty = max(0.0, realised_fraction)

    # Reward equals raw PnL minus penalties.
    reward = pnl - drawdown_penalty - risk_penalty

    # Determine whether episode is done.
    done = False
    # Standard termination conditions (e.g. end of day) are assumed to be checked in
    # your original implementation.  Here we enforce the daily kill‑switch.
    if getattr(self, "realized_drawdown", 0.0) > max_daily_pct:
        done = True

    # Additional diagnostic information can be returned for logging.
    info: Dict[str, Any] = {
        "pnl": pnl,
        "drawdown_penalty": drawdown_penalty,
        "risk_penalty": risk_penalty,
        "realized_drawdown": getattr(self, "realized_drawdown", 0.0),
    }
    return observation, float(reward), bool(done), info