"""Enhanced transaction cost estimation for the ExecutionEngine.

This module defines a modified version of the ``estimate_tc`` method used in
``src/sim/execution.py``.  The original implementation returned only the
commission component of transaction costs.  The revised method calls
``estimate_transaction_costs`` to obtain commission, slippage and impact costs and
returns their sum.

To apply this modification, replace the ``estimate_tc`` method in your
``ExecutionEngine`` class with the one defined below.
"""

from typing import Dict


def estimate_tc(self, position: int, price: float) -> float:
    """Compute total transaction cost for closing or opening a position.

    Parameters
    ----------
    self : ExecutionEngine
        The execution engine instance.
    position : int
        The number of contracts to transact (positive for buying, negative for selling).
    price : float
        The execution price per contract.

    Returns
    -------
    float
        The estimated total transaction cost in currency units, including commission,
        slippage and market impact.
    """
    # Delegate to the full cost estimator.  This method should return a dictionary
    # with keys: 'commission', 'slippage', 'impact'.  If any key is missing, its
    # value defaults to zero.
    costs: Dict[str, float] = self.estimate_transaction_costs(position, price)
    commission = float(costs.get("commission", 0.0))
    slippage = float(costs.get("slippage", 0.0))
    impact = float(costs.get("impact", 0.0))
    return commission + slippage + impact