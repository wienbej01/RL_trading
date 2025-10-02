"""
Composite reward function for RL trading environments.

This module implements a composite reward function that combines multiple components
to encourage profitable trading while penalizing undesirable behaviors like excessive
turnover and inventory holding.
"""

import numpy as np
from typing import Dict, Any, Optional


class CompositeReward:
    """
    Composite reward function with configurable weights for different components.
    
    The reward function combines:
    - PnL (profit and loss) component
    - Turnover penalty (penalizes frequent trading)
    - Inventory penalty (penalizes holding positions)
    - Downside risk penalty (penalizes negative returns)
    - Transaction costs (optional)
    """
    
    def __init__(self, w_ret: float = 1.0, w_turnover: float = 0.2, 
                 w_inv: float = 0.05, w_dsr: float = 0.0, include_costs: bool = True):
        """
        Initialize the composite reward function.
        
        Args:
            w_ret: Weight for the return component
            w_turnover: Weight for the turnover penalty
            w_inv: Weight for the inventory penalty
            w_dsr: Weight for the downside risk penalty
            include_costs: Whether to include transaction costs in the reward
        """
        self.w_ret = w_ret
        self.w_turnover = w_turnover
        self.w_inv = w_inv
        self.w_dsr = w_dsr
        self.include_costs = include_costs
        self._prev_pos = 0
        self._ret_hist = []
        
    def reset(self) -> None:
        """Reset the reward function state."""
        self._prev_pos = 0
        self._ret_hist.clear()
        
    def step(self, dPnL: float, pos: int, tx_cost: float = 0.0) -> tuple[float, Dict[str, Any]]:
        """
        Calculate reward for a step.
        
        Args:
            dPnL: Change in profit and loss
            pos: Current position
            tx_cost: Transaction cost for this step
            
        Returns:
            Tuple of (reward, info_dict)
        """
        # Calculate penalties
        turnover_pen = abs(pos - self._prev_pos)
        inv_pen = abs(pos)
        
        # Adjust PnL for transaction costs if enabled
        pnl_term = dPnL - (tx_cost if self.include_costs else 0.0)
        self._ret_hist.append(pnl_term)
        
        # Calculate downside risk penalty
        dsr_pen = 0.0
        if self.w_dsr > 0.0 and len(self._ret_hist) > 0:
            # Use last 64 returns for downside risk calculation
            x = np.array(self._ret_hist[-64:])
            neg = x[x < 0.0]
            if neg.size > 0:
                dsr_pen = float(np.mean(neg**2))  # downside variance proxy
        
        # Update previous position
        self._prev_pos = pos
        
        # Calculate final reward
        r = (self.w_ret * pnl_term - 
             self.w_turnover * turnover_pen - 
             self.w_inv * inv_pen - 
             self.w_dsr * dsr_pen)
        
        # Return reward and diagnostic info
        return float(r), {
            "pnl_term": float(pnl_term),
            "turnover_pen": float(turnover_pen),
            "inv_pen": float(inv_pen),
            "dsr_pen": float(dsr_pen)
        }