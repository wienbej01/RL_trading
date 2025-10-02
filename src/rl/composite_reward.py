"""
Composite reward function for RL trading environments.

This module provides a configurable composite reward function that combines:
- Return component (PnL)
- Turnover penalty (position changes)
- Inventory penalty (position holding)
- Downside risk penalty (negative return variance)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CompositeReward:
    """
    Composite reward function with configurable components.
    
    Reward = w_ret * pnl_term - w_turnover * turnover_pen - w_inv * inv_pen - w_dsr * dsr_pen
    
    Components:
    - pnl_term: PnL minus transaction costs (if include_costs=True)
    - turnover_pen: Absolute position change |pos - prev_pos|
    - inv_pen: Absolute position |pos|
    - dsr_pen: Downside risk penalty (variance of negative returns)
    """
    
    def __init__(
        self,
        w_ret: float = 1.0,
        w_turnover: float = 0.2,
        w_inv: float = 0.05,
        w_dsr: float = 0.0,
        include_costs: bool = True,
        dsr_window: int = 64
    ):
        """
        Initialize composite reward function.
        
        Args:
            w_ret: Weight for return component
            w_turnover: Weight for turnover penalty
            w_inv: Weight for inventory penalty
            w_dsr: Weight for downside risk penalty
            include_costs: Whether to subtract transaction costs from PnL
            dsr_window: Window size for downside risk calculation
        """
        self.w_ret = float(w_ret)
        self.w_turnover = float(w_turnover)
        self.w_inv = float(w_inv)
        self.w_dsr = float(w_dsr)
        self.include_costs = bool(include_costs)
        self.dsr_window = int(dsr_window)
        
        # State tracking
        self._prev_pos = 0
        self._ret_hist = []
        
        logger.info(f"Initialized CompositeReward: w_ret={w_ret}, w_turnover={w_turnover}, "
                   f"w_inv={w_inv}, w_dsr={w_dsr}, include_costs={include_costs}")
    
    def reset(self) -> None:
        """Reset reward function state."""
        self._prev_pos = 0
        self._ret_hist.clear()
        logger.debug("CompositeReward state reset")
    
    def step(self, dPnL: float, pos: int, tx_cost: float = 0.0) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite reward for current step.
        
        Args:
            dPnL: Delta PnL for this step
            pos: Current position
            tx_cost: Transaction cost for this step
            
        Returns:
            Tuple of (reward, breakdown_dict)
        """
        # Calculate turnover penalty (position change)
        turnover_pen = abs(pos - self._prev_pos)
        
        # Calculate inventory penalty (position holding)
        inv_pen = abs(pos)
        
        # Calculate PnL term (subtract costs if enabled)
        pnl_term = dPnL - (tx_cost if self.include_costs else 0.0)
        
        # Store return for downside risk calculation
        self._ret_hist.append(pnl_term)
        
        # Calculate downside risk penalty
        dsr_pen = 0.0
        if self.w_dsr > 0.0 and len(self._ret_hist) >= 2:
            # Use recent returns for downside risk calculation
            recent_returns = np.array(self._ret_hist[-self.dsr_window:])
            negative_returns = recent_returns[recent_returns < 0.0]
            if len(negative_returns) > 0:
                # Downside variance as risk proxy
                dsr_pen = float(np.mean(negative_returns ** 2))
        
        # Calculate composite reward
        reward = (
            self.w_ret * pnl_term
            - self.w_turnover * turnover_pen
            - self.w_inv * inv_pen
            - self.w_dsr * dsr_pen
        )
        
        # Update state
        self._prev_pos = pos
        
        # Create breakdown dictionary for logging
        breakdown = {
            'pnl_term': float(pnl_term),
            'turnover_pen': float(turnover_pen),
            'inv_pen': float(inv_pen),
            'dsr_pen': float(dsr_pen),
            'reward': float(reward)
        }
        
        # Log detailed breakdown periodically
        if len(self._ret_hist) % 100 == 0:
            logger.debug(f"CompositeReward breakdown: {breakdown}")
        
        return float(reward), breakdown
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reward function statistics."""
        if not self._ret_hist:
            return {}
        
        returns = np.array(self._ret_hist)
        return {
            'total_steps': len(self._ret_hist),
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'total_pnl': float(np.sum(returns)),
            'negative_returns': int(np.sum(returns < 0)),
            'positive_returns': int(np.sum(returns > 0)),
        }