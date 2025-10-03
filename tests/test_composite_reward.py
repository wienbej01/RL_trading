"""
Unit tests for the CompositeReward class.
"""

import numpy as np
import pytest

from src.rl.composite_reward import CompositeReward


def test_composite_reward_initialization():
    """Test that CompositeReward initializes with default parameters."""
    reward = CompositeReward()
    assert reward.w_ret == 1.0
    assert reward.w_turnover == 0.2
    assert reward.w_inv == 0.05
    assert reward.w_dsr == 0.0
    assert reward.include_costs is True
    assert reward._prev_pos == 0
    assert reward._ret_hist == []


def test_composite_reward_reset():
    """Test that CompositeReward reset clears state."""
    reward = CompositeReward()
    reward._prev_pos = 5
    reward._ret_hist = [1.0, 2.0, 3.0]
    
    reward.reset()
    assert reward._prev_pos == 0
    assert reward._ret_hist == []


def test_composite_reward_step():
    """Test that CompositeReward.step calculates reward correctly."""
    reward = CompositeReward(w_ret=1.0, w_turnover=0.2, w_inv=0.05, w_dsr=0.0, include_costs=False)
    
    # First step
    r, info = reward.step(dPnL=10.0, pos=1, tx_cost=0.0)
    
    assert r == 10.0 - 0.2 * 1 - 0.05 * 1  # 10 - 0.2 - 0.05 = 9.75
    assert info["pnl_term"] == 10.0
    assert info["turnover_pen"] == 1.0
    assert info["inv_pen"] == 1.0
    assert info["dsr_pen"] == 0.0
    
    # Second step
    r, info = reward.step(dPnL=5.0, pos=2, tx_cost=0.0)
    
    assert r == 5.0 - 0.2 * 1 - 0.05 * 2  # 5 - 0.2 - 0.1 = 4.7
    assert info["pnl_term"] == 5.0
    assert info["turnover_pen"] == 1.0
    assert info["inv_pen"] == 2.0
    assert info["dsr_pen"] == 0.0


def test_composite_reward_with_costs():
    """Test that CompositeReward includes transaction costs when enabled."""
    reward = CompositeReward(w_ret=1.0, w_turnover=0.2, w_inv=0.05, w_dsr=0.0, include_costs=True)
    
    r, info = reward.step(dPnL=10.0, pos=1, tx_cost=1.0)
    
    assert r == (10.0 - 1.0) - 0.2 * 1 - 0.05 * 1  # 9 - 0.2 - 0.05 = 8.75
    assert info["pnl_term"] == 9.0


def test_composite_reward_dsr():
    """Test that CompositeReward calculates downside risk penalty."""
    reward = CompositeReward(w_ret=1.0, w_turnover=0.2, w_inv=0.05, w_dsr=0.1, include_costs=False)
    
    # Add some negative returns to the history
    reward._ret_hist = [-1.0, -2.0, -3.0, 1.0, 2.0]
    
    r, info = reward.step(dPnL=5.0, pos=1, tx_cost=0.0)
    
    # Calculate expected DSR penalty
    neg_returns = np.array([-1.0, -2.0, -3.0])
    expected_dsr = np.mean(neg_returns ** 2)  # = (1 + 4 + 9) / 3 = 14/3 ≈ 4.67
    
    expected_reward = 5.0 - 0.2 * 1 - 0.05 * 1 - 0.1 * expected_dsr
    assert abs(r - expected_reward) < 1e-10
    assert abs(info["dsr_pen"] - expected_dsr) < 1e-10