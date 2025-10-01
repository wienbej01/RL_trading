from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class _ESParams:
    check_freq: int = 10
    min_delta: float = 1e-3
    patience: int = 5


class EarlyStopNoImprove(BaseCallback):
    """Early stop when episodic return rolling mean stops improving.

    - Every `check_freq` rollout updates, read `rollout/ep_rew_mean` from SB3 logger
      and compare to best-so-far with tolerance `min_delta`.
    - If no improvement for `patience` consecutive checks, stop training.
    - Falls back to an approximation from the latest rollout buffer rewards when
      logger values are unavailable.
    """

    def __init__(self, check_freq: int = 10, min_delta: float = 1e-3, patience: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.params = _ESParams(int(max(1, check_freq)), float(min_delta), int(max(1, patience)))
        self._updates_seen = 0
        self._no_improve = 0
        self._best: float = -np.inf
        self._stop: bool = False
        self._history: Deque[float] = deque(maxlen=100)

    def _read_ep_rew_mean(self) -> Optional[float]:
        # Try SB3 logger first
        try:
            logger = getattr(self.model, "logger", None)
            if logger is not None and hasattr(logger, "name_to_value"):
                val = logger.name_to_value.get("rollout/ep_rew_mean", None)  # type: ignore[attr-defined]
                if val is not None:
                    return float(val)
        except Exception:
            pass
        # Fallback: approximate from rollout buffer rewards (mean sum of rewards across envs)
        try:
            rb = self.locals.get('rollout_buffer', None)  # type: ignore[attr-defined]
            if rb is None:
                return None
            # Flatten last rollout rewards and sum per environment
            rews = np.array(rb.rewards)
            # In SB3, rewards often stored as shape (n_steps, n_envs, 1) or (n_steps, n_envs)
            rews = np.squeeze(rews)
            if rews.ndim == 1:
                return float(np.sum(rews))
            if rews.ndim == 2:
                ret_per_env = rews.sum(axis=0)
                return float(np.mean(ret_per_env))
            return None
        except Exception:
            return None

    def _on_rollout_end(self) -> None:  # type: ignore[override]
        self._updates_seen += 1
        if self._updates_seen % self.params.check_freq != 0:
            return
        cur = self._read_ep_rew_mean()
        if cur is None:
            return
        self._history.append(float(cur))
        # Improvement check
        if cur > self._best + self.params.min_delta:
            self._best = float(cur)
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self.verbose:
                print(f"[EarlyStopNoImprove] no improve #{self._no_improve} (cur={cur:.4f}, best={self._best:.4f})")
        if self._no_improve >= self.params.patience:
            if self.verbose:
                print("[EarlyStopNoImprove] patience exhausted → request stop")
            self._stop = True

    def _on_step(self) -> bool:  # type: ignore[override]
        # Enforce stop at the next opportunity
        if self._stop:
            return False
        return True

