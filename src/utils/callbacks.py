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


class EntropyCollapseWarning(BaseCallback):
    """
    Callback to monitor policy entropy and log a warning if it collapses.
    """

    def __init__(self, threshold: float = 0.1, patience: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.consecutive_low_entropy = 0

    def _on_step(self) -> bool:
        try:
            entropy = None
            # Stable-Baselines3 exposes helper to retrieve running mean logs when available
            if hasattr(self.logger, "get_mean_log"):
                entropy = self.logger.get_mean_log("train/entropy_loss")
            if entropy is None:
                log_dict = getattr(self.logger, "get_log_dict", lambda: {})()
                value = log_dict.get("train/entropy_loss")
                if isinstance(value, (list, tuple)) and value:
                    entropy = value[-1]
                elif value is not None:
                    entropy = value

            if entropy is not None:
                entropy = float(entropy)
                if entropy < self.threshold:
                    self.consecutive_low_entropy += 1
                else:
                    self.consecutive_low_entropy = 0

                if self.consecutive_low_entropy >= self.patience:
                    if self.verbose > 0:
                        print(
                            f"Warning: Policy entropy has been below {self.threshold} "
                            f"for {self.patience} consecutive steps."
                        )
                    self.consecutive_low_entropy = 0  # Reset after warning
        except Exception:
            # Logger interface can change between SB3 versions; ignore unexpected issues silently
            self.consecutive_low_entropy = 0
        return True
