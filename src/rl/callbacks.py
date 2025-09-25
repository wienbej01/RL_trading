from __future__ import annotations

from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class KLStopCallback(BaseCallback):
    """Early stop a training update when approx_kl exceeds target.

    Reads the latest 'train/approx_kl' from the SB3 logger if available.
    """

    def __init__(self, target_kl: float = 0.015, verbose: int = 0):
        super().__init__(verbose)
        self.target_kl = float(target_kl)

    def _on_step(self) -> bool:  # type: ignore[override]
        kl_val: Optional[float] = None
        try:
            logger = getattr(self.model, "logger", None)
            if logger is not None and hasattr(logger, "name_to_value"):
                kl_val = logger.name_to_value.get("train/approx_kl", None)  # type: ignore[attr-defined]
        except Exception:
            kl_val = None
        if kl_val is not None and kl_val > self.target_kl:
            if self.verbose:
                print(f"[KLStopCallback] early stop: approx_kl={kl_val:.5f} > {self.target_kl}")
            return False
        return True


class AdaptiveLRByKL(BaseCallback):
    """Adapt optimizer LR up/down based on observed KL at rollout end.

    - If KL < low: increase LR by 'up' factor (capped by max_lr)
    - If KL > high: decrease LR by 'down' factor (floored by min_lr)
    Writes 'train/adj_lr' to the SB3 logger when an adjustment happens.
    """

    def __init__(
        self,
        low: float = 0.003,
        high: float = 0.015,
        up: float = 1.15,
        down: float = 0.7,
        min_lr: float = 2e-5,
        max_lr: float = 2e-4,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.low, self.high = float(low), float(high)
        self.up, self.down = float(up), float(down)
        self.min_lr, self.max_lr = float(min_lr), float(max_lr)

    def _on_rollout_end(self) -> None:  # type: ignore[override]
        kl_val: Optional[float] = None
        try:
            logger = getattr(self.model, "logger", None)
            if logger is not None and hasattr(logger, "name_to_value"):
                kl_val = logger.name_to_value.get("train/approx_kl", None)  # type: ignore[attr-defined]
        except Exception:
            kl_val = None
        if kl_val is None:
            return
        opt = getattr(self.model.policy, "optimizer", None)
        if opt is None or not getattr(opt, "param_groups", None):
            return
        cur_lr = float(opt.param_groups[0]["lr"])  # type: ignore[index]
        new_lr = cur_lr
        if kl_val < self.low:
            new_lr = min(self.max_lr, cur_lr * self.up)
        elif kl_val > self.high:
            new_lr = max(self.min_lr, cur_lr * self.down)
        if abs(new_lr - cur_lr) / max(cur_lr, 1e-12) > 1e-6:
            for g in opt.param_groups:  # type: ignore[assignment]
                g["lr"] = new_lr
            try:
                self.model.logger.record("train/adj_lr", new_lr)  # type: ignore[union-attr]
            except Exception:
                pass


class LiveLRBump(BaseCallback):
    """One-shot LR bump when a flag file is present in run_dir.

    If a file named '.lr_bump' appears in run_dir, multiply LR once by
    bump_factor and then remove the flag. Useful to nudge LR mid-run.
    """

    def __init__(self, run_dir: str = ".", bump_factor: float = 1.25, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = str(run_dir)
        self.bump_factor = float(bump_factor)
        self.flag_path: Optional[str] = None

    def _on_training_start(self) -> None:  # type: ignore[override]
        import os
        self.flag_path = os.path.join(self.run_dir, ".lr_bump")

    def _on_rollout_end(self) -> None:  # type: ignore[override]
        import os
        if not self.flag_path or not os.path.isfile(self.flag_path):
            return
        opt = getattr(self.model.policy, "optimizer", None)
        if opt and getattr(opt, "param_groups", None):
            for g in opt.param_groups:
                g["lr"] = float(g["lr"]) * self.bump_factor
            if self.verbose:
                print(f"[LiveLRBump] Multiplied LR by {self.bump_factor}")
        try:
            os.remove(self.flag_path)
        except OSError:
            pass

