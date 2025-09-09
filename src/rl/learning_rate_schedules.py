"""
Learning rate schedules for multi-ticker RL trading.

This module implements various learning rate schedules to control
the learning rate during training, adapting to the curriculum learning phases.
"""

import math
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch as th
import torch.optim as optim

from ..utils.logging import get_logger

logger = get_logger(__name__)


class LearningRateSchedule:
    """
    Base class for learning rate schedules.
    
    Provides a framework for implementing different learning rate
    scheduling strategies during training.
    """
    
    def __init__(
        self,
        initial_lr: float,
        final_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        **kwargs
    ):
        """
        Initialize learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            total_steps: Total training steps
            warmup_steps: Warmup steps before scheduling starts
            **kwargs: Additional schedule-specific parameters
        """
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.kwargs = kwargs
        
    def __call__(self, step: int) -> float:
        """
        Calculate learning rate for given step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate for the step
        """
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (step / self.warmup_steps)
            
        return self._calculate(step - self.warmup_steps)
        
    def _calculate(self, step: int) -> float:
        """
        Calculate learning rate after warmup.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Learning rate for the step
        """
        raise NotImplementedError("Subclasses must implement _calculate")
        
    def get_schedule_info(self) -> Dict[str, Union[float, str]]:
        """
        Get information about the schedule.
        
        Returns:
            Dictionary with schedule information
        """
        return {
            'type': self.__class__.__name__,
            'initial_lr': self.initial_lr,
            'final_lr': self.final_lr,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps
        }


class LinearLRSchedule(LearningRateSchedule):
    """
    Linear learning rate schedule.
    
    Linearly interpolates between initial and final learning rates.
    """
    
    def _calculate(self, step: int) -> float:
        """
        Calculate learning rate using linear interpolation.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Learning rate for the step
        """
        progress = min(1.0, step / (self.total_steps - self.warmup_steps))
        return self.initial_lr * (1.0 - progress) + self.final_lr * progress


class CosineLRSchedule(LearningRateSchedule):
    """
    Cosine learning rate schedule.
    
    Uses cosine function for smooth annealing between learning rates.
    """
    
    def _calculate(self, step: int) -> float:
        """
        Calculate learning rate using cosine annealing.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Learning rate for the step
        """
        progress = min(1.0, step / (self.total_steps - self.warmup_steps))
        cosine_progress = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr + (self.initial_lr - self.final_lr) * cosine_progress


class ExponentialLRSchedule(LearningRateSchedule):
    """
    Exponential learning rate schedule.
    
    Uses exponential decay for learning rate scheduling.
    """
    
    def _calculate(self, step: int) -> float:
        """
        Calculate learning rate using exponential decay.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Learning rate for the step
        """
        progress = min(1.0, step / (self.total_steps - self.warmup_steps))
        decay_rate = math.log(self.final_lr / self.initial_lr)
        return self.initial_lr * math.exp(decay_rate * progress)


class StepDecayLRSchedule(LearningRateSchedule):
    """
    Step decay learning rate schedule.
    
    Reduces learning rate in discrete steps at specified intervals.
    """
    
    def __init__(
        self,
        initial_lr: float,
        final_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        decay_steps: int = 100000,
        decay_rate: float = 0.5
    ):
        """
        Initialize step decay learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            total_steps: Total training steps
            warmup_steps: Warmup steps before scheduling starts
            decay_steps: Number of steps between decays
            decay_rate: Multiplicative factor for each decay
        """
        super().__init__(initial_lr, final_lr, total_steps, warmup_steps)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        
    def _calculate(self, step: int) -> float:
        """
        Calculate learning rate using step decay.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Learning rate for the step
        """
        num_decays = step // self.decay_steps
        lr = self.initial_lr * (self.decay_rate ** num_decays)
        return max(self.final_lr, lr)
        
    def get_schedule_info(self) -> Dict[str, Union[float, str]]:
        """
        Get information about the schedule.
        
        Returns:
            Dictionary with schedule information
        """
        info = super().get_schedule_info()
        info.update({
            'decay_steps': self.decay_steps,
            'decay_rate': self.decay_rate
        })
        return info


class OneCycleLRSchedule(LearningRateSchedule):
    """
    One-cycle learning rate schedule.
    
    Implements the 1cycle policy as described in "Super-Convergence".
    """
    
    def __init__(
        self,
        initial_lr: float,
        final_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        max_lr: Optional[float] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos'
    ):
        """
        Initialize one-cycle learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            total_steps: Total training steps
            warmup_steps: Warmup steps before scheduling starts
            max_lr: Maximum learning rate (if None, uses initial_lr)
            pct_start: Percentage of training to increase learning rate
            anneal_strategy: Annealing strategy ('cos' or 'linear')
        """
        super().__init__(initial_lr, final_lr, total_steps, warmup_steps)
        self.max_lr = max_lr or initial_lr * 10
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        
    def _calculate(self, step: int) -> float:
        """
        Calculate learning rate using one-cycle policy.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Learning rate for the step
        """
        effective_steps = self.total_steps - self.warmup_steps
        step_pct = step / effective_steps
        
        if step_pct <= self.pct_start:
            # Increasing phase
            pct = step_pct / self.pct_start
            return self.initial_lr + (self.max_lr - self.initial_lr) * pct
        else:
            # Decreasing phase
            pct = (step_pct - self.pct_start) / (1.0 - self.pct_start)
            
            if self.anneal_strategy == 'cos':
                cosine_progress = 0.5 * (1.0 + math.cos(math.pi * pct))
                return self.final_lr + (self.max_lr - self.final_lr) * cosine_progress
            else:  # linear
                return self.max_lr - (self.max_lr - self.final_lr) * pct
                
    def get_schedule_info(self) -> Dict[str, Union[float, str]]:
        """
        Get information about the schedule.
        
        Returns:
            Dictionary with schedule information
        """
        info = super().get_schedule_info()
        info.update({
            'max_lr': self.max_lr,
            'pct_start': self.pct_start,
            'anneal_strategy': self.anneal_strategy
        })
        return info


class WarmupCosineLRSchedule(LearningRateSchedule):
    """
    Warmup cosine learning rate schedule.
    
    Combines linear warmup with cosine annealing.
    """
    
    def __init__(
        self,
        initial_lr: float,
        final_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        cosine_steps: Optional[int] = None,
        min_lr: Optional[float] = None
    ):
        """
        Initialize warmup cosine learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            total_steps: Total training steps
            warmup_steps: Warmup steps before scheduling starts
            cosine_steps: Steps for cosine annealing (if None, uses total_steps - warmup_steps)
            min_lr: Minimum learning rate for cosine annealing
        """
        super().__init__(initial_lr, final_lr, total_steps, warmup_steps)
        self.cosine_steps = cosine_steps or (total_steps - warmup_steps)
        self.min_lr = min_lr or final_lr
        
    def _calculate(self, step: int) -> float:
        """
        Calculate learning rate using warmup cosine schedule.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Learning rate for the step
        """
        if step >= self.cosine_steps:
            return self.min_lr
            
        progress = step / self.cosine_steps
        cosine_progress = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_progress
        
    def get_schedule_info(self) -> Dict[str, Union[float, str]]:
        """
        Get information about the schedule.
        
        Returns:
            Dictionary with schedule information
        """
        info = super().get_schedule_info()
        info.update({
            'cosine_steps': self.cosine_steps,
            'min_lr': self.min_lr
        })
        return info


class AdaptiveLRSchedule(LearningRateSchedule):
    """
    Adaptive learning rate schedule.
    
    Adjusts learning rate based on training performance metrics.
    """
    
    def __init__(
        self,
        initial_lr: float,
        final_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        factor: float = 0.5,
        patience: int = 10000,
        threshold: float = 1e-4,
        min_lr: Optional[float] = None,
        max_lr: Optional[float] = None,
        cooldown: int = 0,
        mode: str = 'min'
    ):
        """
        Initialize adaptive learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            total_steps: Total training steps
            warmup_steps: Warmup steps before scheduling starts
            factor: Factor by which the learning rate will be reduced
            patience: Number of steps with no improvement after which learning rate will be reduced
            threshold: Threshold for measuring the new optimum
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            cooldown: Number of steps to wait before resuming normal operation after lr reduction
            mode: One of 'min', 'max'. In 'min' mode, lr will be reduced when the quantity monitored has stopped decreasing
        """
        super().__init__(initial_lr, final_lr, total_steps, warmup_steps)
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr or final_lr
        self.max_lr = max_lr or initial_lr
        self.cooldown = cooldown
        self.mode = mode
        
        # Performance tracking
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.current_lr = initial_lr
        self.wait_count = 0
        self.cooldown_counter = 0
        self.num_bad_steps = 0
        
    def update_performance(self, metric: float, step: int) -> bool:
        """
        Update performance metrics and adjust learning rate if needed.
        
        Args:
            metric: Current performance metric
            step: Current training step
            
        Returns:
            True if learning rate was reduced, False otherwise
        """
        if step < self.warmup_steps:
            return False
            
        # Check if we're in cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
            
        # Check if metric improved
        if self.mode == 'min':
            improved = metric < self.best_metric * (1.0 - self.threshold)
        else:
            improved = metric > self.best_metric * (1.0 + self.threshold)
            
        if improved:
            self.best_metric = metric
            self.wait_count = 0
            self.num_bad_steps = 0
        else:
            self.wait_count += 1
            self.num_bad_steps += 1
            
        # Reduce learning rate if patience exceeded
        if self.wait_count >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.min_lr, self.current_lr * self.factor)
            self.cooldown_counter = self.cooldown
            self.wait_count = 0
            
            logger.info(f"Reducing learning rate from {old_lr} to {self.current_lr}")
            return True
            
        return False
        
    def _calculate(self, step: int) -> float:
        """
        Calculate learning rate using adaptive schedule.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Learning rate for the step
        """
        return self.current_lr
        
    def get_schedule_info(self) -> Dict[str, Union[float, str]]:
        """
        Get information about the schedule.
        
        Returns:
            Dictionary with schedule information
        """
        info = super().get_schedule_info()
        info.update({
            'factor': self.factor,
            'patience': self.patience,
            'threshold': self.threshold,
            'current_lr': self.current_lr,
            'best_metric': self.best_metric,
            'wait_count': self.wait_count,
            'cooldown_counter': self.cooldown_counter,
            'num_bad_steps': self.num_bad_steps
        })
        return info


class CurriculumAwareLRSchedule(LearningRateSchedule):
    """
    Curriculum-aware learning rate schedule.
    
    Adjusts learning rate based on curriculum learning phases.
    """
    
    def __init__(
        self,
        initial_lr: float,
        final_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        curriculum_phases: Optional[Dict[int, Dict[str, float]]] = None,
        base_schedule: str = 'cosine'
    ):
        """
        Initialize curriculum-aware learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            total_steps: Total training steps
            warmup_steps: Warmup steps before scheduling starts
            curriculum_phases: Dictionary mapping phase indices to phase configurations
            base_schedule: Base schedule type to use within phases
        """
        super().__init__(initial_lr, final_lr, total_steps, warmup_steps)
        self.curriculum_phases = curriculum_phases or {}
        self.base_schedule = base_schedule
        
        # Create base schedule
        self.base_scheduler = self._create_base_schedule()
        
        # Current phase tracking
        self.current_phase = 0
        self.phase_start_step = 0
        
    def _create_base_schedule(self) -> LearningRateSchedule:
        """
        Create base schedule for within-phase scheduling.
        
        Returns:
            Base schedule instance
        """
        if self.base_schedule == 'linear':
            return LinearLRSchedule(
                self.initial_lr,
                self.final_lr,
                self.total_steps,
                self.warmup_steps
            )
        elif self.base_schedule == 'cosine':
            return CosineLRSchedule(
                self.initial_lr,
                self.final_lr,
                self.total_steps,
                self.warmup_steps
            )
        elif self.base_schedule == 'exponential':
            return ExponentialLRSchedule(
                self.initial_lr,
                self.final_lr,
                self.total_steps,
                self.warmup_steps
            )
        elif self.base_schedule == 'onecycle':
            return OneCycleLRSchedule(
                self.initial_lr,
                self.final_lr,
                self.total_steps,
                self.warmup_steps
            )
        else:
            logger.warning(f"Unknown base schedule type: {self.base_schedule}, using cosine")
            return CosineLRSchedule(
                self.initial_lr,
                self.final_lr,
                self.total_steps,
                self.warmup_steps
            )
            
    def update_curriculum_phase(self, phase: int, phase_start_step: int) -> None:
        """
        Update current curriculum phase.
        
        Args:
            phase: New phase index
            phase_start_step: Step at which the phase started
        """
        self.current_phase = phase
        self.phase_start_step = phase_start_step
        
    def _calculate(self, step: int) -> float:
        """
        Calculate learning rate using curriculum-aware schedule.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Learning rate for the step
        """
        # Get phase-specific configuration
        phase_config = self.curriculum_phases.get(self.current_phase, {})
        
        # Use phase-specific learning rate if available
        if 'learning_rate' in phase_config:
            phase_lr = phase_config['learning_rate']
        else:
            # Use base schedule
            phase_lr = self.base_scheduler(step)
            
        # Apply phase-specific scaling if available
        lr_scale = phase_config.get('lr_scale', 1.0)
        
        return phase_lr * lr_scale
        
    def get_schedule_info(self) -> Dict[str, Union[float, str]]:
        """
        Get information about the schedule.
        
        Returns:
            Dictionary with schedule information
        """
        info = super().get_schedule_info()
        info.update({
            'base_schedule': self.base_schedule,
            'current_phase': self.current_phase,
            'phase_start_step': self.phase_start_step
        })
        return info


class LRSchedulerCallback:
    """
    Callback for integrating learning rate scheduling with training.
    
    This callback can be used with PyTorch optimizers to automatically
    adjust learning rates during training.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        lr_schedule: LearningRateSchedule,
        update_freq: int = 1
    ):
        """
        Initialize learning rate scheduler callback.
        
        Args:
            optimizer: PyTorch optimizer
            lr_schedule: Learning rate schedule
            update_freq: Frequency of learning rate updates (in steps)
        """
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.update_freq = update_freq
        self.current_step = 0
        
    def step(self, step: Optional[int] = None) -> None:
        """
        Update learning rate.
        
        Args:
            step: Current training step (if None, uses internal counter)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if self.current_step % self.update_freq == 0:
            lr = self.lr_schedule(self.current_step)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
    def get_current_lr(self) -> float:
        """
        Get current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.optimizer.param_groups[0]['lr']


def create_lr_schedule(
    schedule_type: str,
    initial_lr: float,
    final_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    **kwargs
) -> LearningRateSchedule:
    """
    Create learning rate schedule of specified type.
    
    Args:
        schedule_type: Type of schedule ('linear', 'cosine', 'exponential', 'step_decay', 'onecycle', 'adaptive', 'curriculum')
        initial_lr: Initial learning rate
        final_lr: Final learning rate
        total_steps: Total training steps
        warmup_steps: Warmup steps before scheduling starts
        **kwargs: Additional schedule-specific parameters
        
    Returns:
        Learning rate schedule instance
    """
    schedule_map = {
        'linear': LinearLRSchedule,
        'cosine': CosineLRSchedule,
        'exponential': ExponentialLRSchedule,
        'step_decay': StepDecayLRSchedule,
        'onecycle': OneCycleLRSchedule,
        'warmup_cosine': WarmupCosineLRSchedule,
        'adaptive': AdaptiveLRSchedule,
        'curriculum': CurriculumAwareLRSchedule
    }
    
    if schedule_type not in schedule_map:
        logger.warning(f"Unknown schedule type: {schedule_type}, using cosine")
        schedule_type = 'cosine'
        
    return schedule_map[schedule_type](
        initial_lr=initial_lr,
        final_lr=final_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        **kwargs
    )