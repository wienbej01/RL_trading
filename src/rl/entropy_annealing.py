"""
Entropy annealing schedules for multi-ticker RL trading.

This module implements various entropy annealing schedules to control
exploration during training, adapting to the curriculum learning phases.
"""

import math
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch as th

from ..utils.logging import get_logger

logger = get_logger(__name__)


class EntropyAnnealingSchedule:
    """
    Base class for entropy annealing schedules.
    
    Provides a framework for implementing different entropy
    annealing strategies during training.
    """
    
    def __init__(
        self,
        initial_entropy_coef: float,
        final_entropy_coef: float,
        total_steps: int,
        warmup_steps: int = 0,
        **kwargs
    ):
        """
        Initialize entropy annealing schedule.
        
        Args:
            initial_entropy_coef: Initial entropy coefficient
            final_entropy_coef: Final entropy coefficient
            total_steps: Total training steps
            warmup_steps: Warmup steps before annealing starts
            **kwargs: Additional schedule-specific parameters
        """
        self.initial_entropy_coef = initial_entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.kwargs = kwargs
        
    def __call__(self, step: int) -> float:
        """
        Calculate entropy coefficient for given step.
        
        Args:
            step: Current training step
            
        Returns:
            Entropy coefficient for the step
        """
        if step < self.warmup_steps:
            return self.initial_entropy_coef
            
        return self._calculate(step - self.warmup_steps)
        
    def _calculate(self, step: int) -> float:
        """
        Calculate entropy coefficient after warmup.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Entropy coefficient for the step
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
            'initial_entropy_coef': self.initial_entropy_coef,
            'final_entropy_coef': self.final_entropy_coef,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps
        }


class LinearEntropySchedule(EntropyAnnealingSchedule):
    """
    Linear entropy annealing schedule.
    
    Linearly interpolates between initial and final entropy coefficients.
    """
    
    def _calculate(self, step: int) -> float:
        """
        Calculate entropy coefficient using linear interpolation.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Entropy coefficient for the step
        """
        progress = min(1.0, step / (self.total_steps - self.warmup_steps))
        return self.initial_entropy_coef * (1.0 - progress) + self.final_entropy_coef * progress


class CosineEntropySchedule(EntropyAnnealingSchedule):
    """
    Cosine entropy annealing schedule.
    
    Uses cosine function for smooth annealing between entropy coefficients.
    """
    
    def _calculate(self, step: int) -> float:
        """
        Calculate entropy coefficient using cosine annealing.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Entropy coefficient for the step
        """
        progress = min(1.0, step / (self.total_steps - self.warmup_steps))
        cosine_progress = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_entropy_coef + (self.initial_entropy_coef - self.final_entropy_coef) * cosine_progress


class ExponentialEntropySchedule(EntropyAnnealingSchedule):
    """
    Exponential entropy annealing schedule.
    
    Uses exponential decay for entropy annealing.
    """
    
    def _calculate(self, step: int) -> float:
        """
        Calculate entropy coefficient using exponential decay.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Entropy coefficient for the step
        """
        progress = min(1.0, step / (self.total_steps - self.warmup_steps))
        decay_rate = math.log(self.final_entropy_coef / self.initial_entropy_coef)
        return self.initial_entropy_coef * math.exp(decay_rate * progress)


class StepDecayEntropySchedule(EntropyAnnealingSchedule):
    """
    Step decay entropy annealing schedule.
    
    Reduces entropy coefficient in discrete steps at specified intervals.
    """
    
    def __init__(
        self,
        initial_entropy_coef: float,
        final_entropy_coef: float,
        total_steps: int,
        warmup_steps: int = 0,
        decay_steps: int = 100000,
        decay_rate: float = 0.5
    ):
        """
        Initialize step decay entropy schedule.
        
        Args:
            initial_entropy_coef: Initial entropy coefficient
            final_entropy_coef: Final entropy coefficient
            total_steps: Total training steps
            warmup_steps: Warmup steps before annealing starts
            decay_steps: Number of steps between decays
            decay_rate: Multiplicative factor for each decay
        """
        super().__init__(initial_entropy_coef, final_entropy_coef, total_steps, warmup_steps)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        
    def _calculate(self, step: int) -> float:
        """
        Calculate entropy coefficient using step decay.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Entropy coefficient for the step
        """
        num_decays = step // self.decay_steps
        coef = self.initial_entropy_coef * (self.decay_rate ** num_decays)
        return max(self.final_entropy_coef, coef)
        
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


class AdaptiveEntropySchedule(EntropyAnnealingSchedule):
    """
    Adaptive entropy annealing schedule.
    
    Adjusts entropy coefficient based on training performance metrics.
    """
    
    def __init__(
        self,
        initial_entropy_coef: float,
        final_entropy_coef: float,
        total_steps: int,
        warmup_steps: int = 0,
        target_entropy: Optional[float] = None,
        adaptation_rate: float = 0.01,
        window_size: int = 10000,
        min_entropy: Optional[float] = None,
        max_entropy: Optional[float] = None
    ):
        """
        Initialize adaptive entropy schedule.
        
        Args:
            initial_entropy_coef: Initial entropy coefficient
            final_entropy_coef: Final entropy coefficient
            total_steps: Total training steps
            warmup_steps: Warmup steps before annealing starts
            target_entropy: Target entropy value for adaptation
            adaptation_rate: Rate of adaptation
            window_size: Window size for performance tracking
            min_entropy: Minimum entropy coefficient
            max_entropy: Maximum entropy coefficient
        """
        super().__init__(initial_entropy_coef, final_entropy_coef, total_steps, warmup_steps)
        self.target_entropy = target_entropy
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        self.min_entropy = min_entropy or final_entropy_coef
        self.max_entropy = max_entropy or initial_entropy_coef
        
        # Performance tracking
        self.entropy_history = []
        self.current_entropy_coef = initial_entropy_coef
        
    def update_performance(self, entropy: float, step: int) -> None:
        """
        Update performance metrics for adaptation.
        
        Args:
            entropy: Current entropy value
            step: Current training step
        """
        self.entropy_history.append((step, entropy))
        
        # Keep only recent history
        cutoff_step = max(0, step - self.window_size)
        self.entropy_history = [(s, e) for s, e in self.entropy_history if s >= cutoff_step]
        
    def _calculate(self, step: int) -> float:
        """
        Calculate entropy coefficient using adaptive schedule.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Entropy coefficient for the step
        """
        if not self.entropy_history or self.target_entropy is None:
            # Fall back to linear schedule if no history or target
            progress = min(1.0, step / (self.total_steps - self.warmup_steps))
            return self.initial_entropy_coef * (1.0 - progress) + self.final_entropy_coef * progress
            
        # Calculate average recent entropy
        recent_entropy = np.mean([e for _, e in self.entropy_history])
        
        # Adapt based on difference from target
        entropy_diff = self.target_entropy - recent_entropy
        
        # Update current entropy coefficient
        self.current_entropy_coef += self.adaptation_rate * entropy_diff
        
        # Apply bounds
        self.current_entropy_coef = np.clip(
            self.current_entropy_coef,
            self.min_entropy,
            self.max_entropy
        )
        
        return self.current_entropy_coef
        
    def get_schedule_info(self) -> Dict[str, Union[float, str]]:
        """
        Get information about the schedule.
        
        Returns:
            Dictionary with schedule information
        """
        info = super().get_schedule_info()
        info.update({
            'target_entropy': self.target_entropy,
            'adaptation_rate': self.adaptation_rate,
            'window_size': self.window_size,
            'current_entropy_coef': self.current_entropy_coef
        })
        return info


class CurriculumAwareEntropySchedule(EntropyAnnealingSchedule):
    """
    Curriculum-aware entropy annealing schedule.
    
    Adjusts entropy coefficient based on curriculum learning phases.
    """
    
    def __init__(
        self,
        initial_entropy_coef: float,
        final_entropy_coef: float,
        total_steps: int,
        warmup_steps: int = 0,
        curriculum_phases: Optional[Dict[int, Dict[str, float]]] = None,
        base_schedule: str = 'cosine'
    ):
        """
        Initialize curriculum-aware entropy schedule.
        
        Args:
            initial_entropy_coef: Initial entropy coefficient
            final_entropy_coef: Final entropy coefficient
            total_steps: Total training steps
            warmup_steps: Warmup steps before annealing starts
            curriculum_phases: Dictionary mapping phase indices to phase configurations
            base_schedule: Base schedule type to use within phases
        """
        super().__init__(initial_entropy_coef, final_entropy_coef, total_steps, warmup_steps)
        self.curriculum_phases = curriculum_phases or {}
        self.base_schedule = base_schedule
        
        # Create base schedule
        self.base_scheduler = self._create_base_schedule()
        
        # Current phase tracking
        self.current_phase = 0
        self.phase_start_step = 0
        
    def _create_base_schedule(self) -> EntropyAnnealingSchedule:
        """
        Create base schedule for within-phase annealing.
        
        Returns:
            Base schedule instance
        """
        if self.base_schedule == 'linear':
            return LinearEntropySchedule(
                self.initial_entropy_coef,
                self.final_entropy_coef,
                self.total_steps,
                self.warmup_steps
            )
        elif self.base_schedule == 'cosine':
            return CosineEntropySchedule(
                self.initial_entropy_coef,
                self.final_entropy_coef,
                self.total_steps,
                self.warmup_steps
            )
        elif self.base_schedule == 'exponential':
            return ExponentialEntropySchedule(
                self.initial_entropy_coef,
                self.final_entropy_coef,
                self.total_steps,
                self.warmup_steps
            )
        else:
            logger.warning(f"Unknown base schedule type: {self.base_schedule}, using cosine")
            return CosineEntropySchedule(
                self.initial_entropy_coef,
                self.final_entropy_coef,
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
        Calculate entropy coefficient using curriculum-aware schedule.
        
        Args:
            step: Current training step (after warmup)
            
        Returns:
            Entropy coefficient for the step
        """
        # Get phase-specific configuration
        phase_config = self.curriculum_phases.get(self.current_phase, {})
        
        # Use phase-specific entropy coefficient if available
        if 'entropy_coef' in phase_config:
            phase_entropy = phase_config['entropy_coef']
        else:
            # Use base schedule
            phase_entropy = self.base_scheduler(step)
            
        # Apply phase-specific scaling if available
        entropy_scale = phase_config.get('entropy_scale', 1.0)
        
        return phase_entropy * entropy_scale
        
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


def create_entropy_schedule(
    schedule_type: str,
    initial_entropy_coef: float,
    final_entropy_coef: float,
    total_steps: int,
    warmup_steps: int = 0,
    **kwargs
) -> EntropyAnnealingSchedule:
    """
    Create entropy annealing schedule of specified type.
    
    Args:
        schedule_type: Type of schedule ('linear', 'cosine', 'exponential', 'step_decay', 'adaptive', 'curriculum')
        initial_entropy_coef: Initial entropy coefficient
        final_entropy_coef: Final entropy coefficient
        total_steps: Total training steps
        warmup_steps: Warmup steps before annealing starts
        **kwargs: Additional schedule-specific parameters
        
    Returns:
        Entropy annealing schedule instance
    """
    schedule_map = {
        'linear': LinearEntropySchedule,
        'cosine': CosineEntropySchedule,
        'exponential': ExponentialEntropySchedule,
        'step_decay': StepDecayEntropySchedule,
        'adaptive': AdaptiveEntropySchedule,
        'curriculum': CurriculumAwareEntropySchedule
    }
    
    if schedule_type not in schedule_map:
        logger.warning(f"Unknown schedule type: {schedule_type}, using cosine")
        schedule_type = 'cosine'
        
    return schedule_map[schedule_type](
        initial_entropy_coef=initial_entropy_coef,
        final_entropy_coef=final_entropy_coef,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        **kwargs
    )