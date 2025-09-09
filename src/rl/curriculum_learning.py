"""
Curriculum learning implementation for multi-ticker RL trading.

This module implements curriculum learning phases for gradually increasing
the complexity of multi-ticker trading tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
import torch.nn as nn

from ..utils.logging import get_logger

logger = get_logger(__name__)


class CurriculumPhase:
    """
    Represents a single phase in the curriculum learning process.
    
    Each phase defines specific parameters and constraints for training
    at a particular level of complexity.
    """
    
    def __init__(
        self,
        name: str,
        start_step: int,
        duration: Optional[int] = None,
        num_active_tickers: int = 1,
        max_position_size: float = 1.0,
        max_portfolio_exposure: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 3e-4,
        reward_weights: Optional[Dict[str, float]] = None,
        risk_limits: Optional[Dict[str, float]] = None,
        performance_threshold: Optional[float] = None,
        min_phase_duration: int = 10000,
        max_phase_duration: int = 100000
    ):
        """
        Initialize a curriculum phase.
        
        Args:
            name: Phase name
            start_step: Step at which this phase starts
            duration: Duration of the phase in steps (None for indefinite)
            num_active_tickers: Number of active tickers in this phase
            max_position_size: Maximum position size per ticker
            max_portfolio_exposure: Maximum portfolio exposure
            entropy_coef: Entropy coefficient for exploration
            learning_rate: Learning rate for this phase
            reward_weights: Weights for different reward components
            risk_limits: Risk limits for this phase
            performance_threshold: Performance threshold for phase transition
            min_phase_duration: Minimum duration before phase can be completed
            max_phase_duration: Maximum duration before forced transition
        """
        self.name = name
        self.start_step = start_step
        self.duration = duration
        self.num_active_tickers = num_active_tickers
        self.max_position_size = max_position_size
        self.max_portfolio_exposure = max_portfolio_exposure
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.reward_weights = reward_weights or {}
        self.risk_limits = risk_limits or {}
        self.performance_threshold = performance_threshold
        self.min_phase_duration = min_phase_duration
        self.max_phase_duration = max_phase_duration
        
        # Phase tracking
        self.current_step = 0
        self.performance_history = []
        self.is_completed = False
        
    def update(self, step: int, performance_metrics: Dict[str, float]) -> bool:
        """
        Update phase state and check if phase should be completed.
        
        Args:
            step: Current training step
            performance_metrics: Current performance metrics
            
        Returns:
            True if phase should be completed, False otherwise
        """
        self.current_step = step - self.start_step
        self.performance_history.append(performance_metrics)
        
        # Check minimum duration
        if self.current_step < self.min_phase_duration:
            return False
            
        # Check maximum duration
        if self.current_step >= self.max_phase_duration:
            self.is_completed = True
            return True
            
        # Check performance threshold if specified
        if self.performance_threshold is not None:
            # Calculate recent performance
            recent_window = min(10000, len(self.performance_history))
            recent_performance = self.performance_history[-recent_window:]
            
            # Use Sharpe ratio as primary metric if available
            if 'sharpe_ratio' in performance_metrics:
                avg_sharpe = np.mean([p.get('sharpe_ratio', 0) for p in recent_performance])
                if avg_sharpe >= self.performance_threshold:
                    self.is_completed = True
                    return True
            # Fallback to total return
            elif 'total_return' in performance_metrics:
                avg_return = np.mean([p.get('total_return', 0) for p in recent_performance])
                if avg_return >= self.performance_threshold:
                    self.is_completed = True
                    return True
                    
        # Check duration if specified
        if self.duration is not None and self.current_step >= self.duration:
            self.is_completed = True
            return True
            
        return False
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get phase configuration.
        
        Returns:
            Phase configuration dictionary
        """
        return {
            'name': self.name,
            'num_active_tickers': self.num_active_tickers,
            'max_position_size': self.max_position_size,
            'max_portfolio_exposure': self.max_portfolio_exposure,
            'entropy_coef': self.entropy_coef,
            'learning_rate': self.learning_rate,
            'reward_weights': self.reward_weights,
            'risk_limits': self.risk_limits
        }
        
    def get_progress(self) -> float:
        """
        Get progress through this phase.
        
        Returns:
            Progress as a fraction (0.0 to 1.0)
        """
        if self.duration is not None:
            return min(1.0, self.current_step / self.duration)
        elif self.max_phase_duration > 0:
            return min(1.0, self.current_step / self.max_phase_duration)
        else:
            return 0.0


class CurriculumLearningManager:
    """
    Manages curriculum learning phases and transitions.
    
    Coordinates the progression through different curriculum phases
    and updates model parameters accordingly.
    """
    
    def __init__(self, phases: List[CurriculumPhase]):
        """
        Initialize curriculum learning manager.
        
        Args:
            phases: List of curriculum phases
        """
        self.phases = phases
        self.current_phase_idx = 0
        self.phase_transitions = []
        self.total_steps = 0
        
        # Validate phases
        if not phases:
            raise ValueError("At least one curriculum phase must be provided")
            
        # Sort phases by start step
        self.phases.sort(key=lambda p: p.start_step)
        
        # Validate phase sequence
        for i in range(1, len(phases)):
            if phases[i].start_step <= phases[i-1].start_step:
                raise ValueError(f"Phase {i} start step must be greater than previous phase")
                
    def update(self, step: int, model: nn.Module, performance_metrics: Dict[str, float]) -> bool:
        """
        Update curriculum learning state and model parameters.
        
        Args:
            step: Current training step
            model: Model to update
            performance_metrics: Current performance metrics
            
        Returns:
            True if phase was changed, False otherwise
        """
        self.total_steps = step
        
        # Check if we should transition to next phase
        if self.current_phase_idx < len(self.phases) - 1:
            next_phase = self.phases[self.current_phase_idx + 1]
            
            # Check if it's time to start next phase
            if step >= next_phase.start_step:
                # Check if current phase is ready to be completed
                current_phase = self.phases[self.current_phase_idx]
                if current_phase.update(step, performance_metrics):
                    # Transition to next phase
                    self._transition_to_phase(self.current_phase_idx + 1, model)
                    return True
                    
        # Update current phase
        if self.current_phase_idx < len(self.phases):
            self.phases[self.current_phase_idx].update(step, performance_metrics)
            
        return False
        
    def _transition_to_phase(self, phase_idx: int, model: nn.Module) -> None:
        """
        Transition to a new phase and update model parameters.
        
        Args:
            phase_idx: Index of the new phase
            model: Model to update
        """
        old_phase_idx = self.current_phase_idx
        self.current_phase_idx = phase_idx
        
        # Get new phase configuration
        new_phase = self.phases[phase_idx]
        phase_config = new_phase.get_config()
        
        # Update model parameters
        if hasattr(model, 'update_curriculum_phase'):
            model.update_curriculum_phase(phase_idx)
            
        # Update phase-specific parameters
        if hasattr(model, 'num_active_tickers'):
            model.num_active_tickers = phase_config['num_active_tickers']
            
        if hasattr(model, 'entropy_coef'):
            model.entropy_coef = phase_config['entropy_coef']
            
        if hasattr(model, 'optimizer'):
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = phase_config['learning_rate']
                
        # Record transition
        self.phase_transitions.append({
            'step': self.total_steps,
            'from_phase': old_phase_idx,
            'to_phase': phase_idx,
            'from_phase_name': self.phases[old_phase_idx].name,
            'to_phase_name': new_phase.name,
            'config': phase_config
        })
        
        logger.info(f"Curriculum phase transition: {self.phases[old_phase_idx].name} -> {new_phase.name}")
        
    def get_current_phase(self) -> Optional[CurriculumPhase]:
        """
        Get current curriculum phase.
        
        Returns:
            Current phase or None if no active phase
        """
        if self.current_phase_idx < len(self.phases):
            return self.phases[self.current_phase_idx]
        return None
        
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current curriculum configuration.
        
        Returns:
            Current configuration dictionary
        """
        current_phase = self.get_current_phase()
        if current_phase:
            config = current_phase.get_config()
            config['phase_idx'] = self.current_phase_idx
            config['phase_name'] = current_phase.name
            config['phase_progress'] = current_phase.get_progress()
            return config
        return {}
        
    def get_progress(self) -> float:
        """
        Get overall curriculum progress.
        
        Returns:
            Overall progress as a fraction (0.0 to 1.0)
        """
        if not self.phases:
            return 0.0
            
        # Calculate progress based on current phase and overall phases
        phase_progress = self.current_phase_idx / len(self.phases)
        current_phase = self.get_current_phase()
        
        if current_phase:
            current_phase_progress = current_phase.get_progress()
            return phase_progress + (current_phase_progress / len(self.phases))
            
        return phase_progress
        
    def get_phase_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all phases.
        
        Returns:
            List of phase summaries
        """
        summary = []
        for i, phase in enumerate(self.phases):
            phase_info = {
                'idx': i,
                'name': phase.name,
                'start_step': phase.start_step,
                'duration': phase.duration,
                'is_current': i == self.current_phase_idx,
                'is_completed': phase.is_completed,
                'progress': phase.get_progress(),
                'config': phase.get_config()
            }
            summary.append(phase_info)
            
        return summary


def create_default_curriculum_phases(total_tickers: int, total_steps: int) -> List[CurriculumPhase]:
    """
    Create default curriculum phases for multi-ticker trading.
    
    Args:
        total_tickers: Total number of tickers in the universe
        total_steps: Total training steps
        
    Returns:
        List of curriculum phases
    """
    phases = []
    
    # Phase 1: Single ticker, basic trading
    phases.append(CurriculumPhase(
        name="Single Ticker Basic",
        start_step=0,
        duration=total_steps // 8,
        num_active_tickers=1,
        max_position_size=0.5,
        max_portfolio_exposure=0.5,
        entropy_coef=0.02,
        learning_rate=3e-4,
        performance_threshold=0.5,  # Minimum Sharpe ratio
        min_phase_duration=total_steps // 16
    ))
    
    # Phase 2: Single ticker, advanced trading
    phases.append(CurriculumPhase(
        name="Single Ticker Advanced",
        start_step=total_steps // 8,
        duration=total_steps // 8,
        num_active_tickers=1,
        max_position_size=1.0,
        max_portfolio_exposure=1.0,
        entropy_coef=0.015,
        learning_rate=2.5e-4,
        performance_threshold=0.75,
        min_phase_duration=total_steps // 16
    ))
    
    # Phase 3: Few tickers, portfolio trading
    phases.append(CurriculumPhase(
        name="Few Tickers Portfolio",
        start_step=total_steps // 4,
        duration=total_steps // 4,
        num_active_tickers=min(3, total_tickers),
        max_position_size=0.5,
        max_portfolio_exposure=1.0,
        entropy_coef=0.01,
        learning_rate=2e-4,
        performance_threshold=0.6,
        min_phase_duration=total_steps // 8
    ))
    
    # Phase 4: Multiple tickers, balanced portfolio
    phases.append(CurriculumPhase(
        name="Multiple Tickers Balanced",
        start_step=total_steps // 2,
        duration=total_steps // 4,
        num_active_tickers=min(5, total_tickers),
        max_position_size=0.4,
        max_portfolio_exposure=1.0,
        entropy_coef=0.008,
        learning_rate=1.5e-4,
        performance_threshold=0.5,
        min_phase_duration=total_steps // 8
    ))
    
    # Phase 5: Full universe, optimal portfolio
    phases.append(CurriculumPhase(
        name="Full Universe Optimal",
        start_step=3 * total_steps // 4,
        duration=total_steps // 4,
        num_active_tickers=total_tickers,
        max_position_size=0.3,
        max_portfolio_exposure=1.0,
        entropy_coef=0.005,
        learning_rate=1e-4,
        performance_threshold=0.4,
        min_phase_duration=total_steps // 8
    ))
    
    return phases


class CurriculumLearningCallback:
    """
    Callback for integrating curriculum learning with training.
    
    This callback can be used with stable_baselines3 training
    to automatically manage curriculum learning phases.
    """
    
    def __init__(self, curriculum_manager: CurriculumLearningManager):
        """
        Initialize curriculum learning callback.
        
        Args:
            curriculum_manager: Curriculum learning manager
        """
        self.curriculum_manager = curriculum_manager
        
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            True if training should continue, False to stop
        """
        return True
        
    def _on_rollout_end(self) -> None:
        """
        Called at the end of a rollout.
        """
        pass
        
    def _on_training_start(self) -> None:
        """
        Called before training starts.
        """
        logger.info("Starting curriculum learning training")
        
    def _on_training_end(self) -> None:
        """
        Called after training ends.
        """
        logger.info("Curriculum learning training completed")
        logger.info(f"Total phase transitions: {len(self.curriculum_manager.phase_transitions)}")
        
        # Log phase summary
        phase_summary = self.curriculum_manager.get_phase_summary()
        for phase_info in phase_summary:
            logger.info(f"Phase {phase_info['name']}: "
                       f"Progress={phase_info['progress']:.2f}, "
                       f"Completed={phase_info['is_completed']}")