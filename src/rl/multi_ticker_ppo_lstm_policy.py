"""
Multi-ticker PPO-LSTM policy for RL trading system.

This module implements an enhanced PPO-LSTM policy that supports
multiple tickers with portfolio management and cross-ticker dependencies.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..utils.logging import get_logger

logger = get_logger(__name__)


class MultiTickerFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for multi-ticker observations.
    
    Extracts and processes features from multiple tickers,
    including portfolio-level features.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_tickers: int,
        portfolio_features_dim: int,
        ticker_features_dim: int
    ):
        """
        Initialize the multi-ticker feature extractor.
        
        Args:
            observation_space: Observation space (Dict)
            num_tickers: Number of tickers
            portfolio_features_dim: Dimension of portfolio features
            ticker_features_dim: Dimension of ticker features
        """
        # Calculate total features dimension
        total_features_dim = portfolio_features_dim + num_tickers * ticker_features_dim
        
        super(MultiTickerFeatureExtractor, self).__init__(
            observation_space,
            total_features_dim
        )
        
        self.num_tickers = num_tickers
        self.portfolio_features_dim = portfolio_features_dim
        self.ticker_features_dim = ticker_features_dim
        
        # Portfolio feature extractor
        portfolio_input_dim = observation_space['portfolio'].shape[0]
        self.portfolio_net = nn.Sequential(
            nn.Linear(portfolio_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, portfolio_features_dim),
            nn.Tanh()
        )
        
        # Ticker feature extractor
        ticker_input_dim = observation_space['tickers'].shape[1]
        self.ticker_net = nn.Sequential(
            nn.Linear(ticker_input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, ticker_features_dim),
            nn.Tanh()
        )
        
        # Normalization layers
        self.portfolio_norm = nn.LayerNorm(portfolio_features_dim)
        self.ticker_norm = nn.LayerNorm(ticker_features_dim)
        
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Forward pass of the feature extractor.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Combined feature tensor
        """
        # Extract portfolio features
        portfolio_features = self.portfolio_net(observations['portfolio'])
        portfolio_features = self.portfolio_norm(portfolio_features)
        
        # Extract ticker features
        ticker_features = self.ticker_net(observations['tickers'])
        ticker_features = self.ticker_norm(ticker_features)
        
        # Flatten ticker features
        batch_size = observations['tickers'].shape[0]
        ticker_features = ticker_features.view(batch_size, -1)
        
        # Combine features
        combined_features = th.cat([portfolio_features, ticker_features], dim=1)
        
        return combined_features


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for modeling ticker relationships.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = th.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        output = th.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Project output
        output = self.out_proj(output)
        
        return output


class CurriculumLearningScheduler:
    """
    Scheduler for curriculum learning phases.
    
    Manages transitions between different curriculum phases
    and updates model parameters accordingly.
    """
    
    def __init__(self, phases: List[Dict[str, Any]]):
        """
        Initialize curriculum learning scheduler.
        
        Args:
            phases: List of curriculum phases
        """
        self.phases = phases
        self.current_phase = 0
        self.phase_steps = 0
        self.phase_transitions = []
        
    def update(self, step: int, model: nn.Module) -> bool:
        """
        Update curriculum phase and model parameters.
        
        Args:
            step: Current training step
            model: Model to update
            
        Returns:
            True if phase was changed, False otherwise
        """
        if not self.phases:
            return False
            
        # Check if we should transition to next phase
        if (self.current_phase < len(self.phases) - 1 and 
            step >= self.phases[self.current_phase + 1]['start_step']):
            
            self.current_phase += 1
            self.phase_steps = 0
            
            # Apply phase-specific updates
            phase_config = self.phases[self.current_phase]
            
            # Update model parameters
            if hasattr(model, 'update_curriculum_phase'):
                model.update_curriculum_phase(self.current_phase)
                
            # Update phase-specific parameters
            if 'num_active_tickers' in phase_config and hasattr(model, 'num_active_tickers'):
                model.num_active_tickers = phase_config['num_active_tickers']
                
            if 'entropy_coef' in phase_config and hasattr(model, 'entropy_coef'):
                model.entropy_coef = phase_config['entropy_coef']
                
            if 'learning_rate' in phase_config and hasattr(model, 'optimizer'):
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = phase_config['learning_rate']
                    
            # Record transition
            self.phase_transitions.append({
                'step': step,
                'from_phase': self.current_phase - 1,
                'to_phase': self.current_phase,
                'config': phase_config
            })
            
            return True
            
        self.phase_steps += 1
        return False
        
    def get_current_phase(self) -> Dict[str, Any]:
        """
        Get current curriculum phase configuration.
        
        Returns:
            Current phase configuration
        """
        if not self.phases or self.current_phase >= len(self.phases):
            return {}
            
        return self.phases[self.current_phase]
        
    def get_progress(self) -> float:
        """
        Get progress through current phase.
        
        Returns:
            Progress as a fraction (0.0 to 1.0)
        """
        if not self.phases or self.current_phase >= len(self.phases):
            return 1.0
            
        current_phase = self.phases[self.current_phase]
        phase_duration = current_phase.get('duration', float('inf'))
        
        if phase_duration == float('inf'):
            # Next phase defines duration
            if self.current_phase < len(self.phases) - 1:
                next_phase = self.phases[self.current_phase + 1]
                phase_duration = next_phase['start_step'] - current_phase['start_step']
            else:
                return 0.0
                
        return min(1.0, self.phase_steps / phase_duration)


class MultiTickerPPOLSTMPolicy(ActorCriticPolicy):
    """
    Enhanced PPO-LSTM policy for multi-ticker trading.
    
    Extends the standard PPO-LSTM policy to handle multiple tickers
    with portfolio management and cross-ticker dependencies.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        """
        Initialize the multi-ticker PPO-LSTM policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture
            activation_fn: Activation function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        # Extract multi-ticker specific parameters
        self.num_tickers = kwargs.pop('num_tickers', 1)
        self.use_attention = kwargs.pop('use_attention', True)
        self.use_hierarchical = kwargs.pop('use_hierarchical', True)
        self.portfolio_features_dim = kwargs.pop('portfolio_features_dim', 16)
        self.ticker_features_dim = kwargs.pop('ticker_features_dim', 64)
        self.lstm_hidden_size = kwargs.pop('lstm_hidden_size', 256)
        self.attention_heads = kwargs.pop('attention_heads', 4)
        self.curriculum_phases = kwargs.pop('curriculum_phases', [])
        
        # Initialize curriculum learning
        self.current_phase = 0
        self.phase_steps = 0
        self.phase_transitions = []
        self.curriculum_scheduler = CurriculumLearningScheduler(self.curriculum_phases)
        
        # Initialize entropy annealing
        self.initial_entropy_coef = kwargs.pop('entropy_coef', 0.01)
        self.final_entropy_coef = kwargs.pop('final_entropy_coef', 0.001)
        self.entropy_anneal_steps = kwargs.pop('entropy_anneal_steps', 1000000)
        self.entropy_anneal_type = kwargs.pop('entropy_anneal_type', 'linear')
        
        # Initialize learning rate schedule
        self.lr_schedule_type = kwargs.pop('lr_schedule_type', 'cosine')
        self.warmup_steps = kwargs.pop('warmup_steps', 10000)
        self.total_timesteps = kwargs.pop('total_timesteps', 1000000)
        self.initial_lr = lr_schedule(1.0)
        
        # Initialize parent class
        super(MultiTickerPPOLSTMPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
        )
        
    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Build the multi-ticker PPO-LSTM policy network.
        
        Args:
            lr_schedule: Learning rate schedule
        """
        # Build feature extractor
        self.features_extractor = MultiTickerFeatureExtractor(
            self.observation_space,
            self.num_tickers,
            self.portfolio_features_dim,
            self.ticker_features_dim
        )
        
        # Build LSTM layers
        self.lstm_policy = nn.LSTM(
            input_size=self.features_extractor.features_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.lstm_value = nn.LSTM(
            input_size=self.features_extractor.features_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Build attention mechanism if enabled
        if self.use_attention:
            self.attention = MultiHeadAttention(
                embed_dim=self.lstm_hidden_size,
                num_heads=self.attention_heads,
                dropout=0.1
            )
        
        # Build hierarchical layers if enabled
        if self.use_hierarchical:
            self.portfolio_policy = nn.Sequential(
                nn.Linear(self.lstm_hidden_size, self.portfolio_features_dim),
                self.activation_fn,
                nn.Linear(self.portfolio_features_dim, self.num_tickers)
            )
            
            self.portfolio_value = nn.Sequential(
                nn.Linear(self.lstm_hidden_size, self.portfolio_features_dim),
                self.activation_fn,
                nn.Linear(self.portfolio_features_dim, 1)
            )
        
        # Build policy and value heads
        latent_dim = self.lstm_hidden_size
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            self.activation_fn,
            nn.Linear(latent_dim, self.action_space.shape[0] * self.num_tickers)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            self.activation_fn,
            nn.Linear(latent_dim, 1)
        )
        
        # Setup optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )
        
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass of the multi-ticker PPO-LSTM policy.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Process through LSTM
        lstm_out_policy, _ = self.lstm_policy(features)
        lstm_out_value, _ = self.lstm_value(features)
        
        # Apply attention if enabled
        if self.use_attention:
            lstm_out_policy = self.attention(lstm_out_policy)
            lstm_out_value = self.attention(lstm_out_value)
        
        # Get policy and value outputs
        policy_logits = self.policy_net(lstm_out_policy)
        values = self.value_net(lstm_out_value)
        
        # Apply hierarchical processing if enabled
        if self.use_hierarchical:
            portfolio_weights = self.portfolio_policy(lstm_out_policy)
            portfolio_value = self.portfolio_value(lstm_out_value)
            
            # Combine ticker-level and portfolio-level decisions
            policy_logits = policy_logits * portfolio_weights.unsqueeze(-1)
            values = values + portfolio_value
        
        # Reshape for multi-ticker actions
        batch_size = obs['portfolio'].shape[0]
        policy_logits = policy_logits.view(batch_size, self.num_tickers, -1)
        values = values.view(batch_size, 1)
        
        # Create distribution
        distribution = self.action_dist.proba_distribution(action_logits=policy_logits)
        
        # Sample actions
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        # Calculate log probabilities
        log_probs = distribution.log_prob(actions)
        
        return actions, values, log_probs
        
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Extract features from observations.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Feature tensor
        """
        return self.features_extractor(obs)
        
    def update_learning_rate(self, progress: float) -> None:
        """
        Update learning rate based on schedule.
        
        Args:
            progress: Training progress (0.0 to 1.0)
        """
        if self.lr_schedule_type == 'cosine':
            lr = self.initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.lr_schedule_type == 'linear':
            lr = self.initial_lr * (1.0 - progress)
        elif self.lr_schedule_type == 'warmup_cosine':
            if progress < self.warmup_steps / self.total_timesteps:
                lr = self.initial_lr * (progress / (self.warmup_steps / self.total_timesteps))
            else:
                adjusted_progress = (progress - self.warmup_steps / self.total_timesteps) / (1.0 - self.warmup_steps / self.total_timesteps)
                lr = self.initial_lr * 0.5 * (1.0 + math.cos(math.pi * adjusted_progress))
        else:
            lr = self.lr_schedule(progress)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def update_entropy_coefficient(self, progress: float) -> None:
        """
        Update entropy coefficient based on annealing schedule.
        
        Args:
            progress: Training progress (0.0 to 1.0)
        """
        if self.entropy_anneal_type == 'linear':
            self.entropy_coef = (self.initial_entropy_coef * (1.0 - progress) + 
                                self.final_entropy_coef * progress)
        elif self.entropy_anneal_type == 'cosine':
            self.entropy_coef = (self.initial_entropy_coef * 0.5 * (1.0 + math.cos(math.pi * progress)) + 
                                self.final_entropy_coef * 0.5 * (1.0 - math.cos(math.pi * progress)))
        elif self.entropy_anneal_type == 'exponential':
            self.entropy_coef = self.initial_entropy_coef * math.exp(
                math.log(self.final_entropy_coef / self.initial_entropy_coef) * progress
            )
        else:
            # Default to linear
            self.entropy_coef = (self.initial_entropy_coef * (1.0 - progress) + 
                                self.final_entropy_coef * progress)
        
    def update_curriculum_phase(self, step: int) -> None:
        """
        Update curriculum learning phase.
        
        Args:
            step: Current training step
        """
        phase_changed = self.curriculum_scheduler.update(step, self)
        
        if phase_changed:
            logger.info(f"Curriculum phase changed to {self.curriculum_scheduler.current_phase}")
            
        self.phase_steps = self.curriculum_scheduler.phase_steps
        
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current configuration including curriculum phase.
        
        Returns:
            Current configuration dictionary
        """
        config = {
            'num_tickers': self.num_tickers,
            'use_attention': self.use_attention,
            'use_hierarchical': self.use_hierarchical,
            'current_phase': self.curriculum_scheduler.current_phase,
            'phase_steps': self.phase_steps,
            'entropy_coef': self.entropy_coef,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        if self.curriculum_phases and self.curriculum_scheduler.current_phase < len(self.curriculum_phases):
            config['phase_config'] = self.curriculum_scheduler.get_current_phase()
            
        return config