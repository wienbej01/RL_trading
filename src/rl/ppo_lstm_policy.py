"""
PPO-LSTM policy implementation for the RL trading system.

This module provides the neural network architecture and policy
for the PPO-LSTM reinforcement learning agent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)





class LSTMFeatureExtractor(nn.Module):
    """
    LSTM-based feature extractor for time series data.
    
    This module processes sequential features using LSTM layers
    and extracts meaningful representations for trading decisions.
    """
    
    def __init__(self, feature_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout):
        """
        Initialize LSTM feature extractor.
        
        Args:
            feature_dim: Dimension of the input features
            lstm_hidden_size: Dimension of the LSTM hidden state
            lstm_num_layers: Number of LSTM layers
            lstm_dropout: Dropout rate for LSTM layers
        """
        super().__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(lstm_dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            hidden: Optional hidden state
            
        Returns:
            Output tensor and hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply layer normalization and dropout
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        return lstm_out, hidden


class SharedFeatureExtractor(nn.Module):
    """
    Shared feature extractor for policy and value functions.
    
    This module processes LSTM outputs through shared fully connected layers.
    """
    
    def __init__(self, lstm_hidden_size, net_arch, lstm_dropout):
        """
        Initialize shared feature extractor.
        
        Args:
            lstm_hidden_size: Dimension of the LSTM hidden state
            net_arch: Architecture of the shared network
            lstm_dropout: Dropout rate for LSTM layers
        """
        super().__init__()
        
        # Policy network
        policy_layers = []
        input_dim = lstm_hidden_size
        for hidden_dim in net_arch['pi']:
            policy_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(lstm_dropout)
            ])
            input_dim = hidden_dim
        self.policy_net = nn.Sequential(*policy_layers)

        # Value network
        value_layers = []
        input_dim = lstm_hidden_size
        for hidden_dim in net_arch['vf']:
            value_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(lstm_dropout)
            ])
            input_dim = hidden_dim
        self.value_net = nn.Sequential(*value_layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through shared feature extractor.
        
        Args:
            x: Input tensor from LSTM
            
        Returns:
            Processed features for policy and value functions
        """
        return self.policy_net(x), self.value_net(x)


class ValueHead(nn.Module):
    """
    Value function head for estimating state value.
    
    This module estimates the value of the current state using
    processed features from the shared extractor.
    """
    
    def __init__(self, net_arch, lstm_dropout):
        """
        Initialize value head.
        
        Args:
            net_arch: Architecture of the value network
            lstm_dropout: Dropout rate for LSTM layers
        """
        super().__init__()
        
        # Value head layers
        value_layers = []
        input_dim = net_arch['vf'][-1]
        
        for hidden_dim in net_arch['vf']:
            value_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(lstm_dropout)
            ])
            input_dim = hidden_dim
        
        # Final value layer
        value_layers.append(nn.Linear(input_dim, 1))
        
        self.value_net = nn.Sequential(*value_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.
        
        Args:
            x: Input tensor from shared extractor
            
        Returns:
            Value estimate
        """
        return self.value_net(x)


class PolicyHead(nn.Module):
    """
    Policy head for action probabilities.
    
    This module computes action probabilities using processed
    features from the shared extractor.
    """
    
    def __init__(self, net_arch, lstm_dropout, action_dim):
        """
        Initialize policy head.
        
        Args:
            net_arch: Architecture of the policy network
            lstm_dropout: Dropout rate for LSTM layers
            action_dim: Dimension of the action space
        """
        super().__init__()
        
        # Policy head layers
        policy_layers = []
        input_dim = net_arch['pi'][-1]
        
        for hidden_dim in net_arch['pi']:
            policy_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(lstm_dropout)
            ])
            input_dim = hidden_dim
        
        # Final policy layer
        policy_layers.append(nn.Linear(input_dim, action_dim))
        
        self.policy_net = nn.Sequential(*policy_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy head.
        
        Args:
            x: Input tensor from shared extractor
            
        Returns:
            Action logits
        """
        return self.policy_net(x)


from stable_baselines3.common.policies import ActorCriticPolicy

class PPOLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Default values for LSTM parameters
        self.lstm_hidden_size = kwargs.pop("lstm_hidden_size", 128)
        self.lstm_num_layers = kwargs.pop("lstm_num_layers", 2)
        self.lstm_dropout = kwargs.pop("lstm_dropout", 0.1)
        
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.feature_dim = observation_space.shape[0]
        self.action_dim = action_space.n

        self.lstm_feature_extractor = LSTMFeatureExtractor(self.feature_dim, self.lstm_hidden_size, self.lstm_num_layers, self.lstm_dropout)
        self.shared_net = SharedFeatureExtractor(self.lstm_hidden_size, self.net_arch, self.lstm_dropout)
        self.policy_net = PolicyHead(self.net_arch, self.lstm_dropout, self.action_dim)
        self.value_net = ValueHead(self.net_arch, self.lstm_dropout)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in policy.
        """
        # Pre-process the observation if needed
        features, hidden_state = self.lstm_feature_extractor(obs)
        latent_pi, latent_vf = self.shared_net(features)
        # NOTE: The policy and value heads run on separate features from the shared network.
        # this is common to avoid issues with value estimation affecting policy gradient updates.
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> "Distribution":
        """
        Retrieve action distribution given the latent codes.
        """
        mean_actions = self.policy_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=mean_actions)


