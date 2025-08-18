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


@dataclass
class PolicyConfig:
    """Policy configuration."""
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    feature_dim: int = 50
    action_dim: int = 3
    shared_layers: List[int] = None
    value_head_layers: List[int] = None
    policy_head_layers: List[int] = None
    
    def __post_init__(self):
        if self.shared_layers is None:
            self.shared_layers = [256, 128]
        if self.value_head_layers is None:
            self.value_head_layers = [128, 64]
        if self.policy_head_layers is None:
            self.policy_head_layers = [128, 64]


class LSTMFeatureExtractor(nn.Module):
    """
    LSTM-based feature extractor for time series data.
    
    This module processes sequential features using LSTM layers
    and extracts meaningful representations for trading decisions.
    """
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize LSTM feature extractor.
        
        Args:
            config: Policy configuration
        """
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.feature_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.lstm_hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.lstm_dropout)
        
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
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize shared feature extractor.
        
        Args:
            config: Policy configuration
        """
        super().__init__()
        self.config = config
        
        # Shared layers
        shared_layers = []
        input_dim = config.lstm_hidden_size
        
        for hidden_dim in config.shared_layers:
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(config.lstm_dropout)
            ])
            input_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*shared_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared feature extractor.
        
        Args:
            x: Input tensor from LSTM
            
        Returns:
            Processed features
        """
        return self.shared_net(x)


class ValueHead(nn.Module):
    """
    Value function head for estimating state value.
    
    This module estimates the value of the current state using
    processed features from the shared extractor.
    """
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize value head.
        
        Args:
            config: Policy configuration
        """
        super().__init__()
        self.config = config
        
        # Value head layers
        value_layers = []
        input_dim = config.shared_layers[-1] if config.shared_layers else config.lstm_hidden_size
        
        for hidden_dim in config.value_head_layers:
            value_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(config.lstm_dropout)
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
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize policy head.
        
        Args:
            config: Policy configuration
        """
        super().__init__()
        self.config = config
        
        # Policy head layers
        policy_layers = []
        input_dim = config.shared_layers[-1] if config.shared_layers else config.lstm_hidden_size
        
        for hidden_dim in config.policy_head_layers:
            policy_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(config.lstm_dropout)
            ])
            input_dim = hidden_dim
        
        # Final policy layer
        policy_layers.append(nn.Linear(input_dim, config.action_dim))
        
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


class PPOLSTMPolicy(nn.Module):
    """
    PPO-LSTM policy for reinforcement learning trading.
    
    This is the complete neural network architecture that combines
    LSTM feature extraction, shared processing, and separate
    policy and value heads.
    """
    
    def __init__(self, config=None, obs_dim=None, action_dim=None, hidden_dim=None, sequence_length=None):
        """
        Initialize PPO-LSTM policy.
        
        Args:
            config: Policy configuration (optional)
            obs_dim: Observation dimension (for test compatibility)
            action_dim: Action dimension (for test compatibility)
            hidden_dim: Hidden dimension (for test compatibility)
            sequence_length: Sequence length (for test compatibility)
        """
        super().__init__()
        
        # Handle test compatibility mode
        if config is None and obs_dim is not None:
            # Create config from individual parameters for test compatibility
            self.config = PolicyConfig(
                feature_dim=obs_dim,
                action_dim=action_dim,
                lstm_hidden_size=hidden_dim,
                lstm_num_layers=1,
                lstm_dropout=0.1,
                shared_layers=[hidden_dim, hidden_dim//2],
                value_head_layers=[hidden_dim//2, hidden_dim//4],
                policy_head_layers=[hidden_dim//2, hidden_dim//4]
            )
        else:
            # Use provided config
            self.config = config if config is not None else PolicyConfig()
        
        # Feature extractor
        self.feature_extractor = LSTMFeatureExtractor(self.config)
        
        # Shared feature extractor
        self.shared_extractor = SharedFeatureExtractor(self.config)
        
        # Value head
        self.value_head = ValueHead(self.config)
        
        # Policy head
        self.policy_head = PolicyHead(self.config)
        
        # Initialize weights
        self._initialize_weights()
        
        # For test compatibility - create actor_critic attribute
        self.actor_critic = self
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the complete policy.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            hidden: Optional hidden state
            
        Returns:
            Action logits, value estimates, and hidden state
        """
        # Feature extraction
        lstm_out, hidden = self.feature_extractor(x, hidden)
        
        # Use last time step for decision making
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Shared processing
        shared_features = self.shared_extractor(last_hidden)
        
        # Value estimation
        value = self.value_head(shared_features)
        
        # Policy computation
        logits = self.policy_head(shared_features)
        
        return logits, value, hidden
    
    def get_initial_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get initial hidden state for LSTM.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Initial hidden state (h_0, c_0)
        """
        h_0 = torch.zeros(
            self.config.lstm_num_layers, 
            batch_size, 
            self.config.lstm_hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.config.lstm_num_layers, 
            batch_size, 
            self.config.lstm_hidden_size,
            device=device
        )
        
        return h_0, c_0
    
    def repackage_hidden(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Repackage hidden state to detach from computation graph.
        
        Args:
            hidden: Hidden state tuple
            
        Returns:
            Repacked hidden state
        """
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in hidden)
    
    def get_action(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action and value estimate.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state
            
        Returns:
            Action, value estimate, and hidden state
        """
        with torch.no_grad():
            logits, value, hidden = self.forward(x, hidden)
            action = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            return action, value, hidden
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions and compute log probabilities and entropy.
        
        Args:
            x: Input tensor
            actions: Actions to evaluate
            hidden: Optional hidden state
            
        Returns:
            Log probabilities, entropy, and value estimates
        """
        logits, value, hidden = self.forward(x, hidden)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions)
        
        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_probs, entropy, value
    
    def init_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM (for test compatibility).
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial hidden state
        """
        return self.get_initial_hidden(batch_size, next(self.parameters()).device)
    
    def act(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action, log probabilities, and value estimate (for test compatibility).
        
        Args:
            x: Input tensor
            hidden: Optional hidden state
            
        Returns:
            Action, log probabilities, value estimate, and hidden state
        """
        with torch.no_grad():
            logits, value, hidden = self.forward(x, hidden)
            
            # Sample action
            action_probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(action_probs, num_samples=1).squeeze()
            
            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze()
            
            return action, action_log_probs, value.squeeze(), hidden
    
    def update(self, obs_batch: torch.Tensor, actions_batch: torch.Tensor, old_log_probs: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor) -> Tuple[float, float, float]:
        """
        Update policy using PPO (for test compatibility).
        
        Args:
            obs_batch: Observation batch
            actions_batch: Action batch
            old_log_probs: Old log probabilities
            returns: Returns
            advantages: Advantages
            
        Returns:
            Policy loss, value loss, entropy loss
        """
        # Forward pass
        log_probs, entropy, values = self.evaluate_actions(obs_batch, actions_batch)
        
        # Compute ratio
        ratio = torch.exp(log_probs.squeeze() - old_log_probs)
        
        # PPO loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def save_model(self, filepath: str):
        """
        Save model to file (for test compatibility).
        
        Args:
            filepath: Path to save file
        """
        torch.save({
            'policy_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file (for test compatibility).
        
        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        
        logger.info(f"Model loaded from {filepath}")


class PPOLSTMPolicyWrapper:
    """
    Wrapper for PPO-LSTM policy with training utilities.
    
    This class provides additional functionality for training
    and managing the PPO-LSTM policy.
    """
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize policy wrapper.
        
        Args:
            config: Policy configuration
        """
        self.config = config
        self.policy = PPOLSTMPolicy(config)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate if hasattr(config, 'learning_rate') else 3e-4
        )
        
        # Training state
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def save(self, filepath: str):
        """
        Save policy to file.
        
        Args:
            filepath: Path to save file
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, filepath)
        
        logger.info(f"Policy saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load policy from file.
        
        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.global_step = checkpoint['global_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        
        logger.info(f"Policy loaded from {filepath}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Training metrics
        """
        # Extract batch data
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        log_probs_old = batch['log_probs_old']
        values_old = batch['values_old']
        
        # Forward pass
        log_probs, entropy, values = self.policy.evaluate_actions(states, actions)
        
        # Compute advantages
        advantages = rewards - values_old
        returns = advantages + values_old
        
        # Policy loss
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus
        entropy_bonus = entropy.mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        # Update global step
        self.global_step += 1
        
        # Log metrics
        metrics = {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_bonus.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def update_learning_rate(self, new_lr: float):
        """
        Update learning rate.
        
        Args:
            new_lr: New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        logger.info(f"Learning rate updated to {new_lr}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Training statistics
        """
        if not self.episode_rewards:
            return {}
        
        stats = {
            'global_step': self.global_step,
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards[-100:]),
            'avg_length': np.mean(self.episode_lengths[-100:]),
            'max_reward': max(self.episode_rewards),
            'min_reward': min(self.episode_rewards)
        }
        
        return stats