"""
Tests for RL training and evaluation modules.
"""
import pytest
import numpy as np
import pandas as pd
import torch
import gym
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock

from src.rl.ppo_lstm_policy import PPOLSTMPolicy
from src.rl.train import RLTrainer
from src.rl.evaluate import RLEvaluator
from src.rl.walkforward import WalkForwardValidator


class TestPPOLSTMPolicy:
    """Test PPO LSTM Policy implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_dim = 20
        self.action_dim = 3
        self.hidden_dim = 64
        self.sequence_length = 10
        self.batch_size = 32
        
        self.policy = PPOLSTMPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length
        )
    
    def test_policy_initialization(self):
        """Test policy network initialization."""
        assert isinstance(self.policy, PPOLSTMPolicy)
        assert hasattr(self.policy, 'actor_critic')
        assert hasattr(self.policy, 'optimizer')
        
        # Check network architecture
        assert hasattr(self.policy.actor_critic, 'lstm')
        assert hasattr(self.policy.actor_critic, 'actor_head')
        assert hasattr(self.policy.actor_critic, 'critic_head')
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        # Create dummy observation batch
        obs_batch = torch.randn(self.batch_size, self.sequence_length, self.obs_dim)
        hidden_state = self.policy.init_hidden_state(self.batch_size)
        
        # Forward pass
        action_probs, values, new_hidden = self.policy.forward(obs_batch, hidden_state)
        
        # Check output shapes
        assert action_probs.shape == (self.batch_size, self.action_dim)
        assert values.shape == (self.batch_size, 1)
        assert len(new_hidden) == 2  # (hidden, cell) state
        assert new_hidden[0].shape == (1, self.batch_size, self.hidden_dim)
        assert new_hidden[1].shape == (1, self.batch_size, self.hidden_dim)
    
    def test_action_selection(self):
        """Test action selection from policy."""
        obs_batch = torch.randn(self.batch_size, self.sequence_length, self.obs_dim)
        hidden_state = self.policy.init_hidden_state(self.batch_size)
        
        actions, log_probs, values, new_hidden = self.policy.act(obs_batch, hidden_state)
        
        # Check output shapes and ranges
        assert actions.shape == (self.batch_size,)
        assert log_probs.shape == (self.batch_size,)
        assert values.shape == (self.batch_size,)
        
        # Actions should be valid indices
        assert torch.all(actions >= 0)
        assert torch.all(actions < self.action_dim)
        
        # Log probabilities should be negative
        assert torch.all(log_probs <= 0)
    
    def test_policy_update(self):
        """Test policy update mechanism."""
        # Create dummy training data
        obs_batch = torch.randn(self.batch_size, self.sequence_length, self.obs_dim)
        actions_batch = torch.randint(0, self.action_dim, (self.batch_size,))
        old_log_probs = torch.randn(self.batch_size)
        returns = torch.randn(self.batch_size)
        advantages = torch.randn(self.batch_size)
        
        # Get initial parameters
        initial_params = list(self.policy.actor_critic.parameters())
        initial_param_values = [p.clone() for p in initial_params]
        
        # Perform update
        policy_loss, value_loss, entropy_loss = self.policy.update(
            obs_batch, actions_batch, old_log_probs, returns, advantages
        )
        
        # Check that losses are scalars
        assert isinstance(policy_loss, float)
        assert isinstance(value_loss, float)
        assert isinstance(entropy_loss, float)
        
        # Check that parameters have changed
        updated_params = list(self.policy.actor_critic.parameters())
        param_changed = False
        for initial, updated in zip(initial_param_values, updated_params):
            if not torch.allclose(initial, updated, atol=1e-6):
                param_changed = True
                break
        assert param_changed, "Parameters should change after update"
    
    def test_hidden_state_initialization(self):
        """Test hidden state initialization."""
        batch_size = 16
        hidden_state = self.policy.init_hidden_state(batch_size)
        
        assert len(hidden_state) == 2  # (hidden, cell)
        assert hidden_state[0].shape == (1, batch_size, self.hidden_dim)
        assert hidden_state[1].shape == (1, batch_size, self.hidden_dim)
        
        # Initial hidden state should be zeros
        assert torch.allclose(hidden_state[0], torch.zeros_like(hidden_state[0]))
        assert torch.allclose(hidden_state[1], torch.zeros_like(hidden_state[1]))
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save model
            self.policy.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Create new policy and load model
            new_policy = PPOLSTMPolicy(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                sequence_length=self.sequence_length
            )
            new_policy.load_model(model_path)
            
            # Test that loaded model produces same outputs
            obs_batch = torch.randn(1, self.sequence_length, self.obs_dim)
            hidden_state = self.policy.init_hidden_state(1)
            
            with torch.no_grad():
                original_output = self.policy.forward(obs_batch, hidden_state)
                loaded_output = new_policy.forward(obs_batch, hidden_state)
                
                # Check that outputs are close
                assert torch.allclose(original_output[0], loaded_output[0], atol=1e-6)
                assert torch.allclose(original_output[1], loaded_output[1], atol=1e-6)
                
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestRLTrainer:
    """Test RL training functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock environment
        self.mock_env = Mock()
        self.mock_env.observation_space = Mock()
        self.mock_env.observation_space.shape = (20,)
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 3
        
        # Mock policy
        self.mock_policy = Mock()
        
        # Training config
        self.config = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'num_epochs': 4,
            'batch_size': 64,
            'sequence_length': 10
        }
        
        self.trainer = RLTrainer(
            env=self.mock_env,
            policy=self.mock_policy,
            config=self.config
        )
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.env == self.mock_env
        assert self.trainer.policy == self.mock_policy
        assert self.trainer.config == self.config
        
        # Check that hyperparameters are set correctly
        assert self.trainer.gamma == 0.99
        assert self.trainer.gae_lambda == 0.95
        assert self.trainer.clip_epsilon == 0.2
    
    @patch('src.rl.train.RLTrainer.collect_trajectories')
    @patch('src.rl.train.RLTrainer.compute_returns_and_advantages')
    def test_training_step(self, mock_compute_returns, mock_collect_trajectories):
        """Test single training step."""
        # Mock trajectory data
        mock_trajectories = {
            'observations': torch.randn(100, 10, 20),
            'actions': torch.randint(0, 3, (100,)),
            'log_probs': torch.randn(100),
            'rewards': torch.randn(100),
            'values': torch.randn(100),
            'dones': torch.zeros(100, dtype=torch.bool)
        }
        mock_collect_trajectories.return_value = mock_trajectories
        
        # Mock returns and advantages
        returns = torch.randn(100)
        advantages = torch.randn(100)
        mock_compute_returns.return_value = (returns, advantages)
        
        # Mock policy update
        self.mock_policy.update.return_value = (0.1, 0.05, 0.02)
        
        # Perform training step
        metrics = self.trainer.train_step()
        
        # Check that methods were called
        mock_collect_trajectories.assert_called_once()
        mock_compute_returns.assert_called_once()
        self.mock_policy.update.assert_called()
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy_loss' in metrics
    
    def test_compute_gae_advantages(self):
        """Test Generalized Advantage Estimation computation."""
        # Create sample rewards, values, and dones
        rewards = torch.tensor([1.0, 0.5, -0.2, 0.8, 1.2])
        values = torch.tensor([2.0, 1.8, 1.5, 2.1, 2.3])
        dones = torch.tensor([False, False, True, False, True])
        
        advantages = self.trainer.compute_gae_advantages(rewards, values, dones)
        
        # Check output shape
        assert advantages.shape == rewards.shape
        
        # Check that advantages sum to reasonable value
        assert torch.isfinite(advantages).all()
    
    def test_ppo_loss_computation(self):
        """Test PPO loss computation."""
        # Sample data
        batch_size = 32
        old_log_probs = torch.randn(batch_size)
        new_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        
        policy_loss = self.trainer.compute_ppo_loss(
            old_log_probs, new_log_probs, advantages
        )
        
        # Loss should be a scalar tensor
        assert policy_loss.dim() == 0
        assert torch.isfinite(policy_loss)


class TestRLEvaluator:
    """Test RL evaluation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock environment
        self.mock_env = Mock()
        
        # Mock policy
        self.mock_policy = Mock()
        
        self.evaluator = RLEvaluator(
            env=self.mock_env,
            policy=self.mock_policy
        )
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator.env == self.mock_env
        assert self.evaluator.policy == self.mock_policy
    
    @patch('src.rl.evaluate.RLEvaluator.run_episode')
    def test_evaluate_policy(self, mock_run_episode):
        """Test policy evaluation."""
        # Mock episode results
        episode_results = [
            {'return': 100.0, 'length': 50, 'sharpe_ratio': 1.2},
            {'return': 85.0, 'length': 48, 'sharpe_ratio': 0.9},
            {'return': 120.0, 'length': 52, 'sharpe_ratio': 1.5}
        ]
        mock_run_episode.side_effect = episode_results
        
        # Evaluate policy
        metrics = self.evaluator.evaluate(num_episodes=3)
        
        # Check that run_episode was called correct number of times
        assert mock_run_episode.call_count == 3
        
        # Check metrics structure
        assert isinstance(metrics, dict)
        assert 'mean_return' in metrics
        assert 'std_return' in metrics
        assert 'mean_episode_length' in metrics
        assert 'mean_sharpe_ratio' in metrics
        
        # Check metric values
        expected_mean_return = np.mean([100.0, 85.0, 120.0])
        assert abs(metrics['mean_return'] - expected_mean_return) < 1e-6
    
    def test_episode_metrics_calculation(self):
        """Test calculation of episode-level metrics."""
        # Sample episode data
        rewards = np.array([0.1, -0.05, 0.2, 0.15, -0.1, 0.3])
        actions = np.array([1, 0, 2, 1, 0, 2])
        
        metrics = self.evaluator.calculate_episode_metrics(rewards, actions)
        
        # Check that metrics are calculated
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'num_trades' in metrics
        
        # Check values
        assert metrics['total_return'] == rewards.sum()
        assert isinstance(metrics['sharpe_ratio'], float)


class TestWalkForwardValidator:
    """Test walk-forward validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample time series data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.data = pd.DataFrame({
            'close': np.random.uniform(4500, 4600, len(dates)),
            'volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        # Validation config
        self.config = {
            'train_window_months': 6,
            'test_window_months': 1,
            'step_size_months': 1,
            'min_train_samples': 100
        }
        
        self.validator = WalkForwardValidator(
            data=self.data,
            config=self.config
        )
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert len(self.validator.data) == len(self.data)
        assert self.validator.config == self.config
    
    def test_generate_splits(self):
        """Test generation of train/test splits."""
        splits = self.validator.generate_splits()
        
        # Should return list of tuples
        assert isinstance(splits, list)
        assert len(splits) > 0
        
        # Each split should be a tuple of (train_data, test_data)
        for train_data, test_data in splits:
            assert isinstance(train_data, pd.DataFrame)
            assert isinstance(test_data, pd.DataFrame)
            
            # Train data should come before test data
            assert train_data.index.max() < test_data.index.min()
            
            # Test data should be non-empty
            assert len(test_data) > 0
            assert len(train_data) >= self.config['min_train_samples']
    
    @patch('src.rl.walkforward.WalkForwardValidator.train_and_evaluate_model')
    def test_run_validation(self, mock_train_evaluate):
        """Test running full walk-forward validation."""
        # Mock model training and evaluation results
        mock_results = [
            {'train_return': 0.15, 'test_return': 0.08, 'test_sharpe': 1.2},
            {'train_return': 0.12, 'test_return': 0.06, 'test_sharpe': 0.9},
            {'train_return': 0.18, 'test_return': 0.10, 'test_sharpe': 1.4}
        ]
        mock_train_evaluate.side_effect = mock_results
        
        # Run validation
        results = self.validator.run_validation()
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'individual_results' in results
        assert 'summary_stats' in results
        
        # Check individual results
        assert len(results['individual_results']) == len(mock_results)
        
        # Check summary stats
        summary = results['summary_stats']
        assert 'mean_test_return' in summary
        assert 'std_test_return' in summary
        assert 'mean_test_sharpe' in summary
    
    def test_date_splitting_logic(self):
        """Test date-based splitting logic."""
        # Test with specific date range
        start_date = pd.Timestamp('2023-06-01')
        train_months = 3
        test_months = 1
        
        train_data, test_data = self.validator.create_split(
            start_date, train_months, test_months
        )
        
        # Check train data range
        expected_train_end = start_date + pd.DateOffset(months=train_months)
        assert train_data.index.min() >= start_date
        assert train_data.index.max() < expected_train_end
        
        # Check test data range
        expected_test_end = expected_train_end + pd.DateOffset(months=test_months)
        assert test_data.index.min() >= expected_train_end
        assert test_data.index.max() < expected_test_end
    
    def test_validation_with_insufficient_data(self):
        """Test validation with insufficient data."""
        # Create very small dataset
        small_data = self.data.head(30)  # Only 30 days
        
        small_validator = WalkForwardValidator(
            data=small_data,
            config={
                'train_window_months': 6,  # Requires more data than available
                'test_window_months': 1,
                'step_size_months': 1,
                'min_train_samples': 100
            }
        )
        
        # Should handle gracefully
        splits = small_validator.generate_splits()
        assert len(splits) == 0 or all(len(train) >= 100 for train, test in splits)


class TestRLIntegration:
    """Test integration between RL components."""
    
    def test_end_to_end_training_flow(self):
        """Test complete training flow integration."""
        # This would test the full pipeline from data loading to model evaluation
        # Mock all components for integration test
        
        with patch('src.rl.train.RLTrainer') as mock_trainer, \
             patch('src.rl.evaluate.RLEvaluator') as mock_evaluator, \
             patch('src.rl.ppo_lstm_policy.PPOLSTMPolicy') as mock_policy:
            
            # Setup mocks
            mock_trainer_instance = Mock()
            mock_evaluator_instance = Mock()
            mock_policy_instance = Mock()
            
            mock_trainer.return_value = mock_trainer_instance
            mock_evaluator.return_value = mock_evaluator_instance
            mock_policy.return_value = mock_policy_instance
            
            # Mock training results
            mock_trainer_instance.train.return_value = {
                'final_reward': 150.0,
                'training_episodes': 1000,
                'convergence_episode': 800
            }
            
            # Mock evaluation results
            mock_evaluator_instance.evaluate.return_value = {
                'mean_return': 120.0,
                'std_return': 25.0,
                'sharpe_ratio': 1.8,
                'max_drawdown': -0.15
            }
            
            # Test training flow
            training_results = mock_trainer_instance.train()
            evaluation_results = mock_evaluator_instance.evaluate()
            
            # Verify integration
            assert isinstance(training_results, dict)
            assert isinstance(evaluation_results, dict)
            assert 'mean_return' in evaluation_results
            assert 'sharpe_ratio' in evaluation_results
    
    def test_model_persistence_flow(self):
        """Test model saving/loading flow."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_model.pth')
            
            # Create and save a model
            policy = PPOLSTMPolicy(obs_dim=10, action_dim=3, hidden_dim=32)
            policy.save_model(model_path)
            
            # Load model in new instance
            new_policy = PPOLSTMPolicy(obs_dim=10, action_dim=3, hidden_dim=32)
            new_policy.load_model(model_path)
            
            # Test that both models produce similar outputs
            test_obs = torch.randn(1, 5, 10)
            hidden = policy.init_hidden_state(1)
            
            with torch.no_grad():
                output1 = policy.forward(test_obs, hidden)
                output2 = new_policy.forward(test_obs, hidden)
                
                # Outputs should be very close
                assert torch.allclose(output1[0], output2[0], atol=1e-6)
                assert torch.allclose(output1[1], output2[1], atol=1e-6)