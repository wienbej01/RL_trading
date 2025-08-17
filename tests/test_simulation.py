"""
Tests for simulation modules (environment, execution, risk management).
"""
import pytest
import numpy as np
import pandas as pd
import gym
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock

from src.sim.env_intraday_rl import IntradayRLEnvironment
from src.sim.execution import ExecutionEngine, Order, Position
from src.sim.risk import RiskManager, RiskMetrics


class TestIntradayRLEnvironment:
    """Test intraday RL environment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample market data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01 09:30:00', '2023-01-01 16:00:00', 
                             freq='1min', tz='America/New_York')
        
        base_price = 4500
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price + np.cumsum(returns * base_price)
        
        self.market_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.5, len(dates)),
            'high': prices + np.abs(np.random.normal(0, 1.0, len(dates))),
            'low': prices - np.abs(np.random.normal(0, 1.0, len(dates))),
            'close': prices,
            'volume': np.random.randint(100, 1000, len(dates)),
            'bid_price': prices - 0.125,
            'ask_price': prices + 0.125,
            'bid_size': np.random.randint(100, 1000, len(dates)),
            'ask_size': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure OHLC constraints
        self.market_data['high'] = np.maximum(
            self.market_data['high'],
            np.maximum(self.market_data['open'], self.market_data['close'])
        )
        self.market_data['low'] = np.minimum(
            self.market_data['low'],
            np.minimum(self.market_data['open'], self.market_data['close'])
        )
        
        # Environment config
        self.config = {
            'initial_cash': 100000.0,
            'max_position_size': 5,
            'transaction_cost': 2.5,
            'slippage': 0.01,
            'lookback_window': 20,
            'triple_barrier': {
                'profit_target': 0.01,
                'stop_loss': 0.005,
                'time_limit': 60  # minutes
            },
            'features': {
                'technical': ['sma_10', 'rsi_14', 'atr_14'],
                'microstructure': ['spread', 'imbalance'],
                'time': ['hour', 'minute', 'time_to_close']
            }
        }
        
        self.env = IntradayRLEnvironment(
            market_data=self.market_data,
            config=self.config
        )
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        assert isinstance(self.env, gym.Env)
        assert hasattr(self.env, 'observation_space')
        assert hasattr(self.env, 'action_space')
        
        # Check action space
        assert self.env.action_space.n == 3  # [hold, buy, sell]
        
        # Check observation space shape
        expected_obs_dim = len(self.config['features']['technical']) + \
                          len(self.config['features']['microstructure']) + \
                          len(self.config['features']['time'])
        assert self.env.observation_space.shape[0] == expected_obs_dim
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        obs = self.env.reset()
        
        # Check observation shape
        assert obs.shape == self.env.observation_space.shape
        
        # Check that environment state is reset
        assert self.env.current_step == 0
        assert self.env.cash == self.config['initial_cash']
        assert self.env.position == 0
        assert len(self.env.trade_history) == 0
    
    def test_environment_step(self):
        """Test environment step functionality."""
        self.env.reset()
        
        # Take a step
        action = 1  # Buy
        obs, reward, done, info = self.env.step(action)
        
        # Check output types and shapes
        assert obs.shape == self.env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check that environment state updated
        assert self.env.current_step == 1
        
        # Check info dictionary
        expected_info_keys = ['position', 'cash', 'portfolio_value', 'unrealized_pnl']
        for key in expected_info_keys:
            assert key in info
    
    def test_buy_action(self):
        """Test buy action execution."""
        self.env.reset()
        initial_cash = self.env.cash
        
        # Execute buy action
        obs, reward, done, info = self.env.step(1)  # Buy
        
        # Position should increase
        assert self.env.position > 0
        
        # Cash should decrease (assuming successful trade)
        if self.env.position > 0:  # Trade executed
            assert self.env.cash < initial_cash
        
        # Check trade is recorded
        if self.env.position > 0:
            assert len(self.env.trade_history) > 0
    
    def test_sell_action(self):
        """Test sell action execution."""
        self.env.reset()
        
        # First buy to establish position
        self.env.step(1)  # Buy
        initial_position = self.env.position
        initial_cash = self.env.cash
        
        if initial_position > 0:  # Buy was successful
            # Execute sell action
            obs, reward, done, info = self.env.step(2)  # Sell
            
            # Position should decrease or become negative
            assert self.env.position < initial_position
            
            # Cash should change based on trade
            assert self.env.cash != initial_cash
    
    def test_hold_action(self):
        """Test hold action execution."""
        self.env.reset()
        initial_state = {
            'position': self.env.position,
            'cash': self.env.cash,
            'step': self.env.current_step
        }
        
        # Execute hold action
        obs, reward, done, info = self.env.step(0)  # Hold
        
        # Position and cash should remain the same
        assert self.env.position == initial_state['position']
        assert self.env.cash == initial_state['cash']
        
        # Step should increment
        assert self.env.current_step == initial_state['step'] + 1
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        self.env.reset()
        
        # Run until end of data
        done = False
        steps = 0
        max_steps = len(self.market_data) - self.config['lookback_window'] - 1
        
        while not done and steps < max_steps:
            obs, reward, done, info = self.env.step(0)  # Hold
            steps += 1
        
        # Should terminate at end of data
        assert done
        assert steps <= max_steps
    
    def test_triple_barrier_exit(self):
        """Test triple barrier exit mechanism."""
        self.env.reset()
        
        # Execute buy to create position
        self.env.step(1)  # Buy
        
        if self.env.position > 0:  # Trade executed
            entry_price = self.env.entry_price
            profit_target = entry_price * (1 + self.config['triple_barrier']['profit_target'])
            stop_loss = entry_price * (1 - self.config['triple_barrier']['stop_loss'])
            
            # Simulate price movement that hits profit target
            # This would require mocking or manipulating market data
            # For now, just check that the logic exists
            assert hasattr(self.env, 'check_triple_barrier')
            assert hasattr(self.env, 'entry_price')
            assert hasattr(self.env, 'entry_time')
    
    def test_transaction_costs(self):
        """Test transaction cost calculation."""
        self.env.reset()
        initial_cash = self.env.cash
        
        # Execute trade
        self.env.step(1)  # Buy
        
        if self.env.position > 0:  # Trade executed
            # Cash should decrease by more than just the position value
            position_value = self.env.position * self.env.current_price
            cash_decrease = initial_cash - self.env.cash
            
            # Should include transaction costs
            assert cash_decrease > position_value
    
    def test_reward_calculation(self):
        """Test reward calculation mechanism."""
        self.env.reset()
        
        # Execute trade and get reward
        obs, reward, done, info = self.env.step(1)  # Buy
        
        # Reward should be finite
        assert np.isfinite(reward)
        
        # Test different reward scenarios
        if self.env.position != 0:
            # Should have some reward based on price movement
            assert isinstance(reward, float)
    
    def test_eod_flatten(self):
        """Test end-of-day position flattening."""
        self.env.reset()
        
        # Create position
        self.env.step(1)  # Buy
        
        # Fast forward to near end of day
        eod_step = len(self.market_data) - self.config['lookback_window'] - 2
        self.env.current_step = eod_step
        
        # Take final step
        obs, reward, done, info = self.env.step(0)  # Hold
        
        # Position should be flattened at end of day
        if done:
            assert self.env.position == 0


class TestExecutionEngine:
    """Test execution engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'slippage_model': 'linear',
            'transaction_cost': 2.5,
            'max_position_size': 10,
            'fill_probability': 0.95
        }
        
        self.execution_engine = ExecutionEngine(config=self.config)
    
    def test_execution_engine_initialization(self):
        """Test execution engine initialization."""
        assert self.execution_engine.config == self.config
        assert hasattr(self.execution_engine, 'pending_orders')
        assert hasattr(self.execution_engine, 'filled_orders')
    
    def test_order_creation(self):
        """Test order creation."""
        order = Order(
            symbol='MES',
            side='BUY',
            quantity=1,
            order_type='MARKET',
            timestamp=datetime.now()
        )
        
        assert order.symbol == 'MES'
        assert order.side == 'BUY'
        assert order.quantity == 1
        assert order.order_type == 'MARKET'
        assert order.status == 'PENDING'
    
    def test_market_order_execution(self):
        """Test market order execution."""
        order = Order(
            symbol='MES',
            side='BUY',
            quantity=1,
            order_type='MARKET',
            timestamp=datetime.now()
        )
        
        market_data = {
            'bid_price': 4500.0,
            'ask_price': 4500.5,
            'bid_size': 100,
            'ask_size': 100
        }
        
        fill = self.execution_engine.execute_order(order, market_data)
        
        if fill is not None:  # Order was filled
            assert fill['symbol'] == 'MES'
            assert fill['quantity'] == 1
            assert fill['side'] == 'BUY'
            assert 'fill_price' in fill
            assert 'fill_time' in fill
    
    def test_limit_order_execution(self):
        """Test limit order execution."""
        order = Order(
            symbol='MES',
            side='BUY',
            quantity=1,
            order_type='LIMIT',
            limit_price=4499.0,
            timestamp=datetime.now()
        )
        
        # Market data where limit order should fill
        market_data = {
            'bid_price': 4498.0,
            'ask_price': 4499.5,
            'bid_size': 100,
            'ask_size': 100
        }
        
        fill = self.execution_engine.execute_order(order, market_data)
        
        if fill is not None:
            assert fill['fill_price'] <= order.limit_price  # Buy limit
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        base_price = 4500.0
        quantity = 1
        side = 'BUY'
        
        slipped_price = self.execution_engine.apply_slippage(base_price, quantity, side)
        
        # Slippage should affect price adversely
        if side == 'BUY':
            assert slipped_price >= base_price
        else:  # SELL
            assert slipped_price <= base_price
    
    def test_transaction_cost_calculation(self):
        """Test transaction cost calculation."""
        fill_value = 4500.0 * 1  # Price * quantity
        
        cost = self.execution_engine.calculate_transaction_cost(fill_value)
        
        assert cost >= 0
        assert isinstance(cost, float)
    
    def test_position_tracking(self):
        """Test position tracking."""
        position = Position(symbol='MES')
        
        # Add long position
        position.add_fill(1, 4500.0, datetime.now())
        assert position.quantity == 1
        assert position.avg_price == 4500.0
        
        # Add more long
        position.add_fill(1, 4510.0, datetime.now())
        assert position.quantity == 2
        assert position.avg_price == 4505.0  # Average
        
        # Reduce position
        position.add_fill(-1, 4520.0, datetime.now())
        assert position.quantity == 1
        
        # Close position
        position.add_fill(-1, 4525.0, datetime.now())
        assert position.quantity == 0


class TestRiskManager:
    """Test risk management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'max_position_size': 5,
            'max_portfolio_value': 200000,
            'max_daily_loss': -5000,
            'max_drawdown': -0.10,
            'position_concentration_limit': 0.5,
            'var_limit': -2000,
            'leverage_limit': 2.0
        }
        
        self.risk_manager = RiskManager(config=self.config)
        
        # Sample portfolio state
        self.portfolio_state = {
            'cash': 100000,
            'positions': {'MES': 2},
            'portfolio_value': 110000,
            'daily_pnl': -1000,
            'unrealized_pnl': 2000,
            'current_drawdown': -0.05
        }
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        assert self.risk_manager.config == self.config
        assert hasattr(self.risk_manager, 'risk_metrics')
    
    def test_position_size_limit(self):
        """Test position size limit check."""
        # Test within limit
        result = self.risk_manager.check_position_size_limit('MES', 3)
        assert result['allowed'] == True
        
        # Test exceeding limit
        result = self.risk_manager.check_position_size_limit('MES', 10)
        assert result['allowed'] == False
        assert 'reason' in result
    
    def test_portfolio_value_limit(self):
        """Test portfolio value limit check."""
        # Test within limit
        result = self.risk_manager.check_portfolio_value_limit(150000)
        assert result['allowed'] == True
        
        # Test exceeding limit
        result = self.risk_manager.check_portfolio_value_limit(250000)
        assert result['allowed'] == False
    
    def test_daily_loss_limit(self):
        """Test daily loss limit check."""
        # Test within limit
        result = self.risk_manager.check_daily_loss_limit(-3000)
        assert result['allowed'] == True
        
        # Test exceeding limit
        result = self.risk_manager.check_daily_loss_limit(-7000)
        assert result['allowed'] == False
    
    def test_drawdown_limit(self):
        """Test drawdown limit check."""
        # Test within limit
        result = self.risk_manager.check_drawdown_limit(-0.08)
        assert result['allowed'] == True
        
        # Test exceeding limit
        result = self.risk_manager.check_drawdown_limit(-0.15)
        assert result['allowed'] == False
    
    def test_comprehensive_risk_check(self):
        """Test comprehensive risk assessment."""
        # Proposed trade
        trade = {
            'symbol': 'MES',
            'quantity': 1,
            'price': 4500.0,
            'side': 'BUY'
        }
        
        risk_assessment = self.risk_manager.assess_trade_risk(
            trade, self.portfolio_state
        )
        
        # Should return risk assessment
        assert isinstance(risk_assessment, dict)
        assert 'allowed' in risk_assessment
        assert 'risk_score' in risk_assessment
        assert 'warnings' in risk_assessment
        
        # Risk score should be between 0 and 1
        assert 0 <= risk_assessment['risk_score'] <= 1
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        # Sample returns data
        returns = np.random.normal(0.001, 0.02, 252)
        
        metrics = self.risk_manager.calculate_risk_metrics(returns)
        
        assert isinstance(metrics, dict)
        expected_metrics = ['var_95', 'var_99', 'expected_shortfall', 'volatility']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
    
    def test_position_concentration(self):
        """Test position concentration limits."""
        positions = {
            'MES': 3,
            'NQ': 2,
            'ES': 1
        }
        
        portfolio_value = 100000
        
        # Check concentration
        concentration = self.risk_manager.calculate_position_concentration(
            positions, portfolio_value
        )
        
        assert isinstance(concentration, dict)
        assert all(0 <= conc <= 1 for conc in concentration.values())
    
    def test_leverage_calculation(self):
        """Test leverage calculation."""
        positions = {'MES': 2}
        position_values = {'MES': 18000}  # 2 * 4500 * 2 (multiplier)
        portfolio_value = 110000
        
        leverage = self.risk_manager.calculate_leverage(
            position_values, portfolio_value
        )
        
        assert isinstance(leverage, float)
        assert leverage >= 0
    
    def test_var_limit_check(self):
        """Test VaR limit enforcement."""
        current_var = -1500  # Within limit
        
        result = self.risk_manager.check_var_limit(current_var)
        assert result['allowed'] == True
        
        # Exceeding limit
        current_var = -3000
        result = self.risk_manager.check_var_limit(current_var)
        assert result['allowed'] == False
    
    def test_emergency_risk_controls(self):
        """Test emergency risk control triggers."""
        # Simulate extreme portfolio state
        extreme_state = {
            'cash': 50000,
            'positions': {'MES': 10},  # Oversized position
            'portfolio_value': 80000,  # Lost money
            'daily_pnl': -8000,  # Large daily loss
            'current_drawdown': -0.15  # Excessive drawdown
        }
        
        emergency_controls = self.risk_manager.check_emergency_controls(extreme_state)
        
        assert isinstance(emergency_controls, dict)
        assert 'flatten_all_positions' in emergency_controls
        assert 'stop_new_trades' in emergency_controls
        
        # Should trigger emergency controls
        assert emergency_controls['flatten_all_positions'] == True or \
               emergency_controls['stop_new_trades'] == True


class TestSimulationIntegration:
    """Test integration between simulation components."""
    
    def test_environment_execution_integration(self):
        """Test integration between environment and execution engine."""
        # Mock environment and execution engine
        with patch('src.sim.env_intraday_rl.ExecutionEngine') as mock_execution:
            mock_execution_instance = Mock()
            mock_execution.return_value = mock_execution_instance
            
            # Mock successful order execution
            mock_execution_instance.execute_order.return_value = {
                'symbol': 'MES',
                'quantity': 1,
                'side': 'BUY',
                'fill_price': 4500.0,
                'fill_time': datetime.now(),
                'transaction_cost': 2.5
            }
            
            # Test that environment uses execution engine
            # This would require actual integration testing
            assert True  # Placeholder for integration test
    
    def test_environment_risk_integration(self):
        """Test integration between environment and risk manager."""
        with patch('src.sim.env_intraday_rl.RiskManager') as mock_risk:
            mock_risk_instance = Mock()
            mock_risk.return_value = mock_risk_instance
            
            # Mock risk check passing
            mock_risk_instance.assess_trade_risk.return_value = {
                'allowed': True,
                'risk_score': 0.3,
                'warnings': []
            }
            
            # Test that environment consults risk manager
            assert True  # Placeholder for integration test
    
    def test_execution_risk_integration(self):
        """Test integration between execution engine and risk manager."""
        # This would test that execution engine respects risk limits
        # and properly interfaces with risk management
        
        config = {'max_position_size': 5, 'transaction_cost': 2.5}
        execution_engine = ExecutionEngine(config=config)
        risk_manager = RiskManager(config={'max_position_size': 5})
        
        # Test that they can work together
        assert hasattr(execution_engine, 'config')
        assert hasattr(risk_manager, 'config')
        
        # More detailed integration testing would go here
        assert True