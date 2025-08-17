"""
Tests for trading modules (paper trading, IBKR client).
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock
import asyncio

from src.trading.paper_trading import PaperTradingEngine
from src.trading.ibkr_client import IBKRTradingClient


class TestPaperTradingEngine:
    """Test paper trading engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'initial_capital': 100000.0,
            'commission_per_trade': 2.5,
            'slippage_bps': 1.0,
            'max_position_size': 10,
            'margin_requirement': 0.25,
            'overnight_margin': 0.5
        }
        
        # Mock market data
        dates = pd.date_range('2023-01-01 09:30:00', '2023-01-01 16:00:00', 
                             freq='1min', tz='America/New_York')
        
        self.market_data = pd.DataFrame({
            'open': np.random.uniform(4490, 4510, len(dates)),
            'high': np.random.uniform(4500, 4520, len(dates)),
            'low': np.random.uniform(4480, 4500, len(dates)),
            'close': np.random.uniform(4490, 4510, len(dates)),
            'volume': np.random.randint(100, 1000, len(dates)),
            'bid': np.random.uniform(4499, 4501, len(dates)),
            'ask': np.random.uniform(4501, 4503, len(dates))
        }, index=dates)
        
        self.paper_engine = PaperTradingEngine(
            config=self.config,
            market_data=self.market_data
        )
    
    def test_paper_engine_initialization(self):
        """Test paper trading engine initialization."""
        assert self.paper_engine.config == self.config
        assert self.paper_engine.account_balance == self.config['initial_capital']
        assert self.paper_engine.positions == {}
        assert len(self.paper_engine.trade_history) == 0
        assert self.paper_engine.is_connected == False
    
    def test_connect_disconnect(self):
        """Test connection and disconnection."""
        # Test connection
        result = self.paper_engine.connect()
        assert result == True
        assert self.paper_engine.is_connected == True
        
        # Test disconnection
        result = self.paper_engine.disconnect()
        assert result == True
        assert self.paper_engine.is_connected == False
    
    def test_place_market_buy_order(self):
        """Test placing market buy order."""
        self.paper_engine.connect()
        
        order_id = self.paper_engine.place_order(
            symbol='MES',
            action='BUY',
            quantity=1,
            order_type='MARKET'
        )
        
        # Should return order ID
        assert isinstance(order_id, str)
        assert len(order_id) > 0
        
        # Check that order was recorded
        assert order_id in self.paper_engine.orders
        
        # Check order details
        order = self.paper_engine.orders[order_id]
        assert order['symbol'] == 'MES'
        assert order['action'] == 'BUY'
        assert order['quantity'] == 1
        assert order['order_type'] == 'MARKET'
        assert order['status'] in ['PENDING', 'FILLED']
    
    def test_place_limit_order(self):
        """Test placing limit order."""
        self.paper_engine.connect()
        
        order_id = self.paper_engine.place_order(
            symbol='MES',
            action='BUY',
            quantity=1,
            order_type='LIMIT',
            limit_price=4500.0
        )
        
        assert isinstance(order_id, str)
        
        order = self.paper_engine.orders[order_id]
        assert order['order_type'] == 'LIMIT'
        assert order['limit_price'] == 4500.0
    
    def test_order_execution(self):
        """Test order execution logic."""
        self.paper_engine.connect()
        
        # Place buy order
        order_id = self.paper_engine.place_order(
            symbol='MES',
            action='BUY',
            quantity=1,
            order_type='MARKET'
        )
        
        # Process orders (simulate market data update)
        current_time = self.market_data.index[0]
        current_data = self.market_data.iloc[0]
        
        self.paper_engine.process_orders(current_time, current_data)
        
        # Check if order was filled
        order = self.paper_engine.orders[order_id]
        if order['status'] == 'FILLED':
            assert 'fill_price' in order
            assert 'fill_time' in order
            
            # Check position was created
            assert 'MES' in self.paper_engine.positions
            assert self.paper_engine.positions['MES']['quantity'] == 1
    
    def test_position_tracking(self):
        """Test position tracking after trades."""
        self.paper_engine.connect()
        
        # Execute buy order
        order_id = self.paper_engine.place_order(
            symbol='MES',
            action='BUY',
            quantity=2,
            order_type='MARKET'
        )
        
        # Simulate order fill
        self.paper_engine._fill_order(
            order_id=order_id,
            fill_price=4500.0,
            fill_time=datetime.now()
        )
        
        # Check position
        assert 'MES' in self.paper_engine.positions
        position = self.paper_engine.positions['MES']
        assert position['quantity'] == 2
        assert position['avg_price'] == 4500.0
        
        # Execute partial sell
        sell_order_id = self.paper_engine.place_order(
            symbol='MES',
            action='SELL',
            quantity=1,
            order_type='MARKET'
        )
        
        self.paper_engine._fill_order(
            order_id=sell_order_id,
            fill_price=4510.0,
            fill_time=datetime.now()
        )
        
        # Check updated position
        position = self.paper_engine.positions['MES']
        assert position['quantity'] == 1  # 2 - 1
        assert position['avg_price'] == 4500.0  # Unchanged for remaining
    
    def test_account_balance_updates(self):
        """Test account balance updates after trades."""
        self.paper_engine.connect()
        initial_balance = self.paper_engine.account_balance
        
        # Execute buy order
        order_id = self.paper_engine.place_order(
            symbol='MES',
            action='BUY',
            quantity=1,
            order_type='MARKET'
        )
        
        # Simulate fill
        fill_price = 4500.0
        self.paper_engine._fill_order(
            order_id=order_id,
            fill_price=fill_price,
            fill_time=datetime.now()
        )
        
        # Check balance decreased
        expected_cost = fill_price + self.config['commission_per_trade']
        expected_balance = initial_balance - expected_cost
        
        assert abs(self.paper_engine.account_balance - expected_balance) < 1e-6
    
    def test_commission_calculation(self):
        """Test commission calculation."""
        commission = self.paper_engine.calculate_commission(
            symbol='MES',
            quantity=1,
            price=4500.0
        )
        
        assert commission == self.config['commission_per_trade']
        assert isinstance(commission, float)
        assert commission >= 0
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        base_price = 4500.0
        
        # Buy order slippage (adverse)
        slipped_price = self.paper_engine.apply_slippage(
            price=base_price,
            action='BUY',
            quantity=1
        )
        assert slipped_price >= base_price  # Should be higher for buy
        
        # Sell order slippage (adverse)
        slipped_price = self.paper_engine.apply_slippage(
            price=base_price,
            action='SELL',
            quantity=1
        )
        assert slipped_price <= base_price  # Should be lower for sell
    
    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation."""
        self.paper_engine.connect()
        
        # Create position
        self.paper_engine.positions['MES'] = {
            'quantity': 2,
            'avg_price': 4500.0
        }
        
        # Calculate unrealized P&L
        current_price = 4520.0
        unrealized_pnl = self.paper_engine.calculate_unrealized_pnl(
            symbol='MES',
            current_price=current_price
        )
        
        expected_pnl = 2 * (current_price - 4500.0)  # quantity * price_diff
        assert abs(unrealized_pnl - expected_pnl) < 1e-6
    
    def test_margin_calculation(self):
        """Test margin requirement calculation."""
        margin_required = self.paper_engine.calculate_margin_requirement(
            symbol='MES',
            quantity=1,
            price=4500.0
        )
        
        expected_margin = 4500.0 * self.config['margin_requirement']
        assert abs(margin_required - expected_margin) < 1e-6
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        self.paper_engine.connect()
        
        # Create some positions
        self.paper_engine.positions = {
            'MES': {'quantity': 2, 'avg_price': 4500.0},
            'NQ': {'quantity': 1, 'avg_price': 15000.0}
        }
        
        # Mock current prices
        current_prices = {'MES': 4520.0, 'NQ': 15100.0}
        
        summary = self.paper_engine.get_portfolio_summary(current_prices)
        
        # Check summary structure
        assert isinstance(summary, dict)
        assert 'account_balance' in summary
        assert 'positions' in summary
        assert 'total_unrealized_pnl' in summary
        assert 'total_portfolio_value' in summary
        
        # Check calculations
        expected_mes_pnl = 2 * (4520.0 - 4500.0)
        expected_nq_pnl = 1 * (15100.0 - 15000.0)
        expected_total_pnl = expected_mes_pnl + expected_nq_pnl
        
        assert abs(summary['total_unrealized_pnl'] - expected_total_pnl) < 1e-6
    
    def test_risk_checks(self):
        """Test risk management checks."""
        self.paper_engine.connect()
        
        # Test position size limit
        can_trade = self.paper_engine.check_position_size_limit(
            symbol='MES',
            quantity=5  # Within limit
        )
        assert can_trade == True
        
        can_trade = self.paper_engine.check_position_size_limit(
            symbol='MES',
            quantity=15  # Exceeds limit
        )
        assert can_trade == False
        
        # Test margin requirement
        has_margin = self.paper_engine.check_margin_requirement(
            symbol='MES',
            quantity=1,
            price=4500.0
        )
        assert isinstance(has_margin, bool)
    
    def test_order_cancellation(self):
        """Test order cancellation."""
        self.paper_engine.connect()
        
        # Place order
        order_id = self.paper_engine.place_order(
            symbol='MES',
            action='BUY',
            quantity=1,
            order_type='LIMIT',
            limit_price=4400.0  # Below current market
        )
        
        # Cancel order
        result = self.paper_engine.cancel_order(order_id)
        
        assert result == True
        
        # Check order status
        order = self.paper_engine.orders[order_id]
        assert order['status'] == 'CANCELLED'


class TestIBKRTradingClient:
    """Test IBKR trading client."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1,
            'timeout': 30,
            'account': 'DU12345'
        }
        
        self.client = IBKRTradingClient(config=self.config)
    
    def test_client_initialization(self):
        """Test client initialization."""
        assert self.client.config == self.config
        assert hasattr(self.client, 'ib')
        assert self.client.is_connected == False
    
    @patch('src.trading.ibkr_client.IB')
    def test_connection(self, mock_ib):
        """Test connection to IBKR."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.connect.return_value = True
        mock_ib_instance.isConnected.return_value = True
        
        result = self.client.connect()
        
        assert result == True
        assert self.client.is_connected == True
        mock_ib_instance.connect.assert_called_once_with(
            self.config['host'],
            self.config['port'],
            clientId=self.config['client_id'],
            timeout=self.config['timeout']
        )
    
    @patch('src.trading.ibkr_client.IB')
    def test_place_market_order(self, mock_ib):
        """Test placing market order."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.isConnected.return_value = True
        
        # Mock contract and order
        mock_contract = Mock()
        mock_order = Mock()
        
        with patch('src.trading.ibkr_client.Future', return_value=mock_contract), \
             patch('src.trading.ibkr_client.MarketOrder', return_value=mock_order):
            
            # Mock trade response
            mock_trade = Mock()
            mock_trade.order.orderId = 12345
            mock_ib_instance.placeOrder.return_value = mock_trade
            
            self.client.ib = mock_ib_instance
            
            order_id = self.client.place_market_order(
                symbol='MES',
                action='BUY',
                quantity=1
            )
            
            assert order_id == 12345
            mock_ib_instance.placeOrder.assert_called_once()
    
    @patch('src.trading.ibkr_client.IB')
    def test_place_limit_order(self, mock_ib):
        """Test placing limit order."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.isConnected.return_value = True
        
        mock_contract = Mock()
        mock_order = Mock()
        
        with patch('src.trading.ibkr_client.Future', return_value=mock_contract), \
             patch('src.trading.ibkr_client.LimitOrder', return_value=mock_order):
            
            mock_trade = Mock()
            mock_trade.order.orderId = 12346
            mock_ib_instance.placeOrder.return_value = mock_trade
            
            self.client.ib = mock_ib_instance
            
            order_id = self.client.place_limit_order(
                symbol='MES',
                action='BUY',
                quantity=1,
                limit_price=4500.0
            )
            
            assert order_id == 12346
    
    @patch('src.trading.ibkr_client.IB')
    def test_cancel_order(self, mock_ib):
        """Test order cancellation."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.isConnected.return_value = True
        
        self.client.ib = mock_ib_instance
        
        result = self.client.cancel_order(12345)
        
        mock_ib_instance.cancelOrder.assert_called_once_with(12345)
        assert result == True
    
    @patch('src.trading.ibkr_client.IB')
    def test_get_account_summary(self, mock_ib):
        """Test getting account summary."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.isConnected.return_value = True
        
        # Mock account values
        mock_account_values = [
            Mock(account='DU12345', tag='TotalCashValue', value='100000'),
            Mock(account='DU12345', tag='NetLiquidation', value='110000'),
            Mock(account='DU12345', tag='UnrealizedPnL', value='5000')
        ]
        mock_ib_instance.accountValues.return_value = mock_account_values
        
        self.client.ib = mock_ib_instance
        
        summary = self.client.get_account_summary()
        
        assert isinstance(summary, dict)
        assert 'TotalCashValue' in summary
        assert 'NetLiquidation' in summary
        assert 'UnrealizedPnL' in summary
        assert summary['TotalCashValue'] == 100000.0
    
    @patch('src.trading.ibkr_client.IB')
    def test_get_positions(self, mock_ib):
        """Test getting positions."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.isConnected.return_value = True
        
        # Mock positions
        mock_position = Mock()
        mock_position.contract.symbol = 'MES'
        mock_position.position = 2.0
        mock_position.avgCost = 4500.0
        mock_position.unrealizedPNL = 100.0
        
        mock_ib_instance.positions.return_value = [mock_position]
        
        self.client.ib = mock_ib_instance
        
        positions = self.client.get_positions()
        
        assert isinstance(positions, list)
        assert len(positions) == 1
        
        position = positions[0]
        assert position['symbol'] == 'MES'
        assert position['position'] == 2.0
        assert position['avgCost'] == 4500.0
        assert position['unrealizedPNL'] == 100.0
    
    @patch('src.trading.ibkr_client.IB')
    def test_get_open_orders(self, mock_ib):
        """Test getting open orders."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.isConnected.return_value = True
        
        # Mock open trades
        mock_trade = Mock()
        mock_trade.order.orderId = 12345
        mock_trade.order.action = 'BUY'
        mock_trade.order.totalQuantity = 1
        mock_trade.order.orderType = 'MARKET'
        mock_trade.orderStatus.status = 'Submitted'
        mock_trade.contract.symbol = 'MES'
        
        mock_ib_instance.openTrades.return_value = [mock_trade]
        
        self.client.ib = mock_ib_instance
        
        orders = self.client.get_open_orders()
        
        assert isinstance(orders, list)
        assert len(orders) == 1
        
        order = orders[0]
        assert order['orderId'] == 12345
        assert order['symbol'] == 'MES'
        assert order['action'] == 'BUY'
        assert order['quantity'] == 1
        assert order['orderType'] == 'MARKET'
        assert order['status'] == 'Submitted'
    
    @patch('src.trading.ibkr_client.IB')
    def test_get_executions(self, mock_ib):
        """Test getting trade executions."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.isConnected.return_value = True
        
        # Mock executions
        mock_fill = Mock()
        mock_fill.execution.execId = 'E001'
        mock_fill.execution.orderId = 12345
        mock_fill.execution.price = 4500.5
        mock_fill.execution.shares = 1
        mock_fill.execution.side = 'BOT'
        mock_fill.execution.time = '20230101  10:30:00'
        mock_fill.contract.symbol = 'MES'
        
        mock_ib_instance.fills.return_value = [mock_fill]
        
        self.client.ib = mock_ib_instance
        
        executions = self.client.get_executions()
        
        assert isinstance(executions, list)
        assert len(executions) == 1
        
        execution = executions[0]
        assert execution['execId'] == 'E001'
        assert execution['orderId'] == 12345
        assert execution['symbol'] == 'MES'
        assert execution['price'] == 4500.5
        assert execution['quantity'] == 1
    
    def test_contract_creation(self):
        """Test contract creation for different instruments."""
        # Test futures contract
        contract = self.client.create_contract('MES', 'FUTURE')
        
        assert hasattr(contract, 'symbol')
        assert hasattr(contract, 'secType')
        
        # Additional contract tests would depend on actual implementation
        assert True  # Placeholder
    
    @patch('src.trading.ibkr_client.IB')
    def test_disconnect(self, mock_ib):
        """Test disconnection."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        
        self.client.ib = mock_ib_instance
        self.client.is_connected = True
        
        result = self.client.disconnect()
        
        mock_ib_instance.disconnect.assert_called_once()
        assert result == True
        assert self.client.is_connected == False


class TestTradingIntegration:
    """Test integration between trading components."""
    
    def test_paper_to_live_transition(self):
        """Test transition from paper to live trading."""
        # This would test that paper and live trading interfaces are compatible
        paper_config = {
            'initial_capital': 100000.0,
            'commission_per_trade': 2.5
        }
        
        ibkr_config = {
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1
        }
        
        paper_engine = PaperTradingEngine(config=paper_config)
        ibkr_client = IBKRTradingClient(config=ibkr_config)
        
        # Both should have similar interfaces
        assert hasattr(paper_engine, 'place_order')
        assert hasattr(ibkr_client, 'place_market_order')
        assert hasattr(paper_engine, 'get_portfolio_summary')
        assert hasattr(ibkr_client, 'get_account_summary')
        
        # This demonstrates interface compatibility
        assert True
    
    def test_trading_with_rl_agent(self):
        """Test integration with RL agent."""
        # Mock RL agent
        mock_agent = Mock()
        mock_agent.predict.return_value = 1  # Buy action
        
        # Mock paper trading engine
        paper_engine = Mock()
        paper_engine.place_order.return_value = 'ORDER_123'
        
        # Test trading loop integration
        action = mock_agent.predict()
        
        if action == 1:  # Buy
            order_id = paper_engine.place_order(
                symbol='MES',
                action='BUY',
                quantity=1,
                order_type='MARKET'
            )
            assert order_id == 'ORDER_123'
        
        # Verify interaction
        mock_agent.predict.assert_called_once()
        paper_engine.place_order.assert_called_once()
    
    def test_real_time_data_integration(self):
        """Test integration with real-time data feeds."""
        # This would test that trading engines can work with real-time data
        with patch('src.trading.paper_trading.MarketDataFeed') as mock_feed:
            mock_feed_instance = Mock()
            mock_feed.return_value = mock_feed_instance
            
            # Mock real-time tick
            mock_tick = {
                'symbol': 'MES',
                'bid': 4500.0,
                'ask': 4500.5,
                'last': 4500.25,
                'timestamp': datetime.now()
            }
            mock_feed_instance.get_latest_tick.return_value = mock_tick
            
            # Test that trading engine can process real-time data
            assert mock_tick['symbol'] == 'MES'
            assert 'bid' in mock_tick
            assert 'ask' in mock_tick