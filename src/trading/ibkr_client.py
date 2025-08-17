"""
Interactive Brokers (IBKR) client for paper trading.

This module provides a comprehensive interface for connecting to IBKR,
handling market data, and executing trades in paper trading mode.
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
import threading
from queue import Queue

try:
    import ib_insync
    from ib_insync import *
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("Warning: ib_insync not available. Install with: pip install ib_insync")

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..sim.risk import RiskManager, RiskConfig

logger = get_logger(__name__)


@dataclass
class IBKRConfig:
    """IBKR configuration."""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: int = 30
    retry_delay: int = 5
    max_retries: int = 3
    paper_trading: bool = True
    account: str = ""
    contract_details: Dict[str, Any] = field(default_factory=dict)
    order_params: Dict[str, Any] = field(default_factory=dict)


class IBKRClient:
    """
    Interactive Brokers client for paper trading.
    
    This class provides a comprehensive interface for connecting to IBKR,
    handling market data, and executing trades in paper trading mode.
    """
    
    def __init__(self, settings: Settings, config: Optional[IBKRConfig] = None):
        """
        Initialize IBKR client.
        
        Args:
            settings: Configuration settings
            config: IBKR configuration
        """
        if not IBKR_AVAILABLE:
            raise ImportError("ib_insync is required for IBKR integration. Install with: pip install ib_insync")
        
        self.settings = settings
        self.config = config or self._load_config()
        
        # IBKR connection
        self.ib = None
        self.connected = False
        
        # Market data
        self.contracts = {}
        self.market_data = {}
        self.historical_data = {}
        
        # Order management
        self.orders = {}
        self.open_orders = {}
        self.filled_orders = {}
        self.order_queue = Queue()
        
        # Risk management
        self.risk_manager = RiskManager(settings)
        
        # Threading
        self.data_thread = None
        self.order_thread = None
        self.running = False
        
        logger.info("IBKR client initialized")
    
    def _load_config(self) -> IBKRConfig:
        """
        Load IBKR configuration from settings.
        
        Returns:
            IBKR configuration
        """
        config = IBKRConfig()
        
        # Connection settings
        config.host = self.settings.get("ibkr", "host", default="127.0.0.1")
        config.port = self.settings.get("ibkr", "port", default=7497)
        config.client_id = self.settings.get("ibkr", "client_id", default=1)
        config.timeout = self.settings.get("ibkr", "timeout", default=30)
        config.retry_delay = self.settings.get("ibkr", "retry_delay", default=5)
        config.max_retries = self.settings.get("ibkr", "max_retries", default=3)
        config.paper_trading = self.settings.get("ibkr", "paper_trading", default=True)
        config.account = self.settings.get("ibkr", "account", default="")
        
        # Contract details
        config.contract_details = self.settings.get("ibkr", "contracts", default={})
        
        # Order parameters
        config.order_params = self.settings.get("ibkr", "orders", default={})
        
        return config
    
    async def connect(self) -> bool:
        """
        Connect to IBKR.
        
        Returns:
            True if connected successfully
        """
        logger.info(f"Connecting to IBKR at {self.config.host}:{self.config.port}...")
        
        try:
            # Create IB instance
            self.ib = IB()
            
            # Connect
            if self.config.paper_trading:
                self.ib.connect(self.config.host, self.config.port, clientId=self.config.client_id, readonly=False)
            else:
                self.ib.connect(self.config.host, self.config.port, clientId=self.config.client_id, readonly=False)
            
            # Wait for connection
            await self.ib.async_run()
            
            self.connected = True
            logger.info("Successfully connected to IBKR")
            
            # Set up event handlers
            self._setup_event_handlers()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False
    
    def _setup_event_handlers(self):
        """Set up IBKR event handlers."""
        if not self.ib:
            return
        
        # Order status updates
        self.ib.orderStatusEvent += self._on_order_status
        
        # Trade executions
        self.ib.execDetailsEvent += self._on_execution
        
        # Account updates
        self.ib.updateAccountValueEvent += self._on_account_update
        
        # Portfolio updates
        self.ib.updatePortfolioEvent += self._on_portfolio_update
    
    def _on_order_status(self, trade: Trade):
        """Handle order status updates."""
        order_id = trade.orderId
        status = trade.orderStatus.status
        
        logger.info(f"Order {order_id} status: {status}")
        
        if status == "Filled":
            self.filled_orders[order_id] = trade
        elif status == "Submitted":
            self.open_orders[order_id] = trade
        elif status == "Cancelled":
            if order_id in self.open_orders:
                del self.open_orders[order_id]
    
    def _on_execution(self, trade: Trade, fill: Fill):
        """Handle trade executions."""
        logger.info(f"Execution: {fill.contract.symbol} {fill.execution.side} {fill.execution.shares} @ {fill.execution.price}")
    
    def _on_account_update(self, account: str, key: str, value: str, currency: str):
        """Handle account updates."""
        logger.debug(f"Account update: {account} {key} = {value} {currency}")
    
    def _on_portfolio_update(self, position: Position):
        """Handle portfolio updates."""
        logger.debug(f"Portfolio update: {position.contract.symbol} {position.position}")
    
    async def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.connected:
            logger.info("Disconnecting from IBKR...")
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def get_contract(self, symbol: str, exchange: str = "CME", currency: str = "USD", **kwargs) -> Optional[Contract]:
        """
        Get contract for a symbol.
        
        Args:
            symbol: Symbol (e.g., "MES")
            exchange: Exchange (e.g., "CME")
            currency: Currency (e.g., "USD")
            **kwargs: Additional contract parameters
            
        Returns:
            IBKR contract or None
        """
        contract_key = f"{symbol}_{exchange}"
        
        if contract_key in self.contracts:
            return self.contracts[contract_key]
        
        try:
            # Create contract
            if symbol.startswith(("MES", "ES", "NQ", "YM")):
                # Futures contract
                contract = Future(symbol=symbol, exchange=exchange, currency=currency, **kwargs)
            elif symbol.startswith(("SPY", "QQQ", "IWM", "DIA")):
                # Stock contract
                contract = Stock(symbol=symbol, exchange="SMART", currency=currency, **kwargs)
            else:
                # Default to futures
                contract = Future(symbol=symbol, exchange=exchange, currency=currency, **kwargs)
            
            # Qualify contract
            qualified_contract = await self.ib.qualifyContractsAsync(contract)
            
            if qualified_contract:
                self.contracts[contract_key] = qualified_contract[0]
                logger.info(f"Contract qualified: {symbol}")
                return qualified_contract[0]
            else:
                logger.error(f"Failed to qualify contract: {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting contract {symbol}: {e}")
            return None
    
    async def request_market_data(self, contract: Contract, data_type: str = "TRADES") -> bool:
        """
        Request market data.
        
        Args:
            contract: IBKR contract
            data_type: Data type (e.g., "TRADES", "MIDPOINT", "BID_ASK")
            
        Returns:
            True if data request successful
        """
        try:
            # Cancel existing market data
            if contract in self.market_data:
                await self.ib.cancelMktDataAsync(contract)
            
            # Request market data
            ticker = await self.ib.reqMktDataAsync(contract, genericTickList=data_type, snapshot=False)
            
            self.market_data[contract] = ticker
            logger.info(f"Market data requested for {contract.symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error requesting market data: {e}")
            return False
    
    async def request_historical_data(self, 
                                    contract: Contract, 
                                    duration: str = "1 D", 
                                    bar_size: str = "1 min",
                                    end_date: Optional[str] = None) -> Optional[List[BarData]]:
        """
        Request historical data.
        
        Args:
            contract: IBKR contract
            duration: Duration (e.g., "1 D", "1 W", "1 M")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour")
            end_date: End date (default: now)
            
        Returns:
            Historical data or None
        """
        try:
            # Request historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1
            )
            
            if bars:
                self.historical_data[contract] = bars
                logger.info(f"Historical data received for {contract.symbol}: {len(bars)} bars")
                return bars
            else:
                logger.warning(f"No historical data received for {contract.symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error requesting historical data: {e}")
            return None
    
    def get_historical_dataframe(self, contract: Contract) -> Optional[pd.DataFrame]:
        """
        Get historical data as DataFrame.
        
        Args:
            contract: IBKR contract
            
        Returns:
            Historical data DataFrame or None
        """
        if contract not in self.historical_data:
            return None
        
        bars = self.historical_data[contract]
        
        if not bars:
            return None
        
        # Convert to DataFrame
        data = []
        for bar in bars:
            data.append({
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'average': bar.average,
                'barCount': bar.barCount
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        return df
    
    async def place_order(self, 
                         contract: Contract, 
                         action: str, 
                         quantity: int, 
                         order_type: str = "MKT",
                         price: Optional[float] = None,
                         **kwargs) -> Optional[int]:
        """
        Place an order.
        
        Args:
            contract: IBKR contract
            action: Action ("BUY" or "SELL")
            quantity: Quantity
            order_type: Order type ("MKT", "LMT", "STP", etc.)
            price: Price for limit/stop orders
            **kwargs: Additional order parameters
            
        Returns:
            Order ID or None
        """
        try:
            # Create order
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = order_type
            
            if price is not None:
                order.lmtPrice = price
            
            # Set additional parameters
            for key, value in kwargs.items():
                setattr(order, key, value)
            
            # Place order
            trade = await self.ib.placeOrderAsync(contract, order)
            
            if trade:
                order_id = trade.orderId
                self.orders[order_id] = trade
                logger.info(f"Order placed: {action} {quantity} {contract.symbol} @ {price or 'MKT'} (ID: {order_id})")
                return order_id
            else:
                logger.error("Failed to place order")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if cancelled successfully
        """
        try:
            if order_id in self.orders:
                await self.ib.cancelOrderAsync(self.orders[order_id])
                logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                logger.error(f"Order not found: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary.
        
        Returns:
            Account summary dictionary
        """
        try:
            account_summary = await self.ib.accountSummaryAsync()
            
            summary = {}
            for item in account_summary:
                summary[item.tag] = item.value
            
            logger.info("Account summary retrieved")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}
    
    async def get_portfolio(self) -> List[Dict[str, Any]]:
        """
        Get portfolio positions.
        
        Returns:
            List of portfolio positions
        """
        try:
            portfolio = await self.ib.portfolioAsync()
            
            positions = []
            for position in portfolio:
                positions.append({
                    'symbol': position.contract.symbol,
                    'position': position.position,
                    'market_price': position.marketPrice,
                    'market_value': position.marketValue,
                    'average_cost': position.averageCost,
                    'unrealized_pnl': position.unrealizedPNL,
                    'realized_pnl': position.realizedPNL
                })
            
            logger.info("Portfolio retrieved")
            return positions
            
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return []
    
    async def start_data_stream(self, contract: Contract, callback=None):
        """
        Start real-time data stream.
        
        Args:
            contract: IBKR contract
            callback: Callback function for data updates
        """
        try:
            # Request market data
            await self.request_market_data(contract)
            
            # Set up data handler
            if callback:
                self.ib.updateEvent += callback
            
            logger.info(f"Data stream started for {contract.symbol}")
            
        except Exception as e:
            logger.error(f"Error starting data stream: {e}")
    
    async def stop_data_stream(self, contract: Contract):
        """
        Stop real-time data stream.
        
        Args:
            contract: IBKR contract
        """
        try:
            if contract in self.market_data:
                await self.ib.cancelMktDataAsync(contract)
                del self.market_data[contract]
                logger.info(f"Data stream stopped for {contract.symbol}")
                
        except Exception as e:
            logger.error(f"Error stopping data stream: {e}")
    
    def get_market_data(self, contract: Contract) -> Optional[Ticker]:
        """
        Get current market data.
        
        Args:
            contract: IBKR contract
            
        Returns:
            Market data ticker or None
        """
        return self.market_data.get(contract)
    
    def get_order_status(self, order_id: int) -> Optional[Trade]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Trade object or None
        """
        return self.orders.get(order_id)
    
    def get_open_orders(self) -> Dict[int, Trade]:
        """
        Get open orders.
        
        Returns:
            Dictionary of open orders
        """
        return self.open_orders
    
    def get_filled_orders(self) -> Dict[int, Trade]:
        """
        Get filled orders.
        
        Returns:
            Dictionary of filled orders
        """
        return self.filled_orders
    
    def is_connected(self) -> bool:
        """
        Check if connected to IBKR.
        
        Returns:
            True if connected
        """
        return self.connected and self.ib is not None
    
    def get_connection_status(self) -> str:
        """
        Get connection status.
        
        Returns:
            Connection status string
        """
        if not self.ib:
            return "Not initialized"
        elif self.connected:
            return "Connected"
        else:
            return "Disconnected"


class PaperTradingInterface:
    """
    Paper trading interface for RL trading system.
    
    This class provides a high-level interface for paper trading
    with integrated risk management and order execution.
    """
    
    def __init__(self, settings: Settings, ibkr_config: Optional[IBKRConfig] = None):
        """
        Initialize paper trading interface.
        
        Args:
            settings: Configuration settings
            ibkr_config: IBKR configuration
        """
        self.settings = settings
        self.ibkr_client = IBKRClient(settings, ibkr_config)
        
        # Trading state
        self.current_position = 0
        self.pending_orders = {}
        self.trade_history = []
        
        # Risk management
        self.risk_config = RiskConfig(
            risk_per_trade_frac=float(settings.get("risk", "risk_per_trade_frac", default=0.02)),
            stop_r_multiple=float(settings.get("risk", "stop_r_multiple", default=1.0)),
            tp_r_multiple=float(settings.get("risk", "tp_r_multiple", default=1.5)),
            max_daily_loss_r=float(settings.get("risk", "max_daily_loss_r", default=3.0))
        )
        
        logger.info("Paper trading interface initialized")
    
    async def connect(self) -> bool:
        """
        Connect to IBKR.
        
        Returns:
            True if connected successfully
        """
        return await self.ibkr_client.connect()
    
    async def disconnect(self):
        """Disconnect from IBKR."""
        await self.ibkr_client.disconnect()
    
    async def execute_trade(self, 
                           symbol: str, 
                           action: str, 
                           quantity: int, 
                           order_type: str = "MKT",
                           price: Optional[float] = None,
                           **kwargs) -> Optional[int]:
        """
        Execute a trade with risk management.
        
        Args:
            symbol: Symbol to trade
            action: Action ("BUY" or "SELL")
            quantity: Quantity
            order_type: Order type
            price: Price for limit orders
            **kwargs: Additional parameters
            
        Returns:
            Order ID or None
        """
        try:
            # Get contract
            contract = await self.ibkr_client.get_contract(symbol)
            if not contract:
                logger.error(f"Failed to get contract for {symbol}")
                return None
            
            # Check risk limits
            if not self._check_risk_limits(action, quantity):
                logger.warning("Trade rejected due to risk limits")
                return None
            
            # Place order
            order_id = await self.ibkr_client.place_order(
                contract, action, quantity, order_type, price, **kwargs
            )
            
            if order_id:
                self.pending_orders[order_id] = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'order_type': order_type,
                    'price': price,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Trade executed: {action} {quantity} {symbol} (Order ID: {order_id})")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def _check_risk_limits(self, action: str, quantity: int) -> bool:
        """
        Check if trade complies with risk limits.
        
        Args:
            action: Action ("BUY" or "SELL")
            quantity: Quantity
            
        Returns:
            True if trade complies with risk limits
        """
        # Calculate position after trade
        new_position = self.current_position
        if action == "BUY":
            new_position += quantity
        elif action == "SELL":
            new_position -= quantity
        
        # Check position limits
        max_position = self.settings.get("risk", "max_position_size", default=100)
        if abs(new_position) > max_position:
            logger.warning(f"Position limit exceeded: {new_position} > {max_position}")
            return False
        
        # Check daily loss limit
        daily_pnl = self._calculate_daily_pnl()
        max_daily_loss = self.settings.get("risk", "max_daily_loss", default=1000)
        if daily_pnl < -max_daily_loss:
            logger.warning(f"Daily loss limit exceeded: {daily_pnl} < {-max_daily_loss}")
            return False
        
        return True
    
    def _calculate_daily_pnl(self) -> float:
        """
        Calculate daily P&L.
        
        Returns:
            Daily P&L
        """
        # This would be implemented based on actual trade executions
        # For now, return 0
        return 0.0
    
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol
            
        Returns:
            Market data dictionary or None
        """
        try:
            contract = await self.ibkr_client.get_contract(symbol)
            if not contract:
                return None
            
            ticker = self.ibkr_client.get_market_data(contract)
            if ticker:
                return {
                    'symbol': ticker.contract.symbol,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'last': ticker.last,
                    'volume': ticker.volume,
                    'time': ticker.time
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information dictionary
        """
        try:
            account_summary = await self.ibkr_client.get_account_summary()
            portfolio = await self.ibkr_client.get_portfolio()
            
            return {
                'account_summary': account_summary,
                'portfolio': portfolio,
                'current_position': self.current_position,
                'pending_orders': len(self.pending_orders),
                'trade_history': len(self.trade_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    async def monitor_orders(self):
        """
        Monitor order status and update trading state.
        """
        while True:
            try:
                # Check pending orders
                completed_orders = []
                
                for order_id, order_info in self.pending_orders.items():
                    trade = self.ibkr_client.get_order_status(order_id)
                    
                    if trade and trade.orderStatus.status == "Filled":
                        # Update position
                        if order_info['action'] == "BUY":
                            self.current_position += order_info['quantity']
                        elif order_info['action'] == "SELL":
                            self.current_position -= order_info['quantity']
                        
                        # Add to trade history
                        self.trade_history.append({
                            'order_id': order_id,
                            'symbol': order_info['symbol'],
                            'action': order_info['action'],
                            'quantity': order_info['quantity'],
                            'price': trade.orderStatus.avgFillPrice,
                            'timestamp': datetime.now()
                        })
                        
                        completed_orders.append(order_id)
                        logger.info(f"Order filled: {order_info['action']} {order_info['quantity']} {order_info['symbol']}")
                
                # Remove completed orders
                for order_id in completed_orders:
                    del self.pending_orders[order_id]
                
                # Wait before next check
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(5)
    
    async def start_monitoring(self):
        """Start order monitoring."""
        logger.info("Starting order monitoring...")
        await self.monitor_orders()
    
    def get_trading_state(self) -> Dict[str, Any]:
        """
        Get current trading state.
        
        Returns:
            Trading state dictionary
        """
        return {
            'current_position': self.current_position,
            'pending_orders': self.pending_orders,
            'trade_history': self.trade_history,
            'risk_config': self.risk_config.__dict__
        }
    
    def reset_trading_state(self):
        """Reset trading state."""
        self.current_position = 0
        self.pending_orders = {}
        self.trade_history = []
        logger.info("Trading state reset")