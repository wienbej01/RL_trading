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

# Mock classes for testing without IBKR
if not IBKR_AVAILABLE:
    class Contract:
        pass
    class Trade:
        pass
    class Fill:
        pass
    class Position:
        pass
    class Ticker:
        pass
    class Order:
        pass
    class BarData:
        pass

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
        
        # Trading settings
        config.account = self.settings.get("ibkr", "account", default="")
        config.currency = self.settings.get("ibkr", "currency", default="USD")
        config.order_type = self.settings.get("ibkr", "order_type", default="MKT")
        config.time_in_force = self.settings.get("ibkr", "time_in_force", default="DAY")
        
        return config

    def connect(self) -> bool:
        """
        Connect to IBKR TWS/Gateway.
        
        Returns:
            True if connection successful
        """
        try:
            self.ib.connect(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout
            )
            
            # Wait for connection
            max_wait = 30
            waited = 0
            while not self.ib.isConnected() and waited < max_wait:
                time.sleep(1)
                waited += 1
            
            if self.ib.isConnected():
                logger.info(f"Connected to IBKR at {self.config.host}:{self.config.port}")
                self._setup_event_handlers()
                return True
            else:
                logger.error("Failed to connect to IBKR")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for IBKR events."""
        # Account updates
        self.ib.accountValueEvent += self._on_account_value
        
        # Order updates
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_execution
        
        # Position updates
        self.ib.positionEvent += self._on_position
        
        # Error handling
        self.ib.errorEvent += self._on_error

    def _on_account_value(self, account_value):
        """Handle account value updates."""
        self.account_value = float(account_value.value)
        logger.debug(f"Account value updated: ${self.account_value:,.2f}")

    def _on_order_status(self, trade):
        """Handle order status updates."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        
        logger.info(f"Order {order_id} status: {status}")
        
        if status == "Filled":
            self.filled_orders[order_id] = trade
            self._update_position(trade)
        elif status in ["Cancelled", "Rejected"]:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

    def _on_execution(self, trade, fill):
        """Handle execution updates."""
        logger.info(f"Execution: {fill.execution.side} {fill.execution.shares} @ ${fill.execution.price}")

    def _on_position(self, position):
        """Handle position updates."""
        symbol = position.contract.symbol
        self.positions[symbol] = position.position
        logger.info(f"Position update: {symbol} = {position.position}")

    def _on_error(self, req_id, error_code, error_string, contract):
        """Handle errors."""
        logger.error(f"IBKR Error {error_code}: {error_string}")
        
        # Handle specific error codes
        if error_code == 200:  # No security definition found
            logger.error(f"Contract not found: {contract}")
        elif error_code == 201:  # Order rejected
            logger.error(f"Order rejected: {error_string}")
        elif error_code == 202:  # Order cancelled
            logger.warning(f"Order cancelled: {error_string}")

    def get_account_summary(self) -> Dict[str, float]:
        """
        Get account summary.
        
        Returns:
            Dictionary with account information
        """
        if not self.ib.isConnected():
            return {}
        
        account_values = self.ib.accountValues()
        
        summary = {}
        for av in account_values:
            if av.tag in ["NetLiquidation", "TotalCashValue", "GrossPositionValue", "AvailableFunds"]:
                summary[av.tag] = float(av.value)
        
        return summary

    def get_positions(self) -> Dict[str, int]:
        """
        Get current positions.
        
        Returns:
            Dictionary of positions by symbol
        """
        if not self.ib.isConnected():
            return {}
        
        positions = {}
        for position in self.ib.positions():
            symbol = position.contract.symbol
            positions[symbol] = int(position.position)
        
        return positions

    def get_portfolio(self) -> List[Dict]:
        """
        Get portfolio information.
        
        Returns:
            List of portfolio items
        """
        if not self.ib.isConnected():
            return []
        
        portfolio = []
        for position in self.ib.positions():
            portfolio.append({
                'symbol': position.contract.symbol,
                'position': int(position.position),
                'market_value': float(position.marketValue),
                'average_cost': float(position.averageCost),
                'unrealized_pnl': float(position.unrealizedPNL),
                'realized_pnl': float(position.realizedPNL)
            })
        
        return portfolio

    def place_market_order(self, symbol: str, action: str, quantity: int) -> Optional[int]:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol
            action: "BUY" or "SELL"
            quantity: Number of shares/contracts
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return None
        
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Create order
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = "MKT"
            order.tif = "DAY"
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            self.pending_orders[order_id] = trade
            logger.info(f"Placed {action} order for {quantity} {symbol}, order ID: {order_id}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None

    def place_limit_order(self, symbol: str, action: str, quantity: int, limit_price: float) -> Optional[int]:
        """
        Place a limit order.
        
        Args:
            symbol: Trading symbol
            action: "BUY" or "SELL"
            quantity: Number of shares/contracts
            limit_price: Limit price
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return None
        
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Create order
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = "LMT"
            order.lmtPrice = limit_price
            order.tif = "DAY"
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            self.pending_orders[order_id] = trade
            logger.info(f"Placed {action} limit order for {quantity} {symbol} @ ${limit_price}, order ID: {order_id}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None

    def place_stop_order(self, symbol: str, action: str, quantity: int, stop_price: float) -> Optional[int]:
        """
        Place a stop order.
        
        Args:
            symbol: Trading symbol
            action: "BUY" or "SELL"
            quantity: Number of shares/contracts
            stop_price: Stop price
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return None
        
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Create order
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = "STP"
            order.auxPrice = stop_price
            order.tif = "DAY"
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            self.pending_orders[order_id] = trade
            logger.info(f"Placed {action} stop order for {quantity} {symbol} @ ${stop_price}, order ID: {order_id}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing stop order: {e}")
            return None

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        if not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return False
        
        try:
            self.ib.cancelOrder(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_order_status(self, order_id: int) -> Optional[str]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status if found, None otherwise
        """
        if order_id in self.pending_orders:
            trade = self.pending_orders[order_id]
            return trade.orderStatus.status
        
        if order_id in self.filled_orders:
            return "Filled"
        
        return None

    def get_market_data(self, symbol: str, duration: str = "1 D", bar_size: str = "1 min") -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            duration: Data duration (e.g., "1 D", "1 W", "1 M")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour")
            
        Returns:
            DataFrame with market data
        """
        if not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return pd.DataFrame()
        
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            # Convert to DataFrame
            if bars:
                df = pd.DataFrame(bars)
                df.set_index('date', inplace=True)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()

    def get_real_time_bars(self, symbol: str, bar_size: int = 5) -> Optional[pd.DataFrame]:
        """
        Get real-time bars.
        
        Args:
            symbol: Trading symbol
            bar_size: Bar size in seconds
            
        Returns:
            DataFrame with real-time bars
        """
        if not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return None
        
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Request real-time bars
            bars = self.ib.reqRealTimeBars(
                contract,
                barSize=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )
            
            # Convert to DataFrame
            if bars:
                df = pd.DataFrame(bars)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting real-time bars: {e}")
            return None

    def _update_position(self, trade):
        """Update position after trade execution."""
        symbol = trade.contract.symbol
        action = trade.order.action
        quantity = trade.order.totalQuantity
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        if action == "BUY":
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] -= quantity
        
        logger.info(f"Updated position: {symbol} = {self.positions[symbol]}")

    def start_data_stream(self, symbols: List[str]) -> None:
        """
        Start real-time data stream.
        
        Args:
            symbols: List of symbols to stream
        """
        if not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return
        
        for symbol in symbols:
            try:
                # Create contract
                contract = Contract()
                contract.symbol = symbol
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"
                
                # Request market data
                ticker = self.ib.reqMktData(contract, '', False, False)
                self.tickers[symbol] = ticker
                
                logger.info(f"Started data stream for {symbol}")
                
            except Exception as e:
                logger.error(f"Error starting data stream for {symbol}: {e}")

    def stop_data_stream(self, symbols: List[str] = None) -> None:
        """
        Stop real-time data stream.
        
        Args:
            symbols: List of symbols to stop streaming (None for all)
        """
        if symbols is None:
            symbols = list(self.tickers.keys())
        
        for symbol in symbols:
            if symbol in self.tickers:
                try:
                    self.ib.cancelMktData(self.tickers[symbol])
                    del self.tickers[symbol]
                    logger.info(f"Stopped data stream for {symbol}")
                except Exception as e:
                    logger.error(f"Error stopping data stream for {symbol}: {e}")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price if available
        """
        if symbol in self.tickers:
            ticker = self.tickers[symbol]
            if ticker.last:
                return float(ticker.last)
        
        # Fallback to market data request
        market_data = self.get_market_data(symbol, duration="1 D", bar_size="1 min")
        if not market_data.empty:
            return float(market_data['close'].iloc[-1])
        
        return None

    def get_account_balance(self) -> float:
        """
        Get current account balance.
        
        Returns:
            Account balance
        """
        summary = self.get_account_summary()
        return summary.get('NetLiquidation', 0.0)

    def is_connected(self) -> bool:
        """
        Check if connected to IBKR.
        
        Returns:
            True if connected
        """
        return self.ib.isConnected()

    def wait_for_fill(self, order_id: int, timeout: int = 60) -> bool:
        """
        Wait for order to fill.
        
        Args:
            order_id: Order ID
            timeout: Timeout in seconds
            
        Returns:
            True if filled
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_order_status(order_id)
            if status == "Filled":
                return True
            elif status in ["Cancelled", "Rejected"]:
                return False
            time.sleep(1)
        
        return False

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        positions = self.get_positions()
        portfolio = self.get_portfolio()
        
        total_value = sum(pos['market_value'] for pos in portfolio)
        account_balance = self.get_account_balance()
        
        # Calculate leverage
        leverage = total_value / account_balance if account_balance > 0 else 0
        
        # Calculate position concentration
        if portfolio:
            max_position = max(abs(pos['position'] * pos['average_cost']) for pos in portfolio)
            concentration = max_position / total_value if total_value > 0 else 0
        else:
            concentration = 0
        
        return {
            'account_balance': account_balance,
            'total_position_value': total_value,
            'leverage': leverage,
            'position_concentration': concentration,
            'num_positions': len(positions)
        }

    def close_all_positions(self) -> List[int]:
        """
        Close all open positions.
        
        Returns:
            List of order IDs
        """
        positions = self.get_positions()
        order_ids = []
        
        for symbol, position in positions.items():
            if position != 0:
                action = "SELL" if position > 0 else "BUY"
                quantity = abs(position)
                
                order_id = self.place_market_order(symbol, action, quantity)
                if order_id:
                    order_ids.append(order_id)
        
        return order_ids

    def shutdown(self) -> None:
        """Shutdown the client."""
        self.running = False
        
        # Stop data streams
        self.stop_data_stream()
        
        # Cancel all pending orders
        for order_id in list(self.pending_orders.keys()):
            self.cancel_order(order_id)
        
        # Disconnect
        self.disconnect()
        
        logger.info("IBKR client shutdown complete")

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
    
    def calculate_daily_pnl(self) -> float:
        """
        Calculate the realised PnL for the current trading day from recorded fills.
        
        This implementation pairs buys and sells using FIFO per symbol. It supports
        dict-like trade entries (as appended in monitor_orders) and object-like entries.
        
        Returns:
            float: Realised PnL for today's completed round trips.
        """
        from datetime import datetime
        from collections import defaultdict, deque
        today = datetime.now().date()
        pnl_total: float = 0.0
        # Maintain separate FIFO queues for open buy/sell lots per symbol
        open_lots = defaultdict(lambda: {"buy": deque(), "sell": deque()})
        for tr in getattr(self, "trade_history", []):
            # Extract fields from dict or object
            ts = tr.get("timestamp") if isinstance(tr, dict) else getattr(tr, "timestamp", None)
            if ts is None:
                continue
            try:
                tr_date = ts.date()
            except Exception:
                # Fallback for string timestamps
                try:
                    tr_date = pd.Timestamp(ts).date()  # type: ignore
                except Exception:
                    continue
            if tr_date != today:
                continue
            symbol = (tr.get("symbol") if isinstance(tr, dict) else getattr(tr, "symbol", None)) or "DEFAULT"
            action = tr.get("action") if isinstance(tr, dict) else getattr(tr, "action", None)
            qty = int(tr.get("quantity", 0) if isinstance(tr, dict) else getattr(tr, "quantity", 0))
            price_val = tr.get("price") if isinstance(tr, dict) else getattr(tr, "price", None)
            # Also support entry/exit fields if present
            entry_price = tr.get("entry_price") if isinstance(tr, dict) else getattr(tr, "entry_price", None)
            exit_price = tr.get("exit_price") if isinstance(tr, dict) else getattr(tr, "exit_price", None)
            price: Optional[float] = None
            if price_val is not None:
                try:
                    price = float(price_val)
                except Exception:
                    price = None
            if price is None and entry_price is not None and exit_price is not None:
                # If the record represents a completed trade, add full PnL and skip lot pairing
                contract_size = int(tr.get("contract_size", 1) if isinstance(tr, dict) else getattr(tr, "contract_size", 1))
                pnl_total += float(qty) * (float(exit_price) - float(entry_price)) * float(contract_size)
                continue
            if price is None or qty == 0 or action not in ("BUY", "SELL"):
                continue
            contract_size = int(tr.get("contract_size", 1) if isinstance(tr, dict) else getattr(tr, "contract_size", 1))
            # Match against opposite-side open lots first (FIFO)
            if action == "SELL":
                remaining = qty
                # Close existing buy lots
                while remaining > 0 and open_lots[symbol]["buy"]:
                    lot_qty, lot_price = open_lots[symbol]["buy"][0]
                    match_qty = min(remaining, lot_qty)
                    pnl_total += match_qty * (price - lot_price) * contract_size
                    lot_qty -= match_qty
                    remaining -= match_qty
                    if lot_qty == 0:
                        open_lots[symbol]["buy"].popleft()
                    else:
                        open_lots[symbol]["buy"][0] = (lot_qty, lot_price)
                # Any remainder opens/extends a short
                if remaining > 0:
                    open_lots[symbol]["sell"].append((remaining, price))
            else:  # action == "BUY"
                remaining = qty
                # Close existing sell (short) lots
                while remaining > 0 and open_lots[symbol]["sell"]:
                    lot_qty, lot_price = open_lots[symbol]["sell"][0]
                    match_qty = min(remaining, lot_qty)
                    # Short PnL: sold at lot_price, bought back at price
                    pnl_total += match_qty * (lot_price - price) * contract_size
                    lot_qty -= match_qty
                    remaining -= match_qty
                    if lot_qty == 0:
                        open_lots[symbol]["sell"].popleft()
                    else:
                        open_lots[symbol]["sell"][0] = (lot_qty, lot_price)
                # Any remainder opens/extends a long
                if remaining > 0:
                    open_lots[symbol]["buy"].append((remaining, price))
        return float(pnl_total)
    
    def _calculate_daily_pnl(self) -> float:
        """
        Backward-compatible alias used by risk checks.
        """
        return self.calculate_daily_pnl()
    
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
    
# Alias for compatibility
IBKRTradingClient = IBKRClient
