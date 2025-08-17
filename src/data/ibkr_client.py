"""
Interactive Brokers (IBKR) client for live and paper trading.

This module provides interfaces for connecting to IBKR, managing orders,
and handling market data in real-time.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from ib_insync import IB, Stock, Future, Option, Index, Contract, util
    from ib_insync import Order, OrderStatus, Trade, TagValue
    from ib_insync import MarketDataType, TickType, BarData
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    logging.warning("ib_insync not available. IBKR client will be disabled.")

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class IBKRClient:
    """
    Interactive Brokers client for trading operations.
    
    This class handles connection to IBKR, order management, and market data
    retrieval for both paper and live trading.
    """
    
    def __init__(self, settings: Settings, paper_trading: bool = True):
        """
        Initialize IBKR client.
        
        Args:
            settings: Configuration settings
            paper_trading: Whether to use paper trading account
        """
        if not IBKR_AVAILABLE:
            raise ImportError("ib_insync is required for IBKR client")
        
        self.settings = settings
        self.paper_trading = paper_trading
        self.ib = IB()
        self.connected = False
        self.trades: Dict[str, Trade] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.order_callbacks: Dict[int, Callable] = {}
        
        # Instrument configuration
        self.instruments = settings.get('instruments', {})
        
    async def connect(self, timeout: int = 30) -> None:
        """
        Connect to IBKR TWS or Gateway.
        
        Args:
            timeout: Connection timeout in seconds
        """
        try:
            if self.paper_trading:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.ib.connect, '127.0.0.1', 7497, clientId=1
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.ib.connect, '127.0.0.1', 7496, clientId=1
                )
            
            self.connected = True
            logger.info(f"Connected to IBKR {'paper' if self.paper_trading else 'live'} trading")
            
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def create_contract(self, symbol: str, sec_type: str = 'FUT', 
                       exchange: str = 'CME', currency: str = 'USD',
                       expiry: Optional[str] = None) -> Contract:
        """
        Create IBKR contract object.
        
        Args:
            symbol: Symbol (e.g., 'MES', '10Y')
            sec_type: Security type
            exchange: Exchange
            currency: Currency
            expiry: Optional expiry date for futures/options
            
        Returns:
            IBKR Contract object
        """
        if sec_type == 'FUT':
            contract = Future(symbol, exchange, currency)
        elif sec_type == 'STK':
            contract = Stock(symbol, 'SMART', currency)
        elif sec_type == 'IND':
            contract = Index(symbol, 'SMART', currency)
        elif sec_type == 'OPT':
            if not expiry:
                raise ValueError("Expiry required for options")
            contract = Option(symbol, expiry, '', 0.0, '', exchange, currency)
        else:
            raise ValueError(f"Unsupported security type: {sec_type}")
        
        return contract
    
    def get_contract_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get contract details for an instrument.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Contract details dictionary
        """
        if symbol not in self.instruments:
            raise ValueError(f"Instrument not configured: {symbol}")
        
        config = self.instruments[symbol]
        contract = self.create_contract(
            symbol=config['symbol'],
            sec_type='FUT',
            exchange=config['exchange'],
            currency=config['currency']
        )
        
        try:
            details = self.ib.reqContractDetails(contract)[0]
            return {
                'symbol': symbol,
                'exchange': config['exchange'],
                'currency': config['currency'],
                'tick_size': config['tick_size'],
                'point_value': config['point_value'],
                'contract_multiplier': config['contract_multiplier'],
                'min_tick': config['min_tick'],
                'ib_details': details
            }
        except Exception as e:
            logger.error(f"Failed to get contract details for {symbol}: {e}")
            raise
    
    def place_order(self, symbol: str, action: str, quantity: int,
                   order_type: str = 'MKT', price: Optional[float] = None,
                   tif: str = 'DAY', **kwargs) -> Trade:
        """
        Place an order with IBKR.
        
        Args:
            symbol: Instrument symbol
            action: 'BUY' or 'SELL'
            quantity: Order quantity
            order_type: Order type (MKT, LMT, STP, etc.)
            price: Limit price for limit orders
            tif: Time in force (DAY, GTC, etc.)
            **kwargs: Additional order parameters
            
        Returns:
            IBKR Trade object
        """
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        if symbol not in self.instruments:
            raise ValueError(f"Instrument not configured: {symbol}")
        
        # Create contract
        config = self.instruments[symbol]
        contract = self.create_contract(
            symbol=config['symbol'],
            sec_type='FUT',
            exchange=config['exchange'],
            currency=config['currency']
        )
        
        # Create order
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        order.tif = tif
        
        if price is not None:
            order.lmtPrice = price
        
        # Additional order parameters
        for key, value in kwargs.items():
            setattr(order, key, value)
        
        # Place order
        trade = self.ib.placeOrder(contract, order)
        self.trades[trade.orderId] = trade
        
        logger.info(f"Placed {action} {quantity} {symbol} {order_type} order")
        return trade
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        if order_id not in self.trades:
            logger.warning(f"Order {order_id} not found")
            return False
        
        try:
            self.ib.cancelOrder(self.trades[order_id].order)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: int) -> Optional[OrderStatus]:
        """
        Get status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status or None if not found
        """
        if order_id not in self.trades:
            return None
        
        return self.trades[order_id].orderStatus
    
    def get_open_orders(self) -> List[Trade]:
        """
        Get all open orders.
        
        Returns:
            List of open trades
        """
        return [trade for trade in self.trades.values() 
                if trade.orderStatus in ['PendingSubmit', 'PreSubmitted', 'Submitted']]
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary information.
        
        Returns:
            Account summary dictionary
        """
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        try:
            summary = self.ib.accountSummary()
            return {tag: value for tag, _, value, _ in summary}
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            Dictionary of positions by symbol
        """
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        try:
            positions = self.ib.positions()
            return {
                pos.contract.symbol: {
                    'symbol': pos.contract.symbol,
                    'position': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_price': pos.marketPrice,
                    'unrealized_pnl': pos.unrealizedPNL,
                    'realized_pnl': pos.realizedPNL
                }
                for pos in positions
            }
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}
    
    def request_market_data(self, symbol: str, duration: str = '1 D',
                          bar_size: str = '1 min', what_to_show: str = 'TRADES') -> pd.DataFrame:
        """
        Request historical market data.
        
        Args:
            symbol: Instrument symbol
            duration: Data duration (e.g., '1 D', '1 W', '1 M')
            bar_size: Bar size (e.g., '1 min', '5 min', '1 hour')
            what_to_show: Data type to show (TRADES, MIDPOINT, BID_ASK)
            
        Returns:
            DataFrame with historical data
        """
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        if symbol not in self.instruments:
            raise ValueError(f"Instrument not configured: {symbol}")
        
        config = self.instruments[symbol]
        contract = self.create_contract(
            symbol=config['symbol'],
            sec_type='FUT',
            exchange=config['exchange'],
            currency=config['currency']
        )
        
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1
            )
            
            df = util.df(bars)
            df.set_index('date', inplace=True)
            
            # Cache the data
            cache_key = f"{symbol}_{duration}_{bar_size}"
            self.market_data[cache_key] = df
            
            logger.info(f"Retrieved {len(df)} bars of {symbol} data")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            raise
    
    def subscribe_real_time_data(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to real-time market data.
        
        Args:
            symbol: Instrument symbol
            callback: Callback function for market data updates
        """
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        if symbol not in self.instruments:
            raise ValueError(f"Instrument not configured: {symbol}")
        
        config = self.instruments[symbol]
        contract = self.create_contract(
            symbol=config['symbol'],
            sec_type='FUT',
            exchange=config['exchange'],
            currency=config['currency']
        )
        
        # Subscribe to market data
        self.ib.reqMktData(contract, '', False, False)
        
        # Store callback
        self.order_callbacks[id(contract)] = callback
        
        logger.info(f"Subscribed to real-time data for {symbol}")
    
    def calculate_transaction_costs(self, symbol: str, quantity: int, 
                                  price: float, order_type: str = 'MKT') -> Dict[str, float]:
        """
        Calculate estimated transaction costs.
        
        Args:
            symbol: Instrument symbol
            quantity: Order quantity
            price: Order price
            order_type: Order type
            
        Returns:
            Dictionary with cost breakdown
        """
        if symbol not in self.instruments:
            raise ValueError(f"Instrument not configured: {symbol}")
        
        config = self.instruments[symbol]
        
        # Base commission
        commission = config.get('commission_per_contract', 0.6) * quantity
        
        # Spread cost (assume 1 tick for market orders)
        spread_cost = config['tick_value'] * quantity
        
        # Impact cost (simplified model)
        # This is a toy model - in production, use more sophisticated models
        adv_estimate = 1000000  # Assumed daily ADV
        impact_bps = self.settings.get('execution', 'impact_bps', 0.5)
        impact_cost = (price * quantity * quantity / adv_estimate) * (impact_bps / 10000)
        
        total_cost = commission + spread_cost + impact_cost
        
        return {
            'commission': commission,
            'spread': spread_cost,
            'impact': impact_cost,
            'total': total_cost
        }
    
    def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """
        Get market depth (order book) data.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Market depth data
        """
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")
        
        if symbol not in self.instruments:
            raise ValueError(f"Instrument not configured: {symbol}")
        
        config = self.instruments[symbol]
        contract = self.create_contract(
            symbol=config['symbol'],
            sec_type='FUT',
            exchange=config['exchange'],
            currency=config['currency']
        )
        
        try:
            self.ib.reqMktDepth(contract, 5)  # Get 5 levels of depth
            # Note: This would need proper event handling to capture depth updates
            logger.info(f"Requested market depth for {symbol}")
            return {}
        except Exception as e:
            logger.error(f"Failed to get market depth for {symbol}: {e}")
            return {}
    
    def wait_for_order_fill(self, order_id: int, timeout: int = 30) -> bool:
        """
        Wait for an order to be filled.
        
        Args:
            order_id: Order ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            True if order was filled, False if timeout
        """
        if order_id not in self.trades:
            return False
        
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = self.get_order_status(order_id)
            if status in ['Filled', 'PartiallyFilled']:
                return True
            elif status in ['Cancelled', 'Inactive']:
                return False
            
            asyncio.sleep(0.1)
        
        return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class IBKRMarketDataClient:
    """
    Specialized client for market data operations.
    """
    
    def __init__(self, ibkr_client: IBKRClient):
        """
        Initialize market data client.
        
        Args:
            ibkr_client: IBKR client instance
        """
        self.ibkr_client = ibkr_client
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for an instrument.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Latest price or None if not available
        """
        try:
            # Request real-time market data
            df = self.ibkr_client.request_market_data(
                symbol, duration='1 min', bar_size='1 sec'
            )
            return df['close'].iloc[-1] if not df.empty else None
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            return None
    
    def get_bid_ask(self, symbol: str) -> Optional[tuple]:
        """
        Get current bid-ask spread.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Tuple of (bid, ask) or None if not available
        """
        # This would require proper market depth subscription
        # For now, return None as placeholder
        return None