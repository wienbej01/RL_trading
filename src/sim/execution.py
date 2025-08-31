"""
Execution simulation module for the RL trading system.

This module provides realistic execution simulation including
slippage, market impact, and transaction costs.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecParams:
    """Execution parameters for realistic trading simulation."""
    tick_value: float = 1.25  # Value per tick (e.g., $1.25 for MES)
    spread_ticks: int = 1     # Number of ticks in bid-ask spread
    impact_bps: float = 0.5   # Market impact in basis points
    commission_per_contract: float = 0.6  # Commission per contract
    min_commission: float = 1.0  # Minimum commission per trade
    slippage_bps: float = 0.2   # Slippage in basis points
    liquidity_threshold: float = 1000  # Minimum liquidity for execution


class ExecutionEngine:
    """
    Realistic execution engine for trading.
    
    This class simulates realistic market execution including
    slippage, market impact, and transaction costs.
    """
    
    def __init__(self, config=None, settings=None):
        """
        Initialize execution engine.
        
        Args:
            config: Configuration dictionary (for test compatibility)
            settings: Configuration settings (for production)
        """
        # Handle test compatibility mode
        if config is not None:
            self.config = config
            # Use dict.get() method for dictionary config
            transaction_cost = config.get('transaction_cost', 2.5) if isinstance(config, dict) else 2.5
            slippage = config.get('slippage', 0.01) if isinstance(config, dict) else 0.01
            
            self.exec_params = ExecParams(
                tick_value=1.25,
                spread_ticks=1,
                impact_bps=0.5,
                commission_per_contract=transaction_cost,
                min_commission=1.0,
                slippage_bps=slippage * 100,  # Convert to bps
                liquidity_threshold=1000
            )
        else:
            # Use settings for production
            self.settings = settings
            self.config = settings.get('execution', {}) if settings else {}
            self.exec_params = ExecParams(**self.config)
        
        # Market microstructure parameters
        self.order_book_depth = self.config.get('order_book_depth', 10) if isinstance(self.config, dict) else 10
        self.liquidity_profile = self.config.get('liquidity_profile', 'normal') if isinstance(self.config, dict) else 'normal'
        
        # Execution history
        self.execution_history: List[Dict] = []
        
        # Order tracking (for test compatibility)
        self.pending_orders = []
        self.filled_orders = []
        
    def estimate_transaction_costs(self, 
                                 quantity: int, 
                                 price: float, 
                                 side: str = 'buy') -> Dict[str, float]:
        """
        Estimate transaction costs for a trade.
        
        Args:
            quantity: Number of contracts
            price: Execution price
            side: Trade side ('buy' or 'sell')
            
        Returns:
            Dictionary with cost breakdown
        """
        # Base commission
        commission = max(
            self.exec_params.commission_per_contract * abs(quantity),
            self.exec_params.min_commission
        )
        
        # Slippage cost
        slippage_cost = (
            price * abs(quantity) * 
            self.exec_params.slippage_bps / 10000
        )
        
        # Market impact cost
        impact_cost = self._estimate_market_impact(quantity, price)
        
        # Total cost
        total_cost = commission + slippage_cost + impact_cost
        
        return {
            'commission': commission,
            'slippage': slippage_cost,
            'impact': impact_cost,
            'total': total_cost,
            'cost_per_contract': total_cost / abs(quantity) if quantity != 0 else 0
        }
    
    def _estimate_market_impact(self, quantity: int, price: float) -> float:
        """
        Estimate market impact using Almgren-Chriss model.
        
        Args:
            quantity: Trade size
            price: Current price
            
        Returns:
            Market impact cost
        """
        # Market impact parameters
        eta = self.exec_params.impact_bps / 10000  # Impact coefficient
        sigma = 0.20  # Daily volatility (20%)
        
        # Trade size as fraction of daily volume
        daily_volume = 1000000  # Assumed daily volume
        trade_fraction = abs(quantity) / daily_volume
        
        # Market impact (simplified Almgren-Chriss)
        impact = eta * sigma * np.sqrt(trade_fraction) * price * abs(quantity)
        
        return impact
    
    def simulate_execution(self, 
                          quantity: int, 
                          price: float, 
                          side: str = 'buy',
                          order_type: str = 'market') -> Dict[str, float]:
        """
        Simulate trade execution.
        
        Args:
            quantity: Number of contracts
            price: Reference price
            side: Trade side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', 'stop')
            
        Returns:
            Dictionary with execution details
        """
        if quantity == 0:
            return {
                'executed_quantity': 0,
                'executed_price': price,
                'total_cost': 0.0,
                'slippage': 0.0,
                'impact': 0.0,
                'commission': 0.0
            }
        
        # Calculate execution price with slippage
        if order_type == 'market':
            # Market order gets filled at worst bid/ask
            slippage_ticks = self.exec_params.spread_ticks / 2
            if side == 'buy':
                executed_price = price + slippage_ticks * self.exec_params.tick_value
            else:
                executed_price = price - slippage_ticks * self.exec_params.tick_value
        else:
            # Limit/stop orders get filled at specified price
            executed_price = price
        
        # Add market impact
        impact_cost = self._estimate_market_impact(quantity, price)
        impact_adjustment = impact_cost / (quantity * price) if quantity != 0 else 0
        
        if side == 'buy':
            executed_price *= (1 + impact_adjustment)
        else:
            executed_price *= (1 - impact_adjustment)
        
        # Calculate costs
        costs = self.estimate_transaction_costs(quantity, executed_price, side)
        
        # Record execution
        execution_record = {
            'timestamp': pd.Timestamp.now(),
            'quantity': quantity,
            'side': side,
            'order_type': order_type,
            'reference_price': price,
            'executed_price': executed_price,
            'total_cost': costs['total'],
            'slippage': costs['slippage'],
            'impact': costs['impact'],
            'commission': costs['commission']
        }
        
        self.execution_history.append(execution_record)
        
        return {
            'executed_quantity': quantity,
            'executed_price': executed_price,
            'total_cost': costs['total'],
            'slippage': costs['slippage'],
            'impact': costs['impact'],
            'commission': costs['commission']
        }
    
    def simulate_partial_fill(self, 
                             quantity: int, 
                             price: float, 
                             side: str = 'buy',
                             available_liquidity: Optional[float] = None) -> Dict[str, float]:
        """
        Simulate partial fill based on available liquidity.
        
        Args:
            quantity: Requested quantity
            price: Reference price
            side: Trade side ('buy' or 'sell')
            available_liquidity: Available liquidity at price level
            
        Returns:
            Dictionary with execution details
        """
        if available_liquidity is None:
            available_liquidity = self.exec_params.liquidity_threshold
        
        # Determine fill quantity
        if abs(quantity) <= available_liquidity:
            fill_quantity = quantity
        else:
            # Handle NaN values in quantity
            if np.isnan(quantity) or np.isinf(quantity):
                fill_quantity = 0
            else:
                fill_quantity = int(np.sign(quantity) * available_liquidity)
        
        # Simulate execution
        result = self.simulate_execution(fill_quantity, price, side)
        
        return {
            **result,
            'fill_ratio': fill_quantity / quantity if quantity != 0 else 1.0,
            'remaining_quantity': quantity - fill_quantity
        }
    
    def simulate_iceberg_order(self, 
                              total_quantity: int, 
                              price: float, 
                              side: str = 'buy',
                              display_size: int = 10) -> List[Dict[str, float]]:
        """
        Simulate iceberg order execution.
        
        Args:
            total_quantity: Total quantity to trade
            price: Reference price
            side: Trade side ('buy' or 'sell')
            display_size: Size of visible portion
            
        Returns:
            List of execution records
        """
        executions = []
        remaining_quantity = total_quantity
        
        while remaining_quantity != 0:
            # Determine current fill size
            current_fill = min(abs(remaining_quantity), display_size)
            # Handle NaN values in remaining_quantity
            if np.isnan(remaining_quantity) or np.isinf(remaining_quantity):
                current_fill = 0
            else:
                current_fill = int(np.sign(remaining_quantity) * current_fill)
            
            # Simulate execution
            execution = self.simulate_execution(current_fill, price, side)
            executions.append(execution)
            
            # Update remaining quantity
            remaining_quantity -= current_fill
            
            # Add some randomness to timing
            if remaining_quantity != 0:
                # Random delay between executions
                delay = np.random.exponential(0.1)  # 100ms average delay
                # In a real implementation, this would be handled by the event loop
        
        return executions
    
    def simulate_vwap_execution(self, 
                               quantity: int, 
                               price_series: pd.Series,
                               side: str = 'buy') -> Dict[str, float]:
        """
        Simulate VWAP execution.
        
        Args:
            quantity: Total quantity to trade
            price_series: Series of prices for VWAP calculation
            side: Trade side ('buy' or 'sell')
            
        Returns:
            Dictionary with VWAP execution details
        """
        if len(price_series) == 0:
            return {
                'executed_quantity': 0,
                'vwap': 0.0,
                'total_cost': 0.0,
                'slippage': 0.0,
                'impact': 0.0,
                'commission': 0.0
            }
        
        # Calculate VWAP
        vwap = (price_series * price_series).sum() / price_series.sum()
        
        # Simulate execution at VWAP
        result = self.simulate_execution(quantity, vwap, side)
        
        return {
            **result,
            'vwap': vwap,
            'execution_type': 'vwap'
        }
    
    def simulate_twap_execution(self, 
                               quantity: int, 
                               price_series: pd.Series,
                               side: str = 'buy',
                               time_horizon: int = 60) -> List[Dict[str, float]]:
        """
        Simulate TWAP execution.
        
        Args:
            quantity: Total quantity to trade
            price_series: Series of prices for TWAP calculation
            side: Trade side ('buy' or 'sell')
            time_horizon: Time horizon in minutes
            
        Returns:
            List of TWAP execution records
        """
        if len(price_series) == 0:
            return []
        
        # Calculate number of slices
        num_slices = min(len(price_series), time_horizon)
        slice_quantity = quantity / num_slices
        
        executions = []
        
        for i in range(num_slices):
            # Get price for this slice
            slice_price = price_series.iloc[i]
            
            # Simulate execution
            execution = self.simulate_execution(
                int(slice_quantity), 
                slice_price, 
                side
            )
            executions.append(execution)
        
        return executions
    
    def get_execution_statistics(self) -> Dict[str, float]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self.execution_history:
            return {}
        
        df = pd.DataFrame(self.execution_history)
        
        stats = {
            'total_trades': len(df),
            'total_volume': df['quantity'].abs().sum(),
            'avg_slippage_bps': (df['slippage'] / (df['executed_price'] * df['quantity'].abs()) * 10000).mean(),
            'avg_impact_bps': (df['impact'] / (df['executed_price'] * df['quantity'].abs()) * 10000).mean(),
            'avg_commission': df['commission'].mean(),
            'total_costs': df['total_cost'].sum(),
            'cost_to_volume_ratio': df['total_cost'].sum() / df['quantity'].abs().sum() if df['quantity'].abs().sum() > 0 else 0
        }
        
        return stats
    
    def clear_execution_history(self) -> None:
        """Clear execution history."""
        self.execution_history = []
        logger.info("Execution history cleared")
    
    def save_execution_history(self, filepath: str) -> None:
        """Save execution history to file."""
        if not self.execution_history:
            logger.warning("No execution history to save")
            return
        
        df = pd.DataFrame(self.execution_history)
        df.to_csv(filepath, index=False)
        logger.info(f"Execution history saved to {filepath}")
    
    def load_execution_history(self, filepath: str) -> None:
        """Load execution history from file."""
        try:
            df = pd.read_csv(filepath)
            self.execution_history = df.to_dict('records')
            logger.info(f"Execution history loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading execution history: {e}")


def estimate_tc(position: int, price: float, exec_sim: ExecutionEngine) -> float:
    """
    Estimate transaction costs for a position change.
    
    Args:
        position: Position size change
        price: Current price
        exec_sim: Execution engine
        
    Returns:
        Total transaction costs
    """
    if position == 0:
        return 0.0

    side = 'buy' if position > 0 else 'sell'
    costs: Dict[str, float] = exec_sim.estimate_transaction_costs(abs(position), price, side)
    commission = float(costs.get('commission', 0.0))
    slippage = float(costs.get('slippage', 0.0))
    impact = float(costs.get('impact', 0.0))
    return commission + slippage + impact


@dataclass
class Order:
    """Order data structure for execution simulation."""
    symbol: str = ""
    side: str = "BUY"
    quantity: int = 0
    order_type: str = "MARKET"
    price: float = 0.0
    limit_price: float = 0.0
    timestamp: pd.Timestamp = None
    status: str = "PENDING"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()


@dataclass
class Position:
    """Position data structure for execution simulation."""
    symbol: str = ""
    quantity: int = 0
    avg_price: float = 0.0
    average_cost: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: pd.Timestamp = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()
        if self.avg_price == 0 and self.average_cost > 0:
            self.avg_price = self.average_cost
        elif self.average_cost == 0 and self.avg_price > 0:
            self.average_cost = self.avg_price
    
    @property
    def market_value(self) -> float:
        """Calculate market value of position."""
        return self.quantity * self.average_cost
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.quantity == 0
    
    def add_fill(self, quantity: int, price: float, timestamp: pd.Timestamp):
        """Add a fill to the position."""
        if self.quantity == 0:
            # New position
            self.quantity = quantity
            self.avg_price = price
            self.average_cost = price
        else:
            # Add to existing position
            total_quantity = self.quantity + quantity
            if total_quantity != 0:
                self.avg_price = (self.quantity * self.avg_price + quantity * price) / total_quantity
                self.average_cost = self.avg_price
            self.quantity = total_quantity
        
        self.timestamp = timestamp


# Add methods expected by tests
def execute_order(self, order: Order, market_data: Dict) -> Optional[Dict]:
    """
    Execute an order based on market data.
    
    Args:
        order: Order to execute
        market_data: Current market data
        
    Returns:
        Fill information if order was filled, None otherwise
    """
    # Check if order should fill based on market data
    fill_probability = self.config.get('fill_probability', 0.95)
    
    if np.random.random() > fill_probability:
        return None
    
    # Determine fill price
    if order.order_type == 'MARKET':
        if order.side == 'BUY':
            fill_price = market_data.get('ask_price', market_data.get('bid_price', 0) + 0.5)
        else:
            fill_price = market_data.get('bid_price', market_data.get('ask_price', 0) - 0.5)
    elif order.order_type == 'LIMIT':
        if order.side == 'BUY' and order.limit_price >= market_data.get('ask_price', float('inf')):
            fill_price = market_data.get('ask_price', order.limit_price)
        elif order.side == 'SELL' and order.limit_price <= market_data.get('bid_price', 0):
            fill_price = market_data.get('bid_price', order.limit_price)
        else:
            return None
    else:
        return None
    
    # Apply slippage
    fill_price = self.apply_slippage(fill_price, order.quantity, order.side)
    
    # Calculate transaction cost
    fill_value = fill_price * abs(order.quantity)
    transaction_cost = self.calculate_transaction_cost(fill_value)
    
    # Create fill record
    fill = {
        'symbol': order.symbol,
        'quantity': order.quantity,
        'side': order.side,
        'fill_price': fill_price,
        'fill_time': pd.Timestamp.now(),
        'transaction_cost': transaction_cost
    }
    
    # Update order status
    order.status = 'FILLED'
    self.filled_orders.append(fill)
    
    return fill


def apply_slippage(self, base_price: float, quantity: int, side: str) -> float:
    """
    Apply slippage to base price.
    
    Args:
        base_price: Base price
        quantity: Order quantity
        side: Order side
        
    Returns:
        Price with slippage applied
    """
    slippage_bps = self.exec_params.slippage_bps
    slippage_amount = base_price * slippage_bps / 10000
    
    if side == 'BUY':
        return base_price + slippage_amount
    else:
        return base_price - slippage_amount


def calculate_transaction_cost(self, fill_value: float) -> float:
    """
    Calculate transaction cost.
    
    Args:
        fill_value: Value of the fill
        
    Returns:
        Transaction cost
    """
    return self.exec_params.commission_per_contract


# Add methods to ExecutionEngine class
ExecutionEngine.execute_order = execute_order
ExecutionEngine.apply_slippage = apply_slippage
ExecutionEngine.calculate_transaction_cost = calculate_transaction_cost


# Alias for compatibility
ExecutionSimulator = ExecutionEngine