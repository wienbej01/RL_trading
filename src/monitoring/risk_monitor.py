"""
Risk monitoring module for the RL trading system.

This module provides comprehensive risk monitoring capabilities
including position limits, drawdown controls, and real-time alerts.
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from enum import Enum
import threading
import time
from queue import Queue, Empty

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..utils.metrics import calculate_performance_metrics, calculate_risk_metrics

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    timestamp: datetime
    alert_type: str
    severity: RiskLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size: int = 10
    max_daily_loss: float = 1000.0
    max_drawdown: float = 0.10  # 10%
    max_leverage: float = 2.0
    max_risk_per_trade: float = 0.02  # 2%
    max_consecutive_losses: int = 5
    max_position_concentration: float = 0.30  # 30%
    min_equity_threshold: float = 50000.0
    volatility_limit: float = 0.30  # 30% annualized
    correlation_limit: float = 0.80  # 80% max correlation


class RiskMonitor:
    """
    Comprehensive risk monitoring system.
    
    This class provides real-time risk monitoring with configurable
    limits, alerts, and automatic risk controls.
    """
    
    def __init__(self, settings: Settings, limits: Optional[RiskLimits] = None):
        """
        Initialize risk monitor.
        
        Args:
            settings: Configuration settings
            limits: Risk limits configuration
        """
        self.settings = settings
        self.limits = limits or self._load_limits()
        
        # Risk state
        self.current_position = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.equity_curve = []
        self.drawdown_curve = []
        self.trade_history = []
        self.position_history = []
        
        # Risk metrics
        self.risk_metrics = {}
        self.performance_metrics = {}
        
        # Alerts
        self.alerts = []
        self.alert_queue = Queue()
        self.active_alerts = []
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        
        # Configuration
        self.risk_enabled = settings.get("risk_monitoring", "enabled", default=True)
        self.alert_thresholds = settings.get("risk_monitoring", "alert_thresholds", default={})
        self.auto_risk_control = settings.get("risk_monitoring", "auto_risk_control", default=True)
        
        logger.info("Risk monitor initialized")
    
    def _load_limits(self) -> RiskLimits:
        """
        Load risk limits from settings.
        
        Returns:
            Risk limits configuration
        """
        limits = RiskLimits()
        
        # Load limits from settings
        limits.max_position_size = self.settings.get("risk_monitoring", "max_position_size", default=10)
        limits.max_daily_loss = self.settings.get("risk_monitoring", "max_daily_loss", default=1000.0)
        limits.max_drawdown = self.settings.get("risk_monitoring", "max_drawdown", default=0.10)
        limits.max_leverage = self.settings.get("risk_monitoring", "max_leverage", default=2.0)
        limits.max_risk_per_trade = self.settings.get("risk_monitoring", "max_risk_per_trade", default=0.02)
        limits.max_consecutive_losses = self.settings.get("risk_monitoring", "max_consecutive_losses", default=5)
        limits.max_position_concentration = self.settings.get("risk_monitoring", "max_position_concentration", default=0.30)
        limits.min_equity_threshold = self.settings.get("risk_monitoring", "min_equity_threshold", default=50000.0)
        limits.volatility_limit = self.settings.get("risk_monitoring", "volatility_limit", default=0.30)
        limits.correlation_limit = self.settings.get("risk_monitoring", "correlation_limit", default=0.80)
        
        return limits
    
    def update_position(self, position_change: int, price: float = 0.0):
        """
        Update position and check risk limits.
        
        Args:
            position_change: Change in position
            price: Current price for P&L calculation
        """
        if not self.risk_enabled:
            return
        
        # Update position
        old_position = self.current_position
        self.current_position += position_change
        
        # Log position change
        self.position_history.append({
            'timestamp': datetime.now(),
            'old_position': old_position,
            'position_change': position_change,
            'new_position': self.current_position,
            'price': price
        })
        
        # Check position limits
        self._check_position_limits()
        
        # Update P&L
        if price > 0:
            pnl_change = position_change * price
            self.daily_pnl += pnl_change
            self.total_pnl += pnl_change
        
        # Update equity curve
        self._update_equity_curve()
        
        # Check drawdown
        self._check_drawdown()
        
        # Check daily loss
        self._check_daily_loss()
        
        # Check consecutive losses
        self._check_consecutive_losses()
        
        # Update risk metrics
        self._update_risk_metrics()
    
    def _check_position_limits(self):
        """Check position limits."""
        # Check maximum position size
        if abs(self.current_position) > self.limits.max_position_size:
            alert = RiskAlert(
                timestamp=datetime.now(),
                alert_type="POSITION_SIZE_EXCEEDED",
                severity=RiskLevel.CRITICAL,
                message=f"Position size {self.current_position} exceeds limit {self.limits.max_position_size}",
                details={'current_position': self.current_position, 'limit': self.limits.max_position_size}
            )
            self.add_alert(alert)
            
            # Auto risk control
            if self.auto_risk_control:
                self._reduce_position_to_limit()
        
        # Check position concentration (simplified)
        if abs(self.current_position) > self.limits.max_position_concentration * 100:  # Assuming 100 shares per contract
            alert = RiskAlert(
                timestamp=datetime.now(),
                alert_type="POSITION_CONCENTRATION",
                severity=RiskLevel.HIGH,
                message=f"Position concentration too high: {self.current_position}",
                details={'current_position': self.current_position, 'concentration_limit': self.limits.max_position_concentration}
            )
            self.add_alert(alert)
    
    def _reduce_position_to_limit(self):
        """Reduce position to within limits."""
        if abs(self.current_position) > self.limits.max_position_size:
            reduction = abs(self.current_position) - self.limits.max_position_size
            if self.current_position > 0:
                self.current_position -= reduction
            else:
                self.current_position += reduction
            
            logger.warning(f"Position reduced to {self.current_position} due to risk limits")
    
    def _update_equity_curve(self):
        """Update equity curve."""
        # Calculate current equity (simplified)
        current_equity = self.settings.get("risk_monitoring", "initial_capital", default=100000.0) + self.total_pnl
        
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': current_equity,
            'pnl': self.total_pnl
        })
        
        # Calculate drawdown
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            peak = equity_series.cummax()
            drawdown = (equity_series - peak) / peak
            
            self.drawdown_curve.append({
                'timestamp': datetime.now(),
                'drawdown': drawdown.iloc[-1],
                'peak': peak.iloc[-1],
                'equity': equity_series.iloc[-1]
            })
    
    def _check_drawdown(self):
        """Check drawdown limits."""
        if not self.drawdown_curve:
            return
        
        current_drawdown = self.drawdown_curve[-1]['drawdown']
        
        if current_drawdown < -self.limits.max_drawdown:
            alert = RiskAlert(
                timestamp=datetime.now(),
                alert_type="DRAWDOWN_EXCEEDED",
                severity=RiskLevel.CRITICAL,
                message=f"Drawdown {current_drawdown:.2%} exceeds limit {self.limits.max_drawdown:.2%}",
                details={'current_drawdown': current_drawdown, 'limit': self.limits.max_drawdown}
            )
            self.add_alert(alert)
            
            # Auto risk control
            if self.auto_risk_control:
                self._emergency_stop()
    
    def _check_daily_loss(self):
        """Check daily loss limits."""
        if self.daily_pnl < -self.limits.max_daily_loss:
            alert = RiskAlert(
                timestamp=datetime.now(),
                alert_type="DAILY_LOSS_EXCEEDED",
                severity=RiskLevel.HIGH,
                message=f"Daily loss ${self.daily_pnl:.2f} exceeds limit ${self.limits.max_daily_loss:.2f}",
                details={'daily_pnl': self.daily_pnl, 'limit': self.limits.max_daily_loss}
            )
            self.add_alert(alert)
            
            # Auto risk control
            if self.auto_risk_control:
                self._close_all_positions()
    
    def _check_consecutive_losses(self):
        """Check consecutive losses."""
        if not self.trade_history:
            return
        
        # Count consecutive losses
        consecutive_losses = 0
        for trade in reversed(self.trade_history):
            if trade['pnl'] < 0:
                consecutive_losses += 1
            else:
                break
        
        if consecutive_losses >= self.limits.max_consecutive_losses:
            alert = RiskAlert(
                timestamp=datetime.now(),
                alert_type="CONSECUTIVE_LOSSES",
                severity=RiskLevel.MEDIUM,
                message=f"Consecutive losses {consecutive_losses} exceeds limit {self.limits.max_consecutive_losses}",
                details={'consecutive_losses': consecutive_losses, 'limit': self.limits.max_consecutive_losses}
            )
            self.add_alert(alert)
            
            # Auto risk control
            if self.auto_risk_control:
                self._reduce_trading_size()
    
    def _emergency_stop(self):
        """Emergency stop - close all positions."""
        logger.warning("Emergency stop triggered - closing all positions")
        self.current_position = 0
        self._close_all_positions()
    
    def _close_all_positions(self):
        """Close all positions."""
        logger.warning("Closing all positions due to risk limits")
        self.current_position = 0
    
    def _reduce_trading_size(self):
        """Reduce trading size."""
        logger.warning("Reducing trading size due to consecutive losses")
        # This would be implemented in the trading system
        # For now, just log the warning
    
    def _update_risk_metrics(self):
        """Update risk metrics."""
        if len(self.equity_curve) < 2:
            return
        
        # Calculate returns
        equity_series = pd.Series([e['equity'] for e in self.equity_curve])
        returns = equity_series.pct_change().dropna()
        
        # Calculate risk metrics
        self.risk_metrics = calculate_risk_metrics(returns)
        
        # Calculate performance metrics
        self.performance_metrics = calculate_performance_metrics(returns)
    
    def add_trade(self, trade: Dict[str, Any]):
        """
        Add trade to history.
        
        Args:
            trade: Trade dictionary
        """
        self.trade_history.append(trade)
        
        # Update consecutive losses check
        self._check_consecutive_losses()
    
    def add_alert(self, alert: RiskAlert):
        """
        Add risk alert.
        
        Args:
            alert: Risk alert
        """
        self.alerts.append(alert)
        self.alert_queue.put(alert)
        self.active_alerts.append(alert)
        
        logger.warning(f"Risk Alert: {alert.severity.value.upper()} - {alert.message}")
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """
        Get active (unacknowledged) alerts.
        
        Returns:
            List of active alerts
        """
        return [alert for alert in self.active_alerts if not alert.acknowledged]
    
    def acknowledge_alert(self, alert_id: int):
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
        """
        for alert in self.active_alerts:
            if id(alert) == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert.message}")
                break
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get risk summary.
        
        Returns:
            Risk summary dictionary
        """
        return {
            'current_position': self.current_position,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'equity': self.equity_curve[-1]['equity'] if self.equity_curve else 0,
            'drawdown': self.drawdown_curve[-1]['drawdown'] if self.drawdown_curve else 0,
            'risk_metrics': self.risk_metrics,
            'performance_metrics': self.performance_metrics,
            'active_alerts': len(self.get_active_alerts()),
            'total_alerts': len(self.alerts),
            'risk_enabled': self.risk_enabled,
            'auto_risk_control': self.auto_risk_control
        }
    
    def start_monitoring(self):
        """Start risk monitoring."""
        if self.monitoring:
            logger.warning("Risk monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            try:
                # Process alerts
                while not self.alert_queue.empty():
                    try:
                        alert = self.alert_queue.get_nowait()
                        # Alert processing would go here
                        # For now, just log the alert
                        logger.warning(f"Processing alert: {alert.message}")
                    except Empty:
                        break
                
                # Check risk limits periodically
                self._periodic_risk_check()
                
                # Sleep
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _periodic_risk_check(self):
        """Perform periodic risk checks."""
        # Check equity threshold
        if self.equity_curve:
            current_equity = self.equity_curve[-1]['equity']
            if current_equity < self.limits.min_equity_threshold:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    alert_type="EQUITY_THRESHOLD",
                    severity=RiskLevel.CRITICAL,
                    message=f"Equity ${current_equity:.2f} below threshold ${self.limits.min_equity_threshold:.2f}",
                    details={'current_equity': current_equity, 'threshold': self.limits.min_equity_threshold}
                )
                self.add_alert(alert)
        
        # Check volatility
        if self.risk_metrics and 'volatility' in self.risk_metrics:
            annual_volatility = self.risk_metrics['volatility'] * np.sqrt(252)
            if annual_volatility > self.limits.volatility_limit:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    alert_type="VOLATILITY_EXCEEDED",
                    severity=RiskLevel.HIGH,
                    message=f"Annual volatility {annual_volatility:.2%} exceeds limit {self.limits.volatility_limit:.2%}",
                    details={'volatility': annual_volatility, 'limit': self.limits.volatility_limit}
                )
                self.add_alert(alert)
    
    def save_risk_report(self, output_path: str):
        """
        Save risk report to file.
        
        Args:
            output_path: Output path for risk report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'risk_limits': self.limits.__dict__,
            'risk_summary': self.get_risk_summary(),
            'alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'alert_type': alert.alert_type,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'acknowledged': alert.acknowledged,
                    'details': alert.details
                }
                for alert in self.alerts
            ],
            'equity_curve': [
                {
                    'timestamp': entry['timestamp'].isoformat(),
                    'equity': entry['equity'],
                    'pnl': entry['pnl']
                }
                for entry in self.equity_curve
            ],
            'drawdown_curve': [
                {
                    'timestamp': entry['timestamp'].isoformat(),
                    'drawdown': entry['drawdown'],
                    'peak': entry['peak'],
                    'equity': entry['equity']
                }
                for entry in self.drawdown_curve
            ],
            'position_history': [
                {
                    'timestamp': entry['timestamp'].isoformat(),
                    'old_position': entry['old_position'],
                    'position_change': entry['position_change'],
                    'new_position': entry['new_position'],
                    'price': entry['price']
                }
                for entry in self.position_history
            ],
            'trade_history': self.trade_history
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Risk report saved to {output_path}")
    
    def reset_daily_metrics(self):
        """Reset daily metrics."""
        self.daily_pnl = 0.0
        self.trade_history = []
        logger.info("Daily metrics reset")
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """
        Get risk dashboard data.
        
        Returns:
            Risk dashboard data
        """
        return {
            'current_risk_metrics': self.risk_metrics,
            'current_performance_metrics': self.performance_metrics,
            'active_alerts': [
                {
                    'id': id(alert),
                    'timestamp': alert.timestamp.isoformat(),
                    'type': alert.alert_type,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'acknowledged': alert.acknowledged
                }
                for alert in self.get_active_alerts()
            ],
            'position_info': {
                'current_position': self.current_position,
                'position_history': len(self.position_history),
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl
            },
            'equity_info': {
                'current_equity': self.equity_curve[-1]['equity'] if self.equity_curve else 0,
                'max_drawdown': min([d['drawdown'] for d in self.drawdown_curve]) if self.drawdown_curve else 0,
                'equity_history': len(self.equity_curve)
            },
            'risk_limits': {
                'max_position_size': self.limits.max_position_size,
                'max_daily_loss': self.limits.max_daily_loss,
                'max_drawdown': self.limits.max_drawdown,
                'max_risk_per_trade': self.limits.max_risk_per_trade
            }
        }