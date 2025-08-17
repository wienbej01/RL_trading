"""
Monitoring dashboard for the RL trading system.

This module provides a comprehensive dashboard for monitoring
trading performance, risk metrics, and system health.
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
import threading
import time
from queue import Queue, Empty

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from .risk_monitor import RiskMonitor, RiskAlert, RiskLevel
from .analytics import TradingAnalytics, TradeAnalysis, PerformanceAnalysis

logger = get_logger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    update_interval: int = 5  # seconds
    max_alerts_display: int = 10
    max_trades_display: int = 50
    max_equity_points: int = 1000
    enable_real_time: bool = True
    enable_alerts: bool = True
    enable_performance: bool = True
    enable_risk: bool = True
    output_dir: str = "monitoring_dashboard"


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard.
    
    This class provides a real-time dashboard for monitoring
    trading performance, risk metrics, and system health.
    """
    
    def __init__(self, settings: Settings, config: Optional[DashboardConfig] = None):
        """
        Initialize monitoring dashboard.
        
        Args:
            settings: Configuration settings
            config: Dashboard configuration
        """
        self.settings = settings
        self.config = config or self._load_config()
        
        # Initialize monitoring components
        self.risk_monitor = RiskMonitor(settings)
        self.analytics = TradingAnalytics(settings)
        
        # Dashboard state
        self.dashboard_data = {}
        self.last_update = None
        self.update_count = 0
        
        # Real-time data
        self.real_time_data = {
            'equity': [],
            'position': [],
            'pnl': [],
            'alerts': [],
            'trades': [],
            'risk_metrics': {},
            'performance_metrics': {}
        }
        
        # Threading
        self.running = False
        self.update_thread = None
        self.data_queue = Queue()
        
        # Configuration
        self.dashboard_enabled = settings.get("monitoring", "dashboard_enabled", default=True)
        
        logger.info("Monitoring dashboard initialized")
    
    def _load_config(self) -> DashboardConfig:
        """
        Load dashboard configuration from settings.
        
        Returns:
            Dashboard configuration
        """
        config = DashboardConfig()
        
        config.update_interval = self.settings.get("monitoring", "dashboard_update_interval", default=5)
        config.max_alerts_display = self.settings.get("monitoring", "max_alerts_display", default=10)
        config.max_trades_display = self.settings.get("monitoring", "max_trades_display", default=50)
        config.max_equity_points = self.settings.get("monitoring", "max_equity_points", default=1000)
        config.enable_real_time = self.settings.get("monitoring", "enable_real_time", default=True)
        config.enable_alerts = self.settings.get("monitoring", "enable_alerts", default=True)
        config.enable_performance = self.settings.get("monitoring", "enable_performance", default=True)
        config.enable_risk = self.settings.get("monitoring", "enable_risk", default=True)
        config.output_dir = self.settings.get("monitoring", "dashboard_output_dir", default="monitoring_dashboard")
        
        return config
    
    def start_dashboard(self):
        """Start monitoring dashboard."""
        if not self.dashboard_enabled:
            logger.warning("Dashboard is disabled")
            return
        
        if self.running:
            logger.warning("Dashboard already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._dashboard_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Monitoring dashboard started")
    
    def stop_dashboard(self):
        """Stop monitoring dashboard."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        
        logger.info("Monitoring dashboard stopped")
    
    def _dashboard_loop(self):
        """Dashboard update loop."""
        while self.running:
            try:
                # Update dashboard data
                self._update_dashboard_data()
                
                # Process real-time data
                self._process_real_time_data()
                
                # Update dashboard
                self._update_dashboard()
                
                # Sleep
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in dashboard loop: {e}")
                time.sleep(5)
    
    def _update_dashboard_data(self):
        """Update dashboard data."""
        # Get risk monitor data
        risk_summary = self.risk_monitor.get_risk_summary()
        
        # Get analytics data
        dashboard_data = self.analytics.get_dashboard_data()
        
        # Combine data
        self.dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'risk_summary': risk_summary,
            'analytics': dashboard_data,
            'system_status': {
                'running': self.running,
                'update_count': self.update_count,
                'last_update': datetime.now().isoformat()
            }
        }
        
        self.last_update = datetime.now()
        self.update_count += 1
    
    def _process_real_time_data(self):
        """Process real-time data."""
        # Process queued data
        while not self.data_queue.empty():
            try:
                data_type, data = self.data_queue.get_nowait()
                
                if data_type == 'equity':
                    self.real_time_data['equity'].append(data)
                    # Limit data points
                    if len(self.real_time_data['equity']) > self.config.max_equity_points:
                        self.real_time_data['equity'] = self.real_time_data['equity'][-self.config.max_equity_points:]
                
                elif data_type == 'position':
                    self.real_time_data['position'].append(data)
                
                elif data_type == 'pnl':
                    self.real_time_data['pnl'].append(data)
                
                elif data_type == 'alert':
                    self.real_time_data['alerts'].append(data)
                    # Limit alerts
                    if len(self.real_time_data['alerts']) > self.config.max_alerts_display:
                        self.real_time_data['alerts'] = self.real_time_data['alerts'][-self.config.max_alerts_display:]
                
                elif data_type == 'trade':
                    self.real_time_data['trades'].append(data)
                    # Limit trades
                    if len(self.real_time_data['trades']) > self.config.max_trades_display:
                        self.real_time_data['trades'] = self.real_time_data['trades'][-self.config.max_trades_display:]
                
                elif data_type == 'risk_metrics':
                    self.real_time_data['risk_metrics'] = data
                
                elif data_type == 'performance_metrics':
                    self.real_time_data['performance_metrics'] = data
                
            except Empty:
                break
    
    def _update_dashboard(self):
        """Update dashboard display."""
        # This would update a web dashboard or GUI
        # For now, just log the update
        if self.update_count % 60 == 0:  # Log every 60 updates
            logger.info(f"Dashboard update #{self.update_count} - Equity: {self.dashboard_data['risk_summary']['equity']:.2f}")
    
    def add_real_time_data(self, data_type: str, data: Any):
        """
        Add real-time data to dashboard.
        
        Args:
            data_type: Type of data
            data: Data to add
        """
        self.data_queue.put((data_type, data))
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get current dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        return self.dashboard_data
    
    def get_real_time_data(self) -> Dict[str, Any]:
        """
        Get real-time data.
        
        Returns:
            Real-time data dictionary
        """
        return self.real_time_data
    
    def generate_dashboard_report(self, output_path: str):
        """
        Generate comprehensive dashboard report.
        
        Args:
            output_path: Output path for report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'dashboard_config': self.config.__dict__,
            'dashboard_data': self.dashboard_data,
            'real_time_data': self.real_time_data,
            'risk_monitor_data': {
                'risk_summary': self.risk_monitor.get_risk_summary(),
                'active_alerts': [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'type': alert.alert_type,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'acknowledged': alert.acknowledged
                    }
                    for alert in self.risk_monitor.get_active_alerts()
                ]
            },
            'analytics_data': self.analytics.get_dashboard_data(),
            'system_status': {
                'running': self.running,
                'update_count': self.update_count,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Dashboard report saved to {output_path}")
    
    def save_dashboard_snapshot(self, output_dir: str):
        """
        Save dashboard snapshot.
        
        Args:
            output_dir: Output directory for snapshot
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dashboard data
        dashboard_file = output_dir / "dashboard_data.json"
        with open(dashboard_file, 'w') as f:
            json.dump(self.dashboard_data, f, indent=2, default=str)
        
        # Save real-time data
        realtime_file = output_dir / "real_time_data.json"
        with open(realtime_file, 'w') as f:
            json.dump(self.real_time_data, f, indent=2, default=str)
        
        # Save risk monitor report
        risk_report = output_dir / "risk_report.json"
        self.risk_monitor.save_risk_report(risk_report)
        
        # Save analytics report
        analytics_report = output_dir / "analytics_report.json"
        self.analytics.generate_report(analytics_report)
        
        logger.info(f"Dashboard snapshot saved to {output_dir}")
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get dashboard summary.
        
        Returns:
            Dashboard summary dictionary
        """
        if not self.dashboard_data:
            return {}
        
        return {
            'timestamp': self.dashboard_data.get('timestamp'),
            'equity': self.dashboard_data.get('risk_summary', {}).get('equity', 0),
            'daily_pnl': self.dashboard_data.get('risk_summary', {}).get('daily_pnl', 0),
            'total_pnl': self.dashboard_data.get('risk_summary', {}).get('total_pnl', 0),
            'current_position': self.dashboard_data.get('risk_summary', {}).get('current_position', 0),
            'active_alerts': len(self.dashboard_data.get('risk_summary', {}).get('active_alerts', [])),
            'total_trades': self.dashboard_data.get('analytics', {}).get('trade_analysis', {}).get('total_trades', 0),
            'win_rate': self.dashboard_data.get('analytics', {}).get('trade_analysis', {}).get('win_rate', 0),
            'sharpe_ratio': self.dashboard_data.get('analytics', {}).get('performance_analysis', {}).get('sharpe_ratio', 0),
            'max_drawdown': self.dashboard_data.get('analytics', {}).get('performance_analysis', {}).get('max_drawdown', 0),
            'system_status': self.dashboard_data.get('system_status', {})
        }
    
    def reset_dashboard(self):
        """Reset dashboard data."""
        self.dashboard_data = {}
        self.real_time_data = {
            'equity': [],
            'position': [],
            'pnl': [],
            'alerts': [],
            'trades': [],
            'risk_metrics': {},
            'performance_metrics': {}
        }
        self.update_count = 0
        
        # Reset monitoring components
        self.risk_monitor.reset_daily_metrics()
        self.analytics.reset_data()
        
        logger.info("Dashboard reset")
    
    def export_dashboard_data(self, output_path: str, format: str = 'json'):
        """
        Export dashboard data.
        
        Args:
            output_path: Output path
            format: Export format ('json', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.dashboard_data, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Convert to DataFrame and save as CSV
            df = pd.json_normalize(self.dashboard_data)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dashboard data exported to {output_path}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            'overall_status': 'healthy',
            'components': {
                'risk_monitor': 'healthy',
                'analytics': 'healthy',
                'dashboard': 'healthy'
            },
            'metrics': {
                'update_frequency': f"{self.config.update_interval}s",
                'last_update': self.last_update.isoformat() if self.last_update else 'never',
                'update_count': self.update_count,
                'active_alerts': len(self.risk_monitor.get_active_alerts()),
                'equity': self.dashboard_data.get('risk_summary', {}).get('equity', 0),
                'drawdown': self.dashboard_data.get('analytics', {}).get('performance_analysis', {}).get('max_drawdown', 0)
            },
            'warnings': [],
            'errors': []
        }
        
        # Check for warnings
        if health_status['metrics']['active_alerts'] > 0:
            health_status['warnings'].append(f"{health_status['metrics']['active_alerts']} active alerts")
        
        if health_status['metrics']['drawdown'] < -0.05:  # 5% drawdown
            health_status['warnings'].append(f"High drawdown: {health_status['metrics']['drawdown']:.2%}")
        
        # Check for errors
        if not self.running:
            health_status['overall_status'] = 'error'
            health_status['errors'].append("Dashboard not running")
        
        if self.last_update and (datetime.now() - self.last_update).total_seconds() > 300:  # 5 minutes
            health_status['overall_status'] = 'warning'
            health_status['warnings'].append("Dashboard update lag detected")
        
        return health_status
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring."""
        if not self.config.enable_real_time:
            logger.warning("Real-time monitoring is disabled")
            return
        
        # Start risk monitoring
        self.risk_monitor.start_monitoring()
        
        logger.info("Real-time monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring."""
        # Stop risk monitoring
        self.risk_monitor.stop_monitoring()
        
        logger.info("Real-time monitoring stopped")