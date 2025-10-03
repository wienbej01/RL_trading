"""
Alert system for performance degradation and other events.

This module provides a comprehensive alert system for monitoring
multi-ticker RL trading strategies, including performance degradation,
system errors, and other events.
"""

import logging
import smtplib
import json
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

import pandas as pd
import requests
from jinja2 import Template

from ..utils.logging import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)


class NotificationChannel:
    """
    Base class for notification channels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notification channel.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
    def send_notification(self, alert: Dict[str, Any]) -> bool:
        """
        Send notification.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            return self._send(alert)
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
            
    def _send(self, alert: Dict[str, Any]) -> bool:
        """
        Send notification (implementation specific).
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement _send method")


class EmailNotificationChannel(NotificationChannel):
    """
    Email notification channel.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email notification channel.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email', self.username)
        self.to_emails = config.get('to_emails', [])
        self.template_path = config.get('template_path')
        
        # Load email template
        if self.template_path and Path(self.template_path).exists():
            with open(self.template_path, 'r') as f:
                self.template = Template(f.read())
        else:
            # Use default template
            self.template = Template("""
            <html>
            <body>
                <h2>Trading System Alert</h2>
                <p><strong>Type:</strong> {{ alert.type }}</p>
                <p><strong>Severity:</strong> {{ alert.severity }}</p>
                <p><strong>Message:</strong> {{ alert.message }}</p>
                <p><strong>Timestamp:</strong> {{ alert.timestamp }}</p>
                {% if 'ticker' in alert %}
                <p><strong>Ticker:</strong> {{ alert.ticker }}</p>
                {% endif %}
                {% if 'value' in alert and 'threshold' in alert %}
                <p><strong>Value:</strong> {{ alert.value }}</p>
                <p><strong>Threshold:</strong> {{ alert.threshold }}</p>
                {% endif %}
            </body>
            </html>
            """)
            
    def _send(self, alert: Dict[str, Any]) -> bool:
        """
        Send email notification.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.username or not self.password or not self.to_emails:
            logger.error("Email configuration incomplete")
            return False
            
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Trading System Alert: {alert['type'].replace('_', ' ').title()}"
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        
        # Add HTML body
        html_content = self.template.render(alert=alert)
        msg.attach(MIMEText(html_content, 'html'))
        
        # Send email
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            logger.info(f"Email notification sent for {alert['type']}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """
    Slack notification channel.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Slack notification channel.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'Trading Bot')
        self.icon_emoji = config.get('icon_emoji', ':warning:')
        
        # Severity to color mapping
        self.severity_colors = {
            'high': 'danger',
            'medium': 'warning',
            'low': 'good'
        }
        
    def _send(self, alert: Dict[str, Any]) -> bool:
        """
        Send Slack notification.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            logger.error("Slack webhook URL not configured")
            return False
            
        # Create message
        color = self.severity_colors.get(alert['severity'], '#808080')
        
        message = {
            'channel': self.channel,
            'username': self.username,
            'icon_emoji': self.icon_emoji,
            'attachments': [
                {
                    'color': color,
                    'title': f"Trading System Alert: {alert['type'].replace('_', ' ').title()}",
                    'text': alert['message'],
                    'fields': [
                        {
                            'title': 'Severity',
                            'value': alert['severity'].title(),
                            'short': True
                        },
                        {
                            'title': 'Timestamp',
                            'value': alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                            'short': True
                        }
                    ]
                }
            ]
        }
        
        # Add optional fields
        if 'ticker' in alert:
            message['attachments'][0]['fields'].append({
                'title': 'Ticker',
                'value': alert['ticker'],
                'short': True
            })
            
        if 'value' in alert and 'threshold' in alert:
            message['attachments'][0]['fields'].extend([
                {
                    'title': 'Value',
                    'value': str(alert['value']),
                    'short': True
                },
                {
                    'title': 'Threshold',
                    'value': str(alert['threshold']),
                    'short': True
                }
            ])
            
        # Send message
        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info(f"Slack notification sent for {alert['type']}")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """
    Webhook notification channel.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize webhook notification channel.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 30)
        
    def _send(self, alert: Dict[str, Any]) -> bool:
        """
        Send webhook notification.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            logger.error("Webhook URL not configured")
            return False
            
        # Convert datetime to string for JSON serialization
        alert_copy = alert.copy()
        if 'timestamp' in alert_copy:
            alert_copy['timestamp'] = alert_copy['timestamp'].isoformat()
            
        # Send webhook
        try:
            response = requests.post(
                self.webhook_url,
                json=alert_copy,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook notification sent for {alert['type']}")
                return True
            else:
                logger.error(f"Webhook error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False


class AlertSystem:
    """
    Alert system for performance degradation and other events.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.alerts = []
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.notification_channels = []
        
        # Initialize notification channels
        self._init_notification_channels()
        
        # Default thresholds
        self.default_thresholds = {
            'drawdown': 0.1,  # 10% drawdown
            'sharpe_ratio': 0.5,  # Sharpe ratio below 0.5
            'win_rate': 0.4,  # Win rate below 40%
            'volatility': 0.3,  # Volatility above 30%
            'max_position_duration': 5,  # Max position duration in days
            'consecutive_losses': 5,  # Consecutive losses
            'regime_change': True,  # Alert on regime change
            'system_error': True,  # Alert on system errors
            'data_quality': True,  # Alert on data quality issues
            'model_performance': True,  # Alert on model performance issues
            'execution_latency': 1.0,  # Execution latency in seconds
            'api_error_rate': 0.05,  # API error rate (5%)
            'memory_usage': 0.9,  # Memory usage (90%)
            'cpu_usage': 0.9,  # CPU usage (90%)
            'disk_usage': 0.9  # Disk usage (90%)
        }
        
        # Merge with config thresholds
        self.thresholds = {**self.default_thresholds, **self.alert_thresholds}
        
        # Alert history
        self.alert_history = []
        self.max_history_size = config.get('max_history_size', 1000)
        
        # Alert suppression
        self.suppression_rules = config.get('suppression_rules', {})
        self.suppressed_alerts = {}
        
        # Alert aggregation
        self.aggregation_rules = config.get('aggregation_rules', {})
        self.aggregated_alerts = {}
        
    def _init_notification_channels(self) -> None:
        """Initialize notification channels."""
        channels_config = self.config.get('notification_channels', {})
        
        # Email channel
        if 'email' in channels_config:
            self.notification_channels.append(
                EmailNotificationChannel(channels_config['email'])
            )
            
        # Slack channel
        if 'slack' in channels_config:
            self.notification_channels.append(
                SlackNotificationChannel(channels_config['slack'])
            )
            
        # Webhook channel
        if 'webhook' in channels_config:
            self.notification_channels.append(
                WebhookNotificationChannel(channels_config['webhook'])
            )
            
    def add_alert(self, alert: Dict[str, Any]) -> None:
        """
        Add alert to system.
        
        Args:
            alert: Alert dictionary
        """
        # Check if alert should be suppressed
        if self._should_suppress_alert(alert):
            logger.debug(f"Alert suppressed: {alert['type']}")
            return
            
        # Check if alert should be aggregated
        aggregated_alert = self._aggregate_alert(alert)
        if aggregated_alert:
            alert = aggregated_alert
            
        # Add timestamp if not present
        if 'timestamp' not in alert:
            alert['timestamp'] = datetime.now()
            
        # Add to alerts list
        self.alerts.append(alert)
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
            
        # Send notifications
        self._send_notifications(alert)
        
        # Log alert
        self._log_alert(alert)
        
    def _should_suppress_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Check if alert should be suppressed.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if alert should be suppressed, False otherwise
        """
        alert_type = alert['type']
        
        # Check suppression rules
        if alert_type in self.suppression_rules:
            rule = self.suppression_rules[alert_type]
            
            # Check time-based suppression
            if 'suppress_for' in rule:
                suppress_for = timedelta(seconds=rule['suppress_for'])
                
                # Check if we had a similar alert recently
                if alert_type in self.suppressed_alerts:
                    last_alert_time = self.suppressed_alerts[alert_type]
                    if datetime.now() - last_alert_time < suppress_for:
                        return True
                        
                # Update suppression time
                self.suppressed_alerts[alert_type] = datetime.now()
                
            # Check count-based suppression
            if 'max_count' in rule and 'time_window' in rule:
                max_count = rule['max_count']
                time_window = timedelta(seconds=rule['time_window'])
                
                # Count similar alerts in time window
                cutoff_time = datetime.now() - time_window
                recent_alerts = [
                    a for a in self.alert_history
                    if a['type'] == alert_type and a['timestamp'] > cutoff_time
                ]
                
                if len(recent_alerts) >= max_count:
                    return True
                    
        return False
        
    def _aggregate_alert(self, alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Aggregate alert if needed.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Aggregated alert or None
        """
        alert_type = alert['type']
        
        # Check aggregation rules
        if alert_type in self.aggregation_rules:
            rule = self.aggregation_rules[alert_type]
            
            if 'aggregate_for' in rule:
                aggregate_for = timedelta(seconds=rule['aggregate_for'])
                
                # Check if we have an active aggregation
                if alert_type in self.aggregated_alerts:
                    agg_alert = self.aggregated_alerts[alert_type]
                    
                    # Check if aggregation window is still active
                    if datetime.now() - agg_alert['first_timestamp'] < aggregate_for:
                        # Update aggregated alert
                        agg_alert['count'] += 1
                        agg_alert['last_timestamp'] = datetime.now()
                        
                        # Update severity if needed
                        if 'severity_upgrade' in rule:
                            if agg_alert['count'] >= rule['severity_upgrade']['count']:
                                agg_alert['severity'] = rule['severity_upgrade']['severity']
                                
                        # Don't send individual alert
                        return None
                    else:
                        # Send aggregated alert and start new one
                        self._send_notifications(agg_alert)
                        self._log_alert(agg_alert)
                        
                        # Start new aggregation
                        self.aggregated_alerts[alert_type] = {
                            'type': alert_type,
                            'severity': alert['severity'],
                            'message': alert['message'],
                            'count': 1,
                            'first_timestamp': datetime.now(),
                            'last_timestamp': datetime.now(),
                            'aggregated': True
                        }
                        
                        return None
                else:
                    # Start new aggregation
                    self.aggregated_alerts[alert_type] = {
                        'type': alert_type,
                        'severity': alert['severity'],
                        'message': alert['message'],
                        'count': 1,
                        'first_timestamp': datetime.now(),
                        'last_timestamp': datetime.now(),
                        'aggregated': True
                    }
                    
                    return None
                    
        return None
        
    def _send_notifications(self, alert: Dict[str, Any]) -> None:
        """
        Send notifications for alert.
        
        Args:
            alert: Alert dictionary
        """
        for channel in self.notification_channels:
            try:
                channel.send_notification(alert)
            except Exception as e:
                logger.error(f"Error sending notification via {channel.__class__.__name__}: {e}")
                
    def _log_alert(self, alert: Dict[str, Any]) -> None:
        """
        Log alert.
        
        Args:
            alert: Alert dictionary
        """
        severity = alert['severity'].upper()
        message = alert['message']
        
        if 'ticker' in alert:
            message = f"[{alert['ticker']}] {message}"
            
        if alert['severity'] == 'high':
            logger.error(f"ALERT [{severity}]: {message}")
        elif alert['severity'] == 'medium':
            logger.warning(f"ALERT [{severity}]: {message}")
        else:
            logger.info(f"ALERT [{severity}]: {message}")
            
    def check_performance_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for performance alerts.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        # Check drawdown
        if metrics.get('max_drawdown', 0) > self.thresholds['drawdown']:
            alert = {
                'type': 'drawdown',
                'severity': 'high' if metrics['max_drawdown'] > 0.15 else 'medium',
                'message': f"Drawdown exceeded threshold: {metrics['max_drawdown']:.2%} > {self.thresholds['drawdown']:.2%}",
                'value': metrics['max_drawdown'],
                'threshold': self.thresholds['drawdown']
            }
            triggered_alerts.append(alert)
            
        # Check Sharpe ratio
        if metrics.get('sharpe_ratio', 0) < self.thresholds['sharpe_ratio']:
            alert = {
                'type': 'sharpe_ratio',
                'severity': 'medium',
                'message': f"Sharpe ratio below threshold: {metrics['sharpe_ratio']:.2f} < {self.thresholds['sharpe_ratio']:.2f}",
                'value': metrics['sharpe_ratio'],
                'threshold': self.thresholds['sharpe_ratio']
            }
            triggered_alerts.append(alert)
            
        # Check win rate
        if metrics.get('win_rate', 0) < self.thresholds['win_rate']:
            alert = {
                'type': 'win_rate',
                'severity': 'medium',
                'message': f"Win rate below threshold: {metrics['win_rate']:.2%} < {self.thresholds['win_rate']:.2%}",
                'value': metrics['win_rate'],
                'threshold': self.thresholds['win_rate']
            }
            triggered_alerts.append(alert)
            
        # Check volatility
        if metrics.get('annual_volatility', 0) > self.thresholds['volatility']:
            alert = {
                'type': 'volatility',
                'severity': 'medium',
                'message': f"Volatility exceeded threshold: {metrics['annual_volatility']:.2%} > {self.thresholds['volatility']:.2%}",
                'value': metrics['annual_volatility'],
                'threshold': self.thresholds['volatility']
            }
            triggered_alerts.append(alert)
            
        # Add alerts to system
        for alert in triggered_alerts:
            self.add_alert(alert)
            
        return triggered_alerts
        
    def check_trade_alerts(self, trade: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for trade alerts.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        # Check position duration
        if 'entry_time' in trade and 'exit_time' in trade:
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            duration = (exit_time - entry_time).total_seconds() / 86400  # days
            
            if duration > self.thresholds['max_position_duration']:
                alert = {
                    'type': 'position_duration',
                    'severity': 'low',
                    'message': f"Position duration exceeded threshold: {duration:.1f} days > {self.thresholds['max_position_duration']} days",
                    'value': duration,
                    'threshold': self.thresholds['max_position_duration'],
                    'ticker': trade.get('ticker', 'unknown')
                }
                triggered_alerts.append(alert)
                
        # Add alerts to system
        for alert in triggered_alerts:
            self.add_alert(alert)
            
        return triggered_alerts
        
    def check_regime_alert(self, from_regime: str, to_regime: str) -> Dict[str, Any]:
        """
        Check for regime change alert.
        
        Args:
            from_regime: Previous regime
            to_regime: New regime
            
        Returns:
            Triggered alert or None
        """
        if not self.thresholds['regime_change']:
            return None
            
        alert = {
            'type': 'regime_change',
            'severity': 'low',
            'message': f"Market regime changed from {from_regime} to {to_regime}",
            'from_regime': from_regime,
            'to_regime': to_regime
        }
        
        self.add_alert(alert)
        return alert
        
    def add_system_error_alert(self, error_message: str, error_type: str = 'system_error') -> Dict[str, Any]:
        """
        Add system error alert.
        
        Args:
            error_message: Error message
            error_type: Type of error
            
        Returns:
            Created alert
        """
        if not self.thresholds['system_error']:
            return None
            
        alert = {
            'type': error_type,
            'severity': 'high',
            'message': f"System error: {error_message}"
        }
        
        self.add_alert(alert)
        return alert
        
    def add_data_quality_alert(self, data_issue: str, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Add data quality alert.
        
        Args:
            data_issue: Data quality issue
            ticker: Ticker symbol (optional)
            
        Returns:
            Created alert
        """
        if not self.thresholds['data_quality']:
            return None
            
        alert = {
            'type': 'data_quality',
            'severity': 'medium',
            'message': f"Data quality issue: {data_issue}"
        }
        
        if ticker:
            alert['ticker'] = ticker
            alert['message'] = f"[{ticker}] Data quality issue: {data_issue}"
            
        self.add_alert(alert)
        return alert
        
    def add_model_performance_alert(self, performance_issue: str, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Add model performance alert.
        
        Args:
            performance_issue: Performance issue
            ticker: Ticker symbol (optional)
            
        Returns:
            Created alert
        """
        if not self.thresholds['model_performance']:
            return None
            
        alert = {
            'type': 'model_performance',
            'severity': 'medium',
            'message': f"Model performance issue: {performance_issue}"
        }
        
        if ticker:
            alert['ticker'] = ticker
            alert['message'] = f"[{ticker}] Model performance issue: {performance_issue}"
            
        self.add_alert(alert)
        return alert
        
    def add_execution_alert(self, execution_issue: str, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Add execution alert.
        
        Args:
            execution_issue: Execution issue
            ticker: Ticker symbol (optional)
            
        Returns:
            Created alert
        """
        alert = {
            'type': 'execution',
            'severity': 'high',
            'message': f"Execution issue: {execution_issue}"
        }
        
        if ticker:
            alert['ticker'] = ticker
            alert['message'] = f"[{ticker}] Execution issue: {execution_issue}"
            
        self.add_alert(alert)
        return alert
        
    def add_resource_alert(self, resource_type: str, usage: float, threshold: float) -> Dict[str, Any]:
        """
        Add resource usage alert.
        
        Args:
            resource_type: Type of resource (cpu, memory, disk)
            usage: Current usage
            threshold: Threshold
            
        Returns:
            Created alert
        """
        threshold_key = f"{resource_type}_usage"
        if threshold_key not in self.thresholds:
            return None
            
        if usage < self.thresholds[threshold_key]:
            return None
            
        alert = {
            'type': f"{resource_type}_usage",
            'severity': 'high' if usage > 0.95 else 'medium',
            'message': f"{resource_type.title()} usage exceeded threshold: {usage:.1%} > {threshold:.1%}",
            'value': usage,
            'threshold': threshold
        }
        
        self.add_alert(alert)
        return alert
        
    def get_active_alerts(self, severity: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get active alerts.
        
        Args:
            severity: Filter by severity (optional)
            hours: Number of hours to look back
            
        Returns:
            List of active alerts
        """
        # Get alerts from specified time period
        cutoff_time = datetime.now() - timedelta(hours=hours)
        active_alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
        
        # Filter by severity
        if severity:
            active_alerts = [alert for alert in active_alerts if alert['severity'] == severity]
            
        # Sort by timestamp (newest first)
        active_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return active_alerts
        
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert summary.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Alert summary dictionary
        """
        # Get alerts from specified time period
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
        
        # Count by severity
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for alert in recent_alerts:
            severity = alert['severity']
            if severity in severity_counts:
                severity_counts[severity] += 1
                
        # Count by type
        type_counts = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            if alert_type not in type_counts:
                type_counts[alert_type] = 0
            type_counts[alert_type] += 1
            
        # Get most recent alert
        most_recent_alert = recent_alerts[0] if recent_alerts else None
        
        # Get most severe alert
        most_severe_alert = None
        if recent_alerts:
            # Sort by severity (high > medium > low)
            severity_order = {'high': 3, 'medium': 2, 'low': 1}
            most_severe_alert = max(recent_alerts, key=lambda x: severity_order.get(x['severity'], 0))
            
        return {
            'total_alerts': len(recent_alerts),
            'severity_counts': severity_counts,
            'type_counts': type_counts,
            'most_recent_alert': most_recent_alert,
            'most_severe_alert': most_severe_alert,
            'time_period_hours': hours
        }
        
    def get_alert_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Alert statistics dictionary
        """
        # Get alerts from specified time period
        cutoff_time = datetime.now() - timedelta(days=days)
        period_alerts = [alert for alert in self.alert_history if alert['timestamp'] > cutoff_time]
        
        if not period_alerts:
            return {
                'total_alerts': 0,
                'alerts_per_day': 0,
                'severity_distribution': {},
                'type_distribution': {},
                'hourly_distribution': {},
                'daily_distribution': {}
            }
            
        # Calculate alerts per day
        alerts_per_day = len(period_alerts) / days
        
        # Severity distribution
        severity_dist = {}
        for alert in period_alerts:
            severity = alert['severity']
            if severity not in severity_dist:
                severity_dist[severity] = 0
            severity_dist[severity] += 1
            
        # Type distribution
        type_dist = {}
        for alert in period_alerts:
            alert_type = alert['type']
            if alert_type not in type_dist:
                type_dist[alert_type] = 0
            type_dist[alert_type] += 1
            
        # Hourly distribution
        hourly_dist = {i: 0 for i in range(24)}
        for alert in period_alerts:
            hour = alert['timestamp'].hour
            hourly_dist[hour] += 1
            
        # Daily distribution
        daily_dist = {i: 0 for i in range(7)}  # 0=Monday, 6=Sunday
        for alert in period_alerts:
            day = alert['timestamp'].weekday()
            daily_dist[day] += 1
            
        return {
            'total_alerts': len(period_alerts),
            'alerts_per_day': alerts_per_day,
            'severity_distribution': severity_dist,
            'type_distribution': type_dist,
            'hourly_distribution': hourly_dist,
            'daily_distribution': daily_dist
        }
        
    def export_alerts(self, output_path: str, hours: int = 24) -> None:
        """
        Export alerts to file.
        
        Args:
            output_path: Output file path
            hours: Number of hours to export
        """
        # Get alerts from specified time period
        cutoff_time = datetime.now() - timedelta(hours=hours)
        export_alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
        
        # Convert datetime to string for JSON serialization
        export_data = []
        for alert in export_alerts:
            alert_copy = alert.copy()
            alert_copy['timestamp'] = alert_copy['timestamp'].isoformat()
            export_data.append(alert_copy)
            
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported {len(export_data)} alerts to {output_path}")