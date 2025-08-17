"""
Logging utilities for the RL trading system.

This module provides structured logging with proper formatting, file rotation,
and consistent log levels across the application.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Union
import json
from datetime import datetime

from .config_loader import Settings


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured logs in JSON format.
    
    This formatter adds structured information like timestamp, logger name,
    level, and module to each log message.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


def setup_logging(
    settings: Optional[Settings] = None,
    config_path: Optional[Union[str, Path]] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        settings: Settings instance with logging configuration
        config_path: Path to configuration file
        level: Logging level override
        
    Returns:
        Configured logger instance
    """
    # Load settings if not provided
    if settings is None and config_path:
        settings = Settings.from_paths(config_path)
    
    # Get logging configuration
    if settings:
        log_level = settings.get('logging', 'level', 'INFO')
        log_file = settings.get('logging', 'file')
        max_bytes = settings.get('logging', 'max_bytes', 10485760)  # 10MB
        backup_count = settings.get('logging', 'backup_count', 5)
    else:
        log_level = level
        log_file = None
        max_bytes = 10485760
        backup_count = 5
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Use simple formatter for console
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with structured logging
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Use structured formatter for file
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set up specific loggers
    _setup_loggers()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")
    
    return root_logger


def _setup_loggers() -> None:
    """Set up specific loggers for different components."""
    
    # Trading system logger
    trading_logger = logging.getLogger('rl_trading')
    trading_logger.setLevel(logging.INFO)
    
    # RL training logger
    rl_logger = logging.getLogger('rl_training')
    rl_logger.setLevel(logging.INFO)
    
    # Data logger
    data_logger = logging.getLogger('data_pipeline')
    data_logger.setLevel(logging.INFO)
    
    # Risk logger
    risk_logger = logging.getLogger('risk_management')
    risk_logger.setLevel(logging.INFO)
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TradingLogger:
    """
    Specialized logger for trading operations with structured logging.
    """
    
    def __init__(self, name: str = "trading"):
        self.logger = get_logger(f"rl_trading.{name}")
    
    def log_trade(self, action: str, symbol: str, contracts: int, price: float, 
                  timestamp: datetime, **kwargs) -> None:
        """Log a trade execution."""
        log_data = {
            'event': 'trade',
            'action': action,
            'symbol': symbol,
            'contracts': contracts,
            'price': price,
            'timestamp': timestamp.isoformat(),
            **kwargs
        }
        self.logger.info(json.dumps(log_data, default=str))
    
    def log_risk_event(self, event_type: str, message: str, **kwargs) -> None:
        """Log a risk management event."""
        log_data = {
            'event': 'risk',
            'event_type': event_type,
            'message': message,
            **kwargs
        }
        self.logger.warning(json.dumps(log_data, default=str))
    
    def log_performance(self, metric: str, value: float, **kwargs) -> None:
        """Log performance metrics."""
        log_data = {
            'event': 'performance',
            'metric': metric,
            'value': value,
            **kwargs
        }
        self.logger.info(json.dumps(log_data, default=str))
    
    def log_error(self, error: Exception, context: str = "", **kwargs) -> None:
        """Log an error with context."""
        log_data = {
            'event': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            **kwargs
        }
        self.logger.error(json.dumps(log_data, default=str))


def configure_loggers_for_training() -> None:
    """Configure loggers specifically for training runs."""
    training_logger = get_logger('rl_training')
    
    # Add file handler specifically for training logs
    training_log_path = Path("logs/training.log")
    training_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    training_handler = logging.FileHandler(training_log_path, encoding='utf-8')
    training_handler.setLevel(logging.INFO)
    training_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    training_handler.setFormatter(training_formatter)
    training_logger.addHandler(training_handler)