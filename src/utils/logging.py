"""
Logging utilities for the RL trading system.

This module provides structured logging with proper formatting, file rotation,
and consistent log levels across the application.
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Union, Optional, Dict, Any
from logging.handlers import RotatingFileHandler


# Global logger cache
_loggers: Dict[str, logging.Logger] = {}


def get_logger(name: str, level: Union[int, str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger
        level: Logging level (can be int or string)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level if provided
    if level is not None:
        if isinstance(level, str):
            # Convert string level to integer
            numeric_level = getattr(logging, level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f'Invalid log level: {level}')
            logger.setLevel(numeric_level)
        else:
            # Ensure level is an integer
            logger.setLevel(int(level))
    
    # Add a default handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "standard",
    format_string: Optional[str] = None,
    json_format: bool = False,
    max_bytes: int = 10485760,
    backup_count: int = 5
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or integer
        log_file: Optional log file path
        log_format: Log format ('standard' or 'json')
        format_string: Custom format string
        json_format: Whether to use JSON formatting
        max_bytes: Maximum log file size in bytes
        backup_count: Number of backup files to keep
    """
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Handle both string and integer level parameters
    if isinstance(level, int):
        root_logger.setLevel(level)
    else:
        root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    if format_string:
        formatter = logging.Formatter(format_string)
    elif json_format:
        formatter = JSONFormatter()
    elif log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if max_bytes is not None and backup_count is not None:
            # Use string path for RotatingFileHandler
            handler = RotatingFileHandler(
                str(log_file),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            # Use string path for FileHandler
            handler = logging.FileHandler(str(log_file), encoding='utf-8')
        
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    # Update cached loggers
    for logger in _loggers.values():
        logger.setLevel(root_logger.level)
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add handlers from root logger
        for handler in root_logger.handlers:
            logger.addHandler(handler)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra') and record.extra:
            log_entry.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)
