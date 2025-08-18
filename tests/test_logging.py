"""
Tests for logging utilities.
"""
import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.logging import get_logger, setup_logging


class TestGetLogger:
    """Test get_logger function."""
    
    def test_get_logger_basic(self):
        """Test basic logger creation."""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_get_logger_different_names(self):
        """Test getting loggers with different names."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
        assert logger1 is not logger2
    
    def test_get_logger_same_name_returns_same_instance(self):
        """Test that getting logger with same name returns same instance."""
        logger1 = get_logger("same_logger")
        logger2 = get_logger("same_logger")
        
        assert logger1 is logger2
    
    def test_logger_has_handlers(self):
        """Test that logger has appropriate handlers."""
        logger = get_logger("handler_test")
        
        # Logger should have at least one handler (console)
        assert len(logger.handlers) >= 1
    
    def test_logger_levels(self):
        """Test that logger can handle different log levels."""
        logger = get_logger("level_test")
        
        # Test that these don't raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    
    def test_logger_with_context(self):
        """Test logger with context information."""
        logger = get_logger("context_test")
        
        # Test logging with extra context
        logger.info("Trade executed", extra={
            'symbol': 'MES',
            'quantity': 1,
            'price': 4500.0
        })


class TestSetupLogging:
    """Test setup_logging function."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging()
        
        # Check that root logger is configured
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO
    
    def test_setup_logging_with_level(self):
        """Test logging setup with specific level."""
        setup_logging(level=logging.DEBUG)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            setup_logging(log_file=log_file)
            
            # Test that file handler is added
            logger = get_logger("file_test")
            logger.info("Test message")
            
            # Check that log file exists and has content
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
                
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_setup_logging_with_format(self):
        """Test logging setup with custom format."""
        custom_format = "%(levelname)s - %(message)s"
        setup_logging(format_string=custom_format)
        
        # This mainly tests that the function doesn't raise an exception
        logger = get_logger("format_test")
        logger.info("Formatted message")
    
    def test_setup_logging_directory_creation(self):
        """Test that log directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = os.path.join(tmp_dir, "subdir", "test.log")
            
            setup_logging(log_file=log_file)
            
            # Test that the directory was created
            assert os.path.exists(os.path.dirname(log_file))
            
            # Test that logging works
            logger = get_logger("dir_test")
            logger.info("Directory creation test")
            
            assert os.path.exists(log_file)
    
    def test_setup_logging_json_format(self):
        """Test logging setup with JSON formatting."""
        setup_logging(json_format=True)
        
        logger = get_logger("json_test")
        logger.info("JSON test message", extra={
            'trade_id': 123,
            'symbol': 'MES'
        })
    
    @patch('src.utils.logging.RotatingFileHandler')
    def test_setup_logging_rotation(self, mock_handler):
        """Test logging setup with file rotation."""
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance

        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            setup_logging(
                log_file=log_file,
                max_bytes=1024*1024,  # 1MB
                backup_count=5
            )

            # Verify that RotatingFileHandler was called with correct parameters
            mock_handler.assert_called_with(
                str(log_file),
                maxBytes=1024*1024,
                backupCount=5,
                encoding='utf-8'
            )

            # Verify that the handler was added to the root logger
            assert mock_handler_instance in logging.getLogger().handlers

        finally:
            # Clean up
            if os.path.exists(log_file):
                os.unlink(log_file)


class TestLoggingIntegration:
    """Test logging integration with the system."""
    
    def test_structured_logging(self):
        """Test structured logging with trade information."""
        logger = get_logger("structured_test")
        
        # Test trade logging
        logger.info("Trade executed", extra={
            'event_type': 'trade_execution',
            'symbol': 'MES',
            'side': 'BUY',
            'quantity': 1,
            'price': 4500.0,
            'timestamp': '2023-01-01T10:00:00Z'
        })
        
        # Test error logging
        logger.error("Trade failed", extra={
            'event_type': 'trade_error',
            'symbol': 'MES',
            'error_code': 'INSUFFICIENT_MARGIN',
            'error_message': 'Insufficient buying power'
        })
    
    def test_performance_logging(self):
        """Test performance-related logging."""
        logger = get_logger("performance_test")
        
        # Test performance metrics logging
        logger.info("Performance metrics", extra={
            'event_type': 'performance',
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05,
            'num_trades': 150
        })
    
    def test_risk_logging(self):
        """Test risk-related logging."""
        logger = get_logger("risk_test")
        
        # Test risk event logging
        logger.warning("Risk limit breached", extra={
            'event_type': 'risk_breach',
            'limit_type': 'position_size',
            'current_value': 1000000,
            'limit_value': 800000,
            'action': 'position_reduced'
        })
    
    def test_system_logging(self):
        """Test system-level logging."""
        logger = get_logger("system_test")
        
        # Test system startup
        logger.info("System startup", extra={
            'event_type': 'system_startup',
            'version': '1.0.0',
            'config_file': 'configs/settings.yaml'
        })
        
        # Test connection events
        logger.info("Connection established", extra={
            'event_type': 'connection',
            'service': 'IBKR',
            'status': 'connected',
            'latency_ms': 50
        })