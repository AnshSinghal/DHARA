"""
Production-ready logging configuration for Agentic Legal RAG system.
Provides structured logging with rotation, filtering, and monitoring capabilities.
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": os.getpid(),
            "thread_id": record.thread,
            "thread_name": record.threadName
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if enabled
        if self.include_extra_fields:
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message', 'exc_info', 'exc_text',
                    'stack_info', 'getMessage'
                }:
                    try:
                        json.dumps(value)  # Check if value is JSON serializable
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)
        
        return json.dumps(log_entry, ensure_ascii=False)


class RequestContextFilter(logging.Filter):
    """Filter to add request context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add request ID and user context if available
        # Note: These would be set by middleware in a real application
        if not hasattr(record, 'request_id'):
            record.request_id = getattr(record, '_request_id', None)
        if not hasattr(record, 'user_id'):
            record.user_id = getattr(record, '_user_id', None)
        return True


class SecuritySanitizer(logging.Filter):
    """Filter to sanitize sensitive information from logs."""
    
    SENSITIVE_PATTERNS = [
        'password', 'secret', 'token', 'key', 'auth', 'credential',
        'api_key', 'private_key', 'access_token', 'refresh_token'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, 'args') and record.args:
            # Sanitize args
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, (dict, str)):
                    sanitized_args.append(self._sanitize_data(arg))
                else:
                    sanitized_args.append(arg)
            record.args = tuple(sanitized_args)
        
        # Sanitize the message
        record.msg = self._sanitize_data(record.msg)
        return True
    
    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize sensitive data."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(pattern in key.lower() for pattern in self.SENSITIVE_PATTERNS):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, str):
            # Basic string sanitization
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern in data.lower():
                    return "***CONTAINS_SENSITIVE_DATA***"
            return data
        return data


class LoggingConfig:
    """Central logging configuration for the application."""
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10,
        enable_json_logging: bool = True,
        enable_console_logging: bool = True,
        enable_security_sanitization: bool = True
    ):
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_json_logging = enable_json_logging
        self.enable_console_logging = enable_console_logging
        self.enable_security_sanitization = enable_security_sanitization
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self) -> None:
        """Configure logging for the entire application."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(self.log_level)
        
        # Configure formatters
        json_formatter = JSONFormatter()
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file handlers
        self._setup_file_handlers(json_formatter)
        
        # Setup console handler
        if self.enable_console_logging:
            self._setup_console_handler(console_formatter)
        
        # Setup filters
        self._setup_filters()
        
        # Configure third-party loggers
        self._configure_third_party_loggers()
        
        # Log startup message
        logger = logging.getLogger(__name__)
        logger.info(
            "Logging system initialized",
            extra={
                "log_level": logging.getLevelName(self.log_level),
                "log_directory": str(self.log_dir),
                "json_logging_enabled": self.enable_json_logging,
                "security_sanitization_enabled": self.enable_security_sanitization
            }
        )
    
    def _setup_file_handlers(self, formatter: logging.Formatter) -> None:
        """Setup rotating file handlers for different log levels."""
        root_logger = logging.getLogger()
        
        # Application logs (all levels)
        app_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "application.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        app_handler.setLevel(logging.DEBUG)
        app_handler.setFormatter(formatter)
        root_logger.addHandler(app_handler)
        
        # Error logs (ERROR and CRITICAL only)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "errors.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Access logs for API requests
        access_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "access.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        access_handler.setLevel(logging.INFO)
        access_handler.setFormatter(formatter)
        
        # Create access logger
        access_logger = logging.getLogger("api.access")
        access_logger.addHandler(access_handler)
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False
        
        # Performance logs
        performance_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "performance.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(formatter)
        
        # Create performance logger
        performance_logger = logging.getLogger("performance")
        performance_logger.addHandler(performance_handler)
        performance_logger.setLevel(logging.INFO)
        performance_logger.propagate = False
    
    def _setup_console_handler(self, formatter: logging.Formatter) -> None:
        """Setup console handler for development and monitoring."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
    
    def _setup_filters(self) -> None:
        """Setup logging filters."""
        root_logger = logging.getLogger()
        
        # Add request context filter
        context_filter = RequestContextFilter()
        for handler in root_logger.handlers:
            handler.addFilter(context_filter)
        
        # Add security sanitization filter
        if self.enable_security_sanitization:
            security_filter = SecuritySanitizer()
            for handler in root_logger.handlers:
                handler.addFilter(security_filter)
    
    def _configure_third_party_loggers(self) -> None:
        """Configure logging levels for third-party libraries."""
        # Reduce noise from third-party libraries
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        
        # Keep FastAPI logs at INFO level
        logging.getLogger("fastapi").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def log_function_call(func_name: str, args: Dict[str, Any], execution_time: Optional[float] = None) -> None:
    """Log function call details for performance monitoring."""
    performance_logger = logging.getLogger("performance")
    log_data = {
        "function": func_name,
        "arguments_count": len(args),
        "execution_time_seconds": execution_time
    }
    
    if execution_time:
        if execution_time > 5.0:
            performance_logger.warning("Slow function execution detected", extra=log_data)
        else:
            performance_logger.info("Function executed", extra=log_data)
    else:
        performance_logger.info("Function called", extra=log_data)


def log_api_request(method: str, path: str, status_code: int, execution_time: float, user_id: Optional[str] = None) -> None:
    """Log API request details."""
    access_logger = logging.getLogger("api.access")
    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "execution_time_seconds": execution_time,
        "user_id": user_id
    }
    
    if status_code >= 500:
        access_logger.error("API request failed with server error", extra=log_data)
    elif status_code >= 400:
        access_logger.warning("API request failed with client error", extra=log_data)
    else:
        access_logger.info("API request completed successfully", extra=log_data)


# Default configuration instance
default_config = LoggingConfig(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs"),
    enable_json_logging=os.getenv("JSON_LOGGING", "true").lower() == "true",
    enable_console_logging=os.getenv("CONSOLE_LOGGING", "true").lower() == "true",
    enable_security_sanitization=os.getenv("SECURITY_SANITIZATION", "true").lower() == "true"
)


def setup_logging() -> None:
    """Setup logging with default configuration."""
    default_config.setup_logging()