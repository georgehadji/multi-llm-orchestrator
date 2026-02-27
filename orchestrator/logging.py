"""
Structured Logging
==================
Structured logging with correlation IDs, context management, and JSON output.

Features:
- Correlation IDs for request tracing
- Structured JSON logging for production
- Human-readable text logging for development
- Context variables for automatic field injection
- Log level configuration via environment

Usage:
    from orchestrator.logging import get_logger, set_correlation_id
    
    # In your entry point
    set_correlation_id("req-12345")
    
    # In your module
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"task_id": "task_001"})
    
    Output:
    {"timestamp": "2024-01-15T10:30:00", "level": "INFO", 
     "logger": "orchestrator.engine", "message": "Processing started",
     "correlation_id": "req-12345", "task_id": "task_001"}
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

# Context variable for correlation ID (async-safe)
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.
    
    Outputs log records as JSON objects for easy parsing by log aggregation
    systems (ELK, Splunk, CloudWatch, etc.).
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "")
            or _correlation_id.get(),
        }
        
        # Add source location in debug mode
        if record.levelno <= logging.DEBUG:
            log_obj["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }
        
        # Add exception info if present
        if record.exc_info and record.exc_info != (None, None, None):
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "asctime", "correlation_id",
            ):
                log_obj[key] = value
        
        return json.dumps(log_obj, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development.
    
    Format: 2024-01-15 10:30:00 | INFO     | correlation_id | logger_name | message
    """
    
    def __init__(self, include_correlation: bool = True) -> None:
        super().__init__()
        self.include_correlation = include_correlation
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text."""
        timestamp = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        
        correlation = getattr(record, "correlation_id", "") or _correlation_id.get()
        correlation_str = f" | {correlation:.8}" if self.include_correlation and correlation else ""
        
        msg = f"{timestamp} | {record.levelname:8} | {record.name:30}{correlation_str} | {record.getMessage()}"
        
        if record.exc_info and record.exc_info != (None, None, None):
            msg += "\n" + self.formatException(record.exc_info)
        
        return msg


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to all log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID from context variable."""
        record.correlation_id = _correlation_id.get()
        return True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the orchestrator prefix.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"orchestrator.{name}")


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context.
    
    The correlation ID is stored in a context variable, making it
    safe for use with async code and concurrent requests.
    
    Args:
        correlation_id: Unique identifier for the current request/operation
    """
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str:
    """Get the current correlation ID.
    
    Returns:
        The current correlation ID or empty string if not set
    """
    return _correlation_id.get()


def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    _correlation_id.set("")


def configure_logging(
    level: str = "INFO",
    format: str = "text",
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ('json' for production, 'text' for development)
        log_file: Optional file path to write logs to
    """
    # Get the root orchestrator logger
    root_logger = logging.getLogger("orchestrator")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create handlers list
    handlers: list[logging.Handler] = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if format.lower() == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(TextFormatter())
    console_handler.addFilter(CorrelationIdFilter())
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())  # Always JSON in files
        file_handler.addFilter(CorrelationIdFilter())
        handlers.append(file_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Configure third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_log_level_from_env(default: str = "INFO") -> str:
    """Get log level from environment variable.
    
    Checks LOG_LEVEL environment variable first, then falls back to default.
    
    Args:
        default: Default log level if not set in environment
        
    Returns:
        Log level string
    """
    import os
    return os.environ.get("LOG_LEVEL", default)


class LogContext:
    """Context manager for temporary correlation ID setting.
    
    Usage:
        with LogContext(request_id="req-123"):
            logger.info("Processing request")
            # correlation_id is automatically set
        # correlation_id is restored to previous value
    """
    
    def __init__(
        self,
        correlation_id: Optional[str] = None,
        **fields: Any,
    ):
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self.fields = fields
        self.token: Optional[Any] = None
    
    def __enter__(self) -> "LogContext":
        """Set correlation ID on enter."""
        self.token = _correlation_id.set(self.correlation_id)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restore previous correlation ID on exit."""
        if self.token:
            _correlation_id.reset(self.token)


# Import uuid here to avoid circular imports
import uuid
