"""
Logging Configuration for IDE Backend
"""

import logging
import sys

_logger: logging.Logger | None = None


def configure_logging(level: str = "INFO"):
    """Configure logging for the IDE backend."""
    global _logger

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set specific levels for noisy loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    _logger = logging.getLogger("ide_backend")
    _logger.info("Logging configured")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"ide_backend.{name}")
