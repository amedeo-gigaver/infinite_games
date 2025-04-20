"""
Logging configuration for the FastAPI application.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging and redirects to loguru.
    
    Allows seamless integration with libraries using standard logging.
    """
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Intercept log record and redirect to loguru.
        
        Args:
            record: The log record
        """
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
            
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
            
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: Optional[str] = None,
    retention: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        rotation: When to rotate logs (e.g., "500 MB", "1 day")
        retention: How long to keep logs (e.g., "10 days")
    """
    # Remove default handlers
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    
    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Explicitly set levels for some chatty loggers
    for logger_name in ("uvicorn", "uvicorn.error", "fastapi"):
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
        
    # Set sqlalchemy to WARNING to reduce noise
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized (level={level})")


def get_logger(name: str) -> logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name, typically the module name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name) 