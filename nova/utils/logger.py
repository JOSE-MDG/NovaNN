import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional
from logging import Formatter
from nova.core import LOGGER_DEFAULT_FORMAT, LOGGER_DATE_FORMAT

LOG_FILE: Optional[Path] = None


class LoggerLevel(Enum):
    """Enumeration for different logging levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class Logger:
    """A custom logger class with proper singleton pattern."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        name: str = "NovaNN",
        logs_file: Optional[Path] = LOG_FILE,
        level: LoggerLevel = LoggerLevel.DEBUG,
        format_string: Optional[Path] = LOGGER_DEFAULT_FORMAT,
    ) -> None:
        """Initializes the Logger (only once due to singleton)."""

        # Skip if already initialized
        if Logger._initialized:
            return

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)

        # Clear existing handlers to avoid duplicates
        self._logger.handlers.clear()

        formatter = logging.Formatter(format_string, datefmt=LOGGER_DATE_FORMAT)

        self._create_console_handler(level, formatter)

        if logs_file:
            self._create_file_handler(logs_file, level, formatter)

        Logger._initialized = True

    def _create_file_handler(
        self, logs_file: str, level: LoggerLevel, formatter: logging.Formatter
    ) -> None:
        """Creates and configures a file handler for logging."""
        log_path = Path(logs_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_file, encoding="utf-8")
        file_handler.setLevel(level.value)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _create_console_handler(
        self, level: LoggerLevel, formatter: logging.Formatter
    ) -> None:
        """Creates and configures a console handler for logging."""
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level.value)
        self._logger.addHandler(console_handler)

    def set_level(self, level: LoggerLevel) -> None:
        """Dynamically change logging level for all handlers."""
        self._logger.setLevel(level.value)
        for handler in self._logger.handlers:
            handler.setLevel(level.value)

    def info(self, msg: str, **kwargs) -> None:
        """Logs a message with the INFO level."""
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}: {v}" for k, v in kwargs.items())
            msg = msg + extra_str
        self._logger.info(msg)

    def debug(self, msg: str, **kwargs) -> None:
        """Logs a message with the DEBUG level."""
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}: {v}" for k, v in kwargs.items())
            msg = msg + extra_str
        self._logger.debug(msg)

    def warning(self, msg: str, **kwargs) -> None:
        """Logs a message with the WARNING level."""
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}: {v}" for k, v in kwargs.items())
            msg = msg + extra_str
        self._logger.warning(msg)

    def error(self, msg: str, **kwargs) -> None:
        """Logs a message with the ERROR level."""
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}: {v}" for k, v in kwargs.items())
            msg = msg + extra_str
        self._logger.error(msg, exc_info=True)


# Create a single instance (singleton by module-level variable)
logger = Logger("NovaNN")


def enable_file_logging(
    path: Optional[Path | str] = None,
    level: LoggerLevel = LoggerLevel.DEBUG,
    replace_existing: bool = True,
) -> logging.Logger:
    """Enables logging to a file.

    Args:
        path (Optional[Path | str]): Path where the log file will be saved.
            If None, uses ~/.novann/logs/nova.log
        level (LoggerLevel): Logging level to use. Default: DEBUG
        replace_existing (bool): If True, removes existing file handlers
            before adding new one. Default: True

    Returns:
        logging.Logger: The configured logger instance

    Raises:
        PermissionError: If the log directory is not writable
        OSError: If directory creation fails

    Examples:
        >>> enable_file_logging()  # Use default path
        >>> enable_file_logging("logs/training.log", LoggerLevel.INFO)
        >>> enable_file_logging("/var/log/app.log", replace_existing=False)
    """
    global LOG_FILE

    # If no path is provided, use the default safe path
    if path is None:
        default_dir = Path.home() / ".novann" / "logs"
        default_dir.mkdir(parents=True, exist_ok=True)
        path = default_dir / "nova.log"
    else:
        path = Path(path)

    # Validate that parent directory exists or can be created
    log_dir = path.parent
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create log directory {log_dir}: {e}")

    # Check if directory is writable
    if not os.access(log_dir, os.W_OK):
        raise PermissionError(f"Log directory {log_dir} is not writable")

    # Remove existing file handlers if requested
    if replace_existing:
        logger._logger.handlers = [
            h for h in logger._logger.handlers if not isinstance(h, logging.FileHandler)
        ]

    # Create file handler on the singleton instance
    formatter = Formatter(LOGGER_DEFAULT_FORMAT, datefmt=LOGGER_DATE_FORMAT)
    logger._create_file_handler(path, level, formatter)

    # Update the tracked log file path
    LOG_FILE = path

    # Adjust level
    logger.set_level(level)

    return logger


def is_file_logging_enabled() -> bool:
    """Check if file logging is currently enabled.

    Returns:
        bool: True if any FileHandler is attached to the logger

    Examples:
        >>> enable_file_logging()
        >>> is_file_logging_enabled()
        True
        >>> # Only console logging
        >>> is_file_logging_enabled()
        False
    """
    return any(
        isinstance(handler, logging.FileHandler) for handler in logger._logger.handlers
    )


def get_log_file_path() -> Optional[Path]:
    """Get the path of the current log file if file logging is enabled.

    Note: If multiple file handlers were added with replace_existing=False,
    this returns the path of the most recently configured file handler.

    Returns:
        Optional[Path]: Path to the log file, or None if no file handler exists

    Examples:
        >>> enable_file_logging("logs/app.log")
        >>> get_log_file_path()
        PosixPath('logs/app.log')
        >>> # No file logging enabled
        >>> get_log_file_path()
        None
    """
    # Return the tracked path if available
    if LOG_FILE is not None:
        return LOG_FILE

    # Otherwise, try to get it from the actual file handler (fallback)
    for handler in logger._logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)

    return None
