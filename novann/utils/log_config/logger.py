import logging
from enum import Enum
from typing import Optional
from novann.core import LOG_FILE, LOGGER_DEFAULT_FORMAT, LOGGER_DATE_FORMAT


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
        logs_file: Optional[str] = LOG_FILE,
        level: LoggerLevel = LoggerLevel.DEBUG,
        format_string: Optional[str] = LOGGER_DEFAULT_FORMAT,
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
