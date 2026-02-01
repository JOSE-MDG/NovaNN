import logging
from enum import Enum
from pathlib import Path
from nova.core import LOGGER_DEFAULT_FORMAT, LOGGER_DATE_FORMAT

_DEFAULT_LOG_DIR = Path.home() / ".novann" / "logs"
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "nova.log"

_logger_instance = None


class LoggerLevel(Enum):
    """Enumeration for different logging levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class Logger:
    """A custom logger class with proper singleton pattern."""

    _logger: logging.Logger

    def __init__(
        self,
        name: str = "NovaNN",
        level: LoggerLevel = LoggerLevel.DEBUG,
        format_string: str = LOGGER_DEFAULT_FORMAT,
    ) -> None:
        """Initializes the Logger (only once due to singleton)."""

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)

        if not self._logger.handlers:
            formatter = logging.Formatter(format_string, datefmt=LOGGER_DATE_FORMAT)
            self._create_console_handler(level, formatter)
            self._create_file_handler(_DEFAULT_LOG_FILE, level, formatter)

    def _create_file_handler(
        self,
        logs_file: Path | str,
        level: LoggerLevel,
        formatter: logging.Formatter,
    ) -> None:
        """Creates and configures a file handler for logging."""
        log_path = Path(logs_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
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

    def flush(self) -> None:
        """Manually flush all handlers to ensure logs are written."""
        for handler in self._logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

    def info(self, msg: str, **kwargs) -> None:
        """Logs a message with the INFO level."""
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}: {v}" for k, v in kwargs.items())
            msg = msg + extra_str
        self._logger.info(msg)
        self.flush()

    def debug(self, msg: str, **kwargs) -> None:
        """Logs a message with the DEBUG level."""
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}: {v}" for k, v in kwargs.items())
            msg = msg + extra_str
        self._logger.debug(msg)
        self.flush()

    def warning(self, msg: str, **kwargs) -> None:
        """Logs a message with the WARNING level."""
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}: {v}" for k, v in kwargs.items())
            msg = msg + extra_str
        self._logger.warning(msg)
        self.flush()

    def error(self, msg: str, **kwargs) -> None:
        """Logs a message with the ERROR level."""
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}: {v}" for k, v in kwargs.items())
            msg = msg + extra_str
        self._logger.error(msg, exc_info=True)
        self.flush()


def get_logger() -> Logger:
    """Returns the singleton Logger instance, creating it on first call.

    The logger always writes to both console and ~/.novann/logs/nova.log.

    Returns:
        Logger: The singleton logger instance

    Examples:
        >>> from nova.utils.logger import get_logger
        >>> logger = get_logger()
        >>> logger.info("Training started")
    """
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = Logger(name="NovaNN")

    return _logger_instance
