import logging

from enum import Enum
from src.core import LOG_FILE, LOGGER_DEFAULT_FORMAT, LOGGER_DATE_FORMAT
from typing import Optional


class LoggerLevel(Enum):
    """Enumeration for different logging levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class Logger:
    """A custom logger class.

    This class provides a simple interface for logging messages to the console
    and/or to a file. It uses the standard `logging` module internally but
    simplifies the configuration and usage.

    It supports different logging levels and custom formatting.
    """

    def __init__(
        self,
        name: str,
        logs_file: Optional[str] = LOG_FILE,
        level: LoggerLevel = LoggerLevel.DEBUG,
        format_string: Optional[str] = LOGGER_DEFAULT_FORMAT,
    ):
        """Initializes the Logger.

        Args:
            name (str): The name of the logger.
            logs_file (Optional[str]): The path to the log file. If None, only
                                         console logging is enabled.
            level (LoggerLevel): The logging level (e.g., DEBUG, INFO, WARNING).
            format_string (Optional[str]): The format string for log messages.
                                             If None, a default format is used.
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)

        if not self._logger.handlers:
            formatter = logging.Formatter(format_string, datefmt=LOGGER_DATE_FORMAT)

            self._create_console_handler(level, formatter)

            if logs_file:
                self._create_file_hadler(logs_file, level, formatter)

    def _create_file_hadler(self, logs_file: str, level, formatter):
        """Creates and configures a file handler for logging.

        Args:
            logs_file (str): The path to the log file.
            level: The logging level.
            formatter: The formatter for log messages.
        """
        file_handler = logging.FileHandler(logs_file)
        file_handler.setLevel(level.value)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _create_console_handler(self, level, formatter):
        """Creates and configures a console handler for logging.

        Args:
            level: The logging level.
            formatter: The formatter for log messages.
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level.value)
        self._logger.addHandler(console_handler)

    def info(self, msg: str, **kwargs):
        """Logs a message with the INFO level.

        Args:
            msg (str): The message to log.
            **kwargs: Extra data to include in the log message.
        """
        self._logger.info(msg, extra=self._extra_data(**kwargs))

    def debug(self, msg: str, **kwargs):
        """Logs a message with the DEBUG level.

        Args:
            msg (str): The message to log.
            **kwargs: Extra data to include in the log message.
        """
        self._logger.debug(msg, extra=self._extra_data(**kwargs))

    def warning(self, msg: str, **kwargs):
        """Logs a message with the WARNING level.

        Args:
            msg (str): The message to log.
            **kwargs: Extra data to include in the log message.
        """
        self._logger.warning(msg, extra=self._extra_data(**kwargs))

    def error(self, msg: str, **kwargs):
        """Logs a message with the ERROR level.

        Args:
            msg (str): The message to log.
            **kwargs: Extra data to include in the log message.
        """
        self._logger.error(msg, extra=self._extra_data(**kwargs))

    def _extra_data(self, **kwargs):
        """Formats extra data to be included in the log message.

        Args:
            **kwargs: Extra data to format.

        Returns:
            dict: A dictionary with the formatted extra data.
        """
        return {f"Extra_{k}": v for k, v in kwargs.items()}


# Create a default logger instance for the project
logger = Logger("Neural Network")
