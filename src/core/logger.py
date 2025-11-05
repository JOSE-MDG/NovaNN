import logging

from enum import Enum
from src.core import config
from typing import Optional


class LoggerLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class Logger:
    def __init__(
        self,
        name: str,
        logs_file: Optional[str] = config.LOG_FILE,
        level: LoggerLevel = LoggerLevel.DEBUG,
        format_string: Optional[str] = config.LOGGER_DEFAULT_FORMAT,
    ):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)

        if not self._logger.handlers:
            formatter = logging.Formatter(
                format_string, datefmt=config.LOGGER_DATE_FORMAT
            )

            self._create_console_handler(level, formatter)

            if logs_file:
                self._create_file_hadler(logs_file, level, formatter)

    def _create_file_hadler(self, logs_file: str, level, formatter):
        file_handler = logging.FileHandler(logs_file)
        file_handler.setLevel(level.value)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _create_console_handler(self, level, formatter):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level.value)
        self._logger.addHandler(console_handler)

    def info(self, msg: str, **kwargs):
        self._logger(msg, extra=self._extra_data(**kwargs))

    def debug(self, msg: str, **kwargs):
        self._logger.debug(msg, extra=self._extra_data(**kwargs))

    def warning(self, msg: str, **kwargs):
        self._logger.warning(msg, extra=self._extra_data(**kwargs))

    def error(self, msg: str, **kwargs):
        self._logger.error(msg, extra=self._extra_data(**kwargs))

    def _extra_data(self, **kwargs):
        return {f"Extra_{k}": v for k, v in kwargs.items()}


logger = Logger("NN Project")
