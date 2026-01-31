"""Custom exceptions for nova module."""


class SerializationError(Exception):
    """Base exception for serialization errors."""

    pass


class SaveError(SerializationError):
    """Raised when saving fails."""

    pass


class LoadError(SerializationError):
    """Raised when loading fails."""

    pass


class UnsafeLoadError(LoadError):
    """Raised when attempting to load unsafe/unregistered objects."""

    pass


class FileNotFoundError(LoadError):
    """Raised when file path doesn't exist."""

    pass


class DatasetCorruptionError(Exception):
    """Raised when dataset files are corrupted or invalid."""

    pass


class DatasetDownloadError(Exception):
    """Raised when dataset download fails."""

    pass


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""

    pass


class PathConfigurationError(Exception):
    """Raised when path configuration fails."""

    pass
