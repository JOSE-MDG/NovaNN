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
