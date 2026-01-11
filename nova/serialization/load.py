"""
Serialization loading utilities.

Provides safe and unsafe loading mechanisms for NovaNN objects.
By default, loading is restricted to weights-only checkpoints to
prevent unsafe code execution.
"""

import pickle
import io
from pathlib import Path
from typing import Any
from nova.utils.logger import logger
from nova.exceptions import (
    LoadError,
    UnsafeLoadError,
    FileNotFoundError as NovaFileNotFoundError,
)


def load(f: str | Path | io.BufferedIOBase, *, weights_only: bool = True) -> Any:
    """
    Load a serialized object from a file or file-like object.

    By default, loading is performed using a restricted unpickler that
    allows only registered NovaNN classes and safe dependencies. This
    prevents execution of arbitrary code during deserialization.

    Args:
        f: File path (str or Path) or file-like object opened in binary mode
        weights_only: If True, use a restricted unpickler (recommended).
            If False, fall back to standard pickle loading (unsafe).

    Returns:
        The deserialized object

    Raises:
        FileNotFoundError: If file path doesn't exist
        LoadError: If deserialization fails
        UnsafeLoadError: If attempting to load unregistered objects with weights_only=True

    Examples::

        >>> import nova
        >>>
        >>> # Load a saved model
        >>> model = nova.load("model.pth")
        ✅ Successfully loaded from model.pth
        >>>
        >>> # Load from Path object
        >>> from pathlib import Path
        >>> model = nova.load(Path("checkpoints/model.pth"))
        >>>
        >>> # Load from a buffer
        >>> import io
        >>> buffer = io.BytesIO(saved_bytes)
        >>> buffer.seek(0)
        >>> model = nova.load(buffer)
        >>>
        >>> # Unsafe loading (not recommended - security risk!)
        >>> model = nova.load("model.pth", weights_only=False)
        ⚠️  Warning: Loading with weights_only=False is unsafe
        >>>
        >>> # Load state dict and apply to model
        >>> state_dict = nova.load("weights.pth")
        >>> model.load_state_dict(state_dict)
    """

    if not weights_only:
        logger.warning(
            "⚠️  Loading with weights_only=False is unsafe and may execute arbitrary code"
        )

    try:
        if isinstance(f, (str, Path)):
            file_path = Path(f)

            if not file_path.exists():
                raise NovaFileNotFoundError(
                    f"File does not exist: {file_path.absolute()}"
                )

            if not file_path.is_file():
                raise LoadError(f"Path is not a file: {file_path.absolute()}")

            with open(file_path, "rb") as file:
                result = _load_from_file(file, weights_only=weights_only)

            logger.info(f"✅ Successfully loaded from {file_path}")
            return result

        elif hasattr(f, "read"):
            result = _load_from_file(f, weights_only=weights_only)
            logger.info("✅ Successfully loaded from buffer")
            return result

        else:
            raise TypeError(
                f"Expected file path (str/Path) or file-like object, got {type(f).__name__}"
            )

    except NovaFileNotFoundError:
        raise  # Re-raise our custom exception

    except UnsafeLoadError:
        raise  # Re-raise unsafe load errors

    except pickle.UnpicklingError as e:
        if weights_only:
            raise UnsafeLoadError(
                f"Failed to load with weights_only=True. "
                f"The object may contain unregistered classes. "
                f"Original error: {e}"
            ) from e
        else:
            raise LoadError(f"Failed to unpickle object: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error during loading: {type(e).__name__}: {e}")
        raise LoadError(f"Failed to load object: {e}") from e


def _load_from_file(file: io.BufferedIOBase, weights_only: bool = True) -> Any:
    """
    Internal helper to load from an open file handle.

    Args:
        file: Open binary file handle
        weights_only: Whether to use safe unpickler

    Returns:
        Deserialized object

    Raises:
        UnsafeLoadError: If safe loading fails
        LoadError: If general loading fails
    """
    from ._safe_load import SafeUnpickler

    if weights_only:
        try:
            return SafeUnpickler(file).load()
        except pickle.UnpicklingError as e:
            raise UnsafeLoadError(
                f"Object contains unregistered or unsafe classes: {e}"
            ) from e
    else:
        return pickle.load(file)
