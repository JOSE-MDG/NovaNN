"""
Serialization loading utilities.

Provides safe and unsafe loading mechanisms for NovaNN objects.
By default, loading is restricted to weights-only checkpoints to
prevent unsafe code execution.
"""

import traceback
import pickle
import io
from nova.utils.logger import logger


def load(f: str | io.BufferedIOBase, *, weights_only: bool = True):
    """
    Load a serialized object from a file or file-like object.

    By default, loading is performed using a restricted unpickler that
    allows only registered NovaNN classes and safe dependencies. This
    prevents execution of arbitrary code during deserialization.

    Args:
        f: File path or file-like object opened in binary mode.
        weights_only: If True, use a restricted unpickler (recommended).
            If False, fall back to standard pickle loading (unsafe).

    Returns:
        The deserialized object.

    Example:
        >>> import nova
        >>>
        >>> model = nova.load("model.pt")
        >>>
        >>> # Load from a buffer
        >>> import io
        >>> buffer = io.BytesIO()
        >>> buffer.seek(0)
        >>> model = nova.load(buffer)
        >>>
        >>> # Unsafe loading (not recommended)
        >>> model = nova.load("model.pt", weights_only=False)
    """

    try:
        res = None
        if isinstance(f, str):
            with open(f, "rb") as file:
                res = _load_from_file(file, weights_only=weights_only)
        else:
            res = _load_from_file(f, weights_only=weights_only)

        logger.info("✅ Successfully loaded")
        return res
    except Exception as e:
        lines = [line for line in traceback.format_exception(e)]
        logger.error("An error occurred during loading.")
        print(*lines)


def _load_from_file(file: str, weights_only: bool = True):
    from ._safe_load import SafeUnpickler

    if weights_only:
        return SafeUnpickler(file).load()
    else:
        return pickle.load(file)
