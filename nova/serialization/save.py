"""
Serialization saving utilities.

Provides helpers for serializing NovaNN objects using pickle.
"""

import traceback
import pickle
import io
from nova.utils.logger import logger
from typing import Any


def save(obj: Any, f: str | io.BufferedIOBase, protocol: int = pickle.HIGHEST_PROTOCOL):
    """
    Serialize an object to a file or file-like object using pickle.

    This function is typically used to store model weights, optimizers,
    or training checkpoints.

    Args:
        obj: Object to serialize.
        f: File path or file-like object opened in binary mode.
        protocol: Pickle protocol version to use.

    Example:
        >>> import nova
        >>> import nova.nn as nn
        >>>
        >>> model = nn.Linear(10, 5)
        >>> save(model, "model.pt")
        >>>
        >>> # Save to an in-memory buffer
        >>> import io
        >>> buffer = io.BytesIO()
        >>> nova.save(model, buffer)
    """

    try:
        if isinstance(f, str):
            with open(f, "wb") as file:
                pickle.dump(obj, file, protocol=protocol)
        else:
            pickle.dump(obj, f, protocol=protocol)

        logger.info("✅ Saved successfully")
    except Exception as e:
        lines = [line for line in traceback.format_exception(e)]
        logger.error("An error occurred during saving.")
        print(*lines)
