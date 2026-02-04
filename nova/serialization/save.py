"""
Serialization saving utilities.

Provides helpers for serializing NovaNN objects using pickle.
"""

from __future__ import annotations
import pickle
import io
from pathlib import Path
from typing import Any
from nova.utils.logger import get_logger
from nova.exceptions import SaveError

logger = get_logger()


def save(
    obj: Any, f: str | Path | io.BufferedIOBase, protocol: int = pickle.HIGHEST_PROTOCOL
) -> None:
    """
    Serialize an object to a file or file-like object using pickle.

    This function is typically used to store model weights, optimizers,
    or training checkpoints.

    Args:
        obj: Object to serialize. Should be a NovaNN object (Module, Tensor, etc.)
        f: File path (str or Path) or file-like object opened in binary mode
        protocol: Pickle protocol version to use. Default: highest available protocol

    Raises:
        SaveError: If serialization fails for any reason
        TypeError: If f is not a valid file path or file-like object
        PermissionError: If write permission is denied

    Examples::

        >>> import nova
        >>> import nova.nn as nn
        >>>
        >>> # Save a model
        >>> model = nn.Linear(10, 5)
        >>> nova.save(model, "model.pth")
        ✅ Saved successfully to model.pth
        >>>
        >>> # Save model state dict
        >>> nova.save(model.state_dict(), "weights.pth")
        >>>
        >>> # Save to Path object
        >>> from pathlib import Path
        >>> path = Path("checkpoints") / "model.pth"
        >>> nova.save(model, path)
        >>>
        >>> # Save to an in-memory buffer
        >>> import io
        >>> buffer = io.BytesIO()
        >>> nova.save(model, buffer)
        >>> # Can be sent over network, stored in database, etc.
        >>>
        >>> # Save with specific protocol
        >>> nova.save(model, "model.pth", protocol=4)
    """

    if obj is None:
        raise SaveError("Cannot save None object")

    try:
        if isinstance(f, (str, Path)):
            file_path = Path(f)

            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "wb") as file:
                pickle.dump(obj, file, protocol=protocol)

            logger.info(f"✅ Saved successfully to {file_path}")

        elif hasattr(f, "write"):
            pickle.dump(obj, f, protocol=protocol)
            logger.info("✅ Saved successfully to buffer")

        else:
            raise TypeError(
                f"Expected file path (str/Path) or file-like object, got {type(f).__name__}"
            )

    except PermissionError as e:
        raise SaveError(f"Permission denied when writing to {f}") from e

    except pickle.PicklingError as e:
        raise SaveError(f"Failed to pickle object: {e}") from e

    except OSError as e:
        raise SaveError(f"OS error occurred while saving: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error during saving: {type(e).__name__}: {e}")
        raise SaveError(f"Failed to save object: {e}") from e
