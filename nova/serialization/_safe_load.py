"""
Safe deserialization utilities.

This module provides a restricted unpickler that prevents arbitrary
code execution by allowing only a controlled set of modules and
explicitly registered classes to be deserialized.

It is used internally by `nova.serialization.load` when loading
weights-only checkpoints.
"""

import pickle
from nova.utils.decorators.registry import _MODULES
from nova.utils import get_registered_classes
from nova.exceptions import UnsafeLoadError


ALLOWED_MODULES: set[str] = {
    "numpy",
    "numpy.core.multiarray",
    "numpy.core.numeric",
    "numpy._core.numeric",
    "numpy._core.multiarray",
    "nova.dtypes",
}


ALLOWED_BUILTINS: set[str] = {
    "dict",
    "list",
    "tuple",
    "set",
    "frozenset",
    "int",
    "float",
    "str",
    "bytes",
    "bool",
    "complex",
    "bytearray",
    "range",
    "slice",
    "type",
    "object",
    "NoneType",
}


class SafeUnpickler(pickle.Unpickler):
    """
    Restricted unpickler for NovaNN objects.

    This unpickler allows loading:
    - Selected NumPy internals
    - Built-in Python types (dict, list, int, etc.)
    - Explicitly registered NovaNN classes

    Any other class resolution attempt is blocked to prevent execution
    of arbitrary or unsafe code during deserialization.

    Examples::

        >>> import io
        >>> from nova.serialization._safe_load import SafeUnpickler
        >>>
        >>> # Safe loading of registered objects
        >>> buffer = io.BytesIO(pickled_data)
        >>> unpickler = SafeUnpickler(buffer)
        >>> obj = unpickler.load()
        >>>
        >>> # Will raise error for unregistered classes
        >>> # unpickler.load()  # UnsafeLoadError if unsafe
    """

    def find_class(self, module_name: str, global_name: str):
        """
        Resolve a class during unpickling with strict safety checks.

        Args:
            module_name: Name of the module containing the object
            global_name: Name of the object to resolve

        Returns:
            The resolved class or object

        Raises:
            UnsafeLoadError: If the object is not in the allowlist

        Note:
            This method is called by the pickle module during deserialization.
            It implements a strict allowlist approach for security.
        """

        # Allow explicitly whitelisted numpy modules
        if module_name in ALLOWED_MODULES:
            return super().find_class(module_name, global_name)

        # Allow specific numpy internals needed for array reconstruction
        if module_name.startswith("numpy") and global_name in (
            "_frombuffer",
            "scalar",
            "dtype",
            "ndarray",
        ):
            return super().find_class(module_name, global_name)

        # Allow safe built-in types
        if module_name == "builtins":
            import builtins

            if global_name in ALLOWED_BUILTINS and hasattr(builtins, global_name):
                return getattr(builtins, global_name)

        # Allow OrderedDict for state_dict loading
        if module_name == "collections" and global_name == "OrderedDict":
            import collections

            return collections.OrderedDict

        # Check if class is explicitly registered in NovaNN
        cls = get_registered_classes(module=module_name, name=global_name)

        if cls is None:
            # Fallback: check registry by name only
            for (_, name), registered_cls in _MODULES.items():
                if name == global_name:
                    return registered_cls

        if cls is not None:
            return cls

        # Block everything else
        raise UnsafeLoadError(
            f"Blocked unpickling of unregistered class: {module_name}.{global_name}. "
            f"To fix this, either:\n"
            f"  1. Register the class using @registry_class decorator\n"
            f"  2. Load with weights_only=False (not recommended - security risk)"
        )
