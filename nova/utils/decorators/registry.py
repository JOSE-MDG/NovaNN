from __future__ import annotations
from typing import Callable, Type, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from nova.autograd.function import Function
    from nova._typing import ModuleTypes

_MODULES: dict[tuple[str, str], ModuleTypes] = {}
_OPS_REGISTERED: dict[str, Type[Function]] = {}

T = TypeVar("T", bound=type)


def registry_class(cls: T):
    """
    Register a class for safe serialization and deserialization.

    This decorator registers the given class using its fully qualified
    name (module + class name). Registered classes can later be resolved
    during safe unpickling.

    The registration is idempotent: re-registering the same class has
    no effect.

    Args:
        cls: Class to register.

    Returns:
        The original class, unmodified.
    """
    key = (cls.__module__, cls.__name__)
    if key not in _MODULES:
        _MODULES[key] = cls
    return cls


def registry_op(op_name: str) -> Callable[[Type[Function]], Type[Function]]:
    """
    Register an autograd Function under a public operation name.

    This decorator associates a string operation name with a subclass
    of `Function`. It is primarily used to map serialized operations
    back to their corresponding Function classes during deserialization.

    Args:
        op_name: Public name of the operation (e.g., "add", "relu").

    Returns:
        A decorator that registers a Function subclass.

    Raises:
        ValueError: If the decorated object is not a Function subclass.
    """
    from nova.autograd.function import Function

    def register(cls: Type[Function]):
        if not (isinstance(cls, type) and issubclass(cls, Function)):
            raise ValueError(
                f"Only Function classes can be registered, but got "
                f"'{cls.__name__ if hasattr(cls, '__name__') else cls}'"
            )

        if op_name not in _OPS_REGISTERED:
            _OPS_REGISTERED.setdefault(op_name, cls)
        return cls

    return register


def get_registered_classes(module, name) -> ModuleTypes:
    """
    Retrieve a previously registered class by module and name.

    This function is used during safe deserialization to resolve
    classes that were explicitly registered via `registry_class`.

    Args:
        module: Module path of the class.
        name: Class name.

    Returns:
        The registered class.

    Raises:
        KeyError: If the class is not found in the registry.
    """
    if (module, name) in _MODULES:
        return _MODULES[(module, name)]
    else:
        raise KeyError(f"key '{(module, name)}' not found")
