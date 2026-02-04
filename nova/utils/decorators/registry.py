from __future__ import annotations
from typing import Callable, Type, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from nova.autograd.function import Function
    from nova._typing import ModuleTypes

_MODULES: dict[tuple[str, str], ModuleTypes] = {}
_OPS_REGISTERED: dict[str, Type[Function]] = {}
_NO_INPLACE_OPS: set[str] = set()

T = TypeVar("T", bound=type)


def no_inplace_op(func: type[Function]) -> type[Function]:
    """
    Mark a Function as not supporting in-place operations.

    This decorator registers the given autograd Function class in a set
    of operations that cannot be performed in-place. Operations marked
    with this decorator will raise an error if an in-place variant is
    attempted (e.g., `add_()` on an operation that doesn't support it).

    The decorator is used to prevent incorrect in-place modifications
    on operations where in-place semantics would be invalid or unsafe,
    such as operations that require specific memory layouts or operations
    that would break the autograd graph.

    Args:
        func: Function class to mark as non-in-place compatible.

    Returns:
        The original Function class, unmodified.

    Example:
        >>> from nova.autograd.function import Function
        >>> from nova.utils.decorators.registry import no_inplace_op
        >>>
        >>> @no_inplace_op
        ... class ComplexOperation(Function):
        ...     @staticmethod
        ...     def forward(ctx, x):
        ...         # Operation that cannot be done in-place
        ...         return x.copy()
        ...
        >>> # Attempting ComplexOperation_() will raise an error

    Note:
        This is typically used internally for operations like BatchNorm,
        LayerNorm, and certain view operations where in-place modifications
        would corrupt intermediate states needed for backpropagation.
    """
    name = func.__name__
    _NO_INPLACE_OPS.add(name)
    return func


def registry_class(cls: T) -> T:
    """
    Register a class for safe serialization and deserialization.

    This decorator registers the given class using its fully qualified
    name (module + class name). Registered classes can later be safely
    resolved during unpickling operations without executing arbitrary code.

    The registration is idempotent: re-registering the same class has
    no effect. This allows the same class to be decorated multiple times
    without causing conflicts.

    Args:
        cls: Class to register. Can be any type, but is typically used
            for Module, Parameter, Buffer, Optimizer, or Tensor subclasses.

    Returns:
        The original class, unmodified.

    Example:
        >>> from nova.utils.decorators.registry import registry_class
        >>>
        >>> @registry_class
        ... class CustomLayer(nn.Module):
        ...     def __init__(self, in_features, out_features):
        ...         super().__init__()
        ...         self.weight = nn.Parameter(torch.randn(out_features, in_features))
        ...
        >>> # CustomLayer can now be safely serialized/deserialized

    Note:
        Classes registered with this decorator can be loaded with
        `nova.load(..., weights_only=True)` in safe mode, preventing
        arbitrary code execution during deserialization.

    See Also:
        - `get_registered_classes()`: Retrieve registered classes by name
        - `nova.save()` / `nova.load()`: Serialization functions
    """
    key = (cls.__module__, cls.__name__)
    if key not in _MODULES:
        _MODULES[key] = cls
    return cls


def registry_op(op_name: str) -> Callable[[Type[Function]], Type[Function]]:
    """
    Register an autograd Function under a public operation name.

    This decorator associates a string operation name with a subclass
    of `Function`. The mapping is used during deserialization to safely
    reconstruct computation graphs without executing arbitrary code.

    Each operation name can only be registered once. Attempting to
    register the same operation name multiple times will be silently
    ignored (the first registration wins).

    Args:
        op_name: Public name of the operation (e.g., "add", "relu", "conv2d").
            This should match the operation name used in the tensor API.

    Returns:
        A decorator that registers the Function subclass and returns it
        unmodified.

    Raises:
        ValueError: If the decorated object is not a subclass of `Function`.

    Example:
        >>> from nova.autograd.function import Function
        >>> from nova.utils.decorators.registry import registry_op
        >>>
        >>> @registry_op("add")
        ... class Add(Function):
        ...     @staticmethod
        ...     def forward(ctx, x, y):
        ...         return x + y
        ...
        ...     @staticmethod
        ...     def backward(ctx, grad_output):
        ...         return grad_output, grad_output
        ...
        >>> # The operation "add" is now safely registered for serialization

    Note:
        - This is used internally by the autograd system to register all
          built-in operations.
        - Custom operations should also use this decorator to ensure
          they can be safely serialized and deserialized.
        - The operation name should match the name used in `native_functions.yaml`.

    See Also:
        - `registry_class()`: Register module/parameter classes
        - `no_inplace_op()`: Mark operations as non-in-place compatible
    """

    from nova.autograd.function import Function

    def register(cls: Type[Function]) -> Type[Function]:
        if not (isinstance(cls, type) and issubclass(cls, Function)):
            raise ValueError(
                f"Only Function classes can be registered, but got "
                f"'{cls.__name__ if hasattr(cls, '__name__') else cls}'"
            )

        if op_name not in _OPS_REGISTERED:
            _OPS_REGISTERED.setdefault(op_name, cls)
        return cls

    return register


def get_registered_classes(module: str, name: str) -> ModuleTypes | None:
    """
    Retrieve a previously registered class by module and name.

    This function is used during safe deserialization to resolve
    classes that were explicitly registered via `registry_class`.
    It provides a secure way to reconstruct objects without using
    `__import__` or other dynamic code execution mechanisms.

    Args:
        module: Fully qualified module path of the class (e.g., 'nova.nn.modules.linear').
        name: Class name within the module (e.g., 'Linear').

    Returns:
        The registered class if found, None otherwise.

    Example:
        >>> from nova.utils.decorators.registry import get_registered_classes
        >>>
        >>> # Assuming Linear was registered with @registry_class
        >>> cls = get_registered_classes('nova.nn.modules.linear', 'Linear')
        >>> if cls is not None:
        ...     layer = cls(10, 5)
        ...
        >>> # If not registered, returns None
        >>> unknown = get_registered_classes('unknown.module', 'UnknownClass')
        >>> print(unknown)  # None

    Note:
        - This function returns None for unregistered classes rather than
          raising an exception, allowing callers to handle missing classes
          gracefully.
        - Only classes decorated with `@registry_class` can be retrieved.
        - This is the primary mechanism for safe deserialization in NovaNN.

    See Also:
        - `registry_class()`: Decorator to register classes
        - `nova.load()`: Uses this function for safe deserialization
    """
    return _MODULES.get((module, name), None)
