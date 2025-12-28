from __future__ import annotations
from typing import Callable, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.function import Function
    from nova._typing import Modules

_MODULES: dict[tuple[str, str], Modules] = {}
_OPS_REGISTERED: dict[str, Type[Function]] = {}


def registry_class(cls: Type[Modules]):
    key = (cls.__module__, cls.__name__)
    if key not in _MODULES:
        _MODULES[key] = cls
    return cls


def registry_op(op_name: str) -> Callable[[Type[Function]], Type[Function]]:
    from nova.autograd.function import Function

    def register(cls: Type[Function]):
        if not (isinstance(cls, type) and issubclass(cls, Function)):
            raise ValueError(
                f"Only Function classes can be registered, but got '{cls.__name__ if hasattr(cls, '__name__') else cls}'"
            )

        if op_name not in _OPS_REGISTERED:
            _OPS_REGISTERED.setdefault(op_name, cls)
        return cls

    return register


def get_registered_classes(module, name) -> Modules:
    if (module, name) in _MODULES:
        return _MODULES[(module, name)]
    else:
        raise KeyError(f"key '{(module, name)}' not found")
