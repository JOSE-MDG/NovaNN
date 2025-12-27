import pickle
from nova.utils.decorators.registry import _MODULES
from nova.utils import get_registered_classes

ALLOWED_MODULES = {
    "numpy",
    "numpy.core.multiarray",
    "numpy.core.numeric",
    "numpy._core.numeric",
    "numpy._core.multiarray",
    "nova.dtypes",
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module_name, global_name):

        # Allow numpy modules
        if module_name in ALLOWED_MODULES:
            return super().find_class(module_name, global_name)

        if module_name.startswith("numpy") and global_name in (
            "_frombuffer",
            "scalar",
            "dtype",
        ):
            return super().find_class(module_name, global_name)

        if module_name == "builtins":
            import builtins

            if hasattr(builtins, global_name):
                return getattr(builtins, global_name)

        if module_name == "collections" and global_name == "OrderedDict":
            import collections

            return collections.OrderedDict

        cls = get_registered_classes(module=module_name, name=global_name)

        if cls is None:
            for (_, n), registered_cls in _MODULES.items():
                if n == global_name:
                    return registered_cls

        if cls is not None:
            return cls
        raise pickle.UnpicklingError(
            f"Blocked unpickling of {module_name}.{global_name}"
        )
