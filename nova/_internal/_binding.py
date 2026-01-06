from __future__ import annotations
import yaml
import traceback
from nova.core import YAML_FILE_PATH
from nova.utils.logger import logger
from typing import TYPE_CHECKING
from nova.utils.decorators.registry import _OPS_REGISTERED
from ._generators import (
    make_forward_func,
    make_reverse_func,
    make_inplace_func,
    make_method,
)


if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import YAMLFile


def native_yaml(path: str = YAML_FILE_PATH) -> YAMLFile:
    """
    Loads and parses the native operations YAML configuration file.

    This function reads the YAML file that defines all native operations,
    their signatures, and how they should be bound to the Tensor class.

    Args:
        path (str): Path to the YAML configuration file. Defaults to the framework's
            native operations file.

    Returns:
        Parsed YAML content as a dictionary containing operation definitions.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist at the specified path.
        yaml.YAMLError: If the file contains invalid YAML syntax.

    Examples:
        >>> ops_config = native_yaml()
        >>> print(ops_config['ops'][0]['name'])
        'add'
    """
    try:
        with open(path, "r") as file:
            yml = yaml.safe_load(file)
        logger.debug("✅ YAML ops successfully loaded")
        return yml
    except Exception as e:
        exception_lines = [line for line in traceback.format_exception(e)]
        logger.error(
            "An error occurred during the loading; please check that the path is correct.\n\n"
        )
        print(*exception_lines)
        raise


def bootstrap_to(tensor_cls: type[Tensor], yaml_path: str = YAML_FILE_PATH) -> None:
    """
    Dynamically binds operations from YAML configuration to the Tensor class.

    This is the core bootstrapping mechanism that reads operation definitions
    from a YAML file and dynamically attaches them as methods to the Tensor class.
    It handles multiple method types: dunder methods (__add__), reverse operations
    (__radd__), regular methods (add), and in-place variants (add_).

    The binding process:
    1. Loads operation definitions from YAML
    2. Retrieves registered Function classes from the operations registry
    3. Generates appropriate method wrappers using generator functions
    4. Attaches methods to the Tensor class if they don't already exist

    Args:
        tensor_cls (type[Tensor]): The Tensor class to which operations will be bound.
        yaml_path (str): Path to the YAML configuration file defining operations.

    Raises:
        KeyError: If an operation references a Function that isn't registered.
        RuntimeError: If method binding fails due to configuration errors.

    Notes:
        - Only binds methods that don't already exist on the class
        - Supports both unary (single input) and binary operations
        - In-place operations are automatically generated for mutable variants
        - Raw_args flag controls whether arguments are auto-converted to Tensors

    Examples:
        >>> # Internal usage during framework initialization `nova/__init__.py`
        >>> from nova import Tensor
        >>> bootstrap_to(Tensor)  # Binds all operations from YAML
    """
    try:
        native = native_yaml(yaml_path)

        for ops in native["ops"]:
            name = ops["name"]
            op = _OPS_REGISTERED[name]
            cfg = ops["tensor"]
            inplace = cfg.get("inplace", None)
            raw_args = ops.get("raw_args", False)
            is_unary = ops.get("is_unary", False)

            # Bind dunder method (e.g., __add__)
            if "dunder" in cfg and not hasattr(tensor_cls, cfg["dunder"]):
                setattr(
                    tensor_cls, cfg["dunder"], make_forward_func(op, raw_args, is_unary)
                )

            # Bind reverse dunder method (e.g., __radd__)
            if "reverse" in cfg and not hasattr(tensor_cls, cfg["reverse"]):
                setattr(tensor_cls, cfg["reverse"], make_reverse_func(op))

            # Bind regular method (e.g., add)
            if "method" in cfg and not hasattr(tensor_cls, cfg["method"]):
                setattr(tensor_cls, cfg["method"], make_method(op))

            # Bind in-place variants (e.g., add_, __iadd__)
            if inplace is not None:
                for key in ["method", "dunder"]:
                    if key in inplace and not hasattr(tensor_cls, inplace[key]):
                        setattr(
                            tensor_cls,
                            inplace[key],
                            make_inplace_func(op, raw_args, name, is_unary),
                        )

        logger.debug("All operations were successfully registered ✅")

    except Exception as e:
        exception_lines = [line for line in traceback.format_exception(e)]
        logger.error(
            "An error occurred during the loading of operations in the module.\n\n"
        )
        print(*exception_lines)
        raise
