from __future__ import annotations
import yaml
import traceback
from nova.core import YAML_FILE_PATH
from nova.utils.log_config import logger
from typing import Any, TYPE_CHECKING
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


def bootstrap_to(tensor_cls: Tensor | Any, yaml_path: str = YAML_FILE_PATH) -> None:

    try:
        native = native_yaml(yaml_path)

        for ops in native["ops"]:
            name = ops["name"]
            op = _OPS_REGISTERED[name]
            cfg = ops["tensor"]
            inplace = cfg.get("inplace", None)
            raw_args = ops.get("raw_args", False)
            is_unary = ops.get("is_unary", False)

            if "dunder" in cfg and not hasattr(tensor_cls, cfg["dunder"]):
                setattr(
                    tensor_cls, cfg["dunder"], make_forward_func(op, raw_args, is_unary)
                )

            if "reverse" in cfg and not hasattr(tensor_cls, cfg["reverse"]):
                setattr(tensor_cls, cfg["reverse"], make_reverse_func(op))

            if "method" in cfg and not hasattr(tensor_cls, cfg["method"]):
                setattr(tensor_cls, cfg["method"], make_method(op))

            if inplace is not None:
                for key in ["method", "dunder"]:
                    if key in inplace and not hasattr(tensor_cls, inplace[key]):
                        setattr(
                            tensor_cls,
                            inplace[key],
                            make_inplace_func(op, name, is_unary),
                        )

        logger.debug("All operations were successfully registered ✅")

    except Exception as e:
        exception_lines = [line for line in traceback.format_exception(e)]
        logger.error(
            "An error occurred during the loading of operations in the module.\n\n"
        )
        print(*exception_lines)
        raise
