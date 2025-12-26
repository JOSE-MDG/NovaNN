from __future__ import annotations
import traceback
import yaml
from nova.core import YAML_FILE_PATH
from nova.utils.log_config import logger
from typing import Any, TYPE_CHECKING, Type
from nova.utils.registry import _OPS_REGISTERED
from nova.utils import ensure_tensor


if TYPE_CHECKING:
    from nova import Tensor
    from nova.autograd.function import Function


def native_yaml(path: str = YAML_FILE_PATH) -> Any:
    try:
        with open(path, "r") as file:
            yml = yaml.safe_load(file)
        logger.debug("✅ YAML ops succefully loaded")
        return yml
    except Exception as e:
        exception_lines = [line for line in traceback.format_exception(e)]
        logger.error(
            "An error occurred during the chargin; please check that the path in correct.\n\n"
        )
        print(*exception_lines)


def bootstrap_to(tensor_cls: Type[Tensor] | Any, yaml_path: str = YAML_FILE_PATH):

    try:
        native = native_yaml(yaml_path)

        for ops in native["ops"]:

            name = ops["name"]
            op = _OPS_REGISTERED[name]
            tensor_cfg = ops["tensor"]

            if "dunder" in tensor_cfg:

                def make_function(cls: Type[Function]):
                    def method(self: Type[Tensor], other: Type[Tensor] | Any):
                        return cls.apply(self, ensure_tensor(other))

                    return method

                setattr(tensor_cls, tensor_cfg["dunder"], make_function(op))

            if "reverse" in tensor_cfg:

                def make_function(cls: Type[Function]):
                    def method(self: Type[Tensor], other: Type[Tensor] | Any):
                        return cls.apply(ensure_tensor(other), self)

                    return method

                setattr(tensor_cls, tensor_cfg["reverse"], make_function(op))

            if "method" in tensor_cfg:

                def make_function(cls: Type[Function]):
                    def func(self: Type[Tensor], *args, **kwargs):
                        return cls.apply(self, *args, **kwargs)

                    return func

                setattr(tensor_cls, ops["method"], make_function(op))

        logger.debug("All transactions were successfully processed ✅")
    except Exception as e:
        exception_lines = [line for line in traceback.format_exception(e)]
        logger.error(
            "An error occurred during the loading of operations in the module.\n\n"
        )
        print(*exception_lines)
