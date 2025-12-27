from __future__ import annotations
import yaml
import traceback
import numpy as np
from nova.core import YAML_FILE_PATH
from nova.utils.log_config import logger
from typing import Any, TYPE_CHECKING, Type
from nova.utils.decorators.registry import _OPS_REGISTERED
from nova.utils import ensure_tensor


if TYPE_CHECKING:
    from nova import Tensor
    from nova.autograd.function import Function


def native_yaml(path: str = YAML_FILE_PATH) -> dict[str, list[dict[str, dict | Any]]]:
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


def bootstrap_to(tensor_cls: Type[Tensor] | Any, yaml_path: str = YAML_FILE_PATH):
    from nova import Tensor

    try:
        native = native_yaml(yaml_path)

        for ops in native["ops"]:
            name = ops["name"]
            op = _OPS_REGISTERED[name]
            tensor_cfg = ops["tensor"]
            inplace_cfg = tensor_cfg.get("inplace", None)
            raw_args = ops.get("raw_args", False)

            if "dunder" in tensor_cfg:

                def make_forward_function(cls: Type[Function], raw: bool):
                    def method(self, other, _cls=cls, _raw: bool = raw):
                        if not _raw:
                            if not isinstance(other, Tensor):
                                other = ensure_tensor(other)
                        return _cls.apply(self, other)

                    return method

                if not hasattr(tensor_cls, tensor_cfg["dunder"]):
                    setattr(
                        tensor_cls,
                        tensor_cfg["dunder"],
                        make_forward_function(op, raw_args),
                    )

            if "reverse" in tensor_cfg:

                def make_reverse_function(cls: Type[Function]):
                    def method(self, other, _cls=cls):
                        if not isinstance(other, Tensor):
                            other = ensure_tensor(other)
                        return _cls.apply(other, self)

                    return method

                if not hasattr(tensor_cls, tensor_cfg["reverse"]):
                    setattr(
                        tensor_cls, tensor_cfg["reverse"], make_reverse_function(op)
                    )

            if "method" in tensor_cfg:

                def make_method_function(cls: Type[Function]):
                    def func(self: Type[Tensor], *args, _cls=cls, **kwargs):
                        return _cls.apply(self, *args, **kwargs)

                    return func

                if not hasattr(tensor_cls, tensor_cfg["method"]):
                    setattr(tensor_cls, tensor_cfg["method"], make_method_function(op))

            if inplace_cfg is not None:

                def make_inplace_function(cls: Type[Function], op_name: str):
                    def inplace_method(
                        self: Type[Tensor],
                        other: Type[Tensor] | Any,
                        _cls=cls,
                        _op_name=op_name,
                    ):
                        if self.requires_grad:
                            raise RuntimeError(
                                f"Cannot perform inplace operation '{_op_name}_' on a tensor "
                                f"that requires gradients. Use the out-of-place version instead."
                            )

                        if not isinstance(other, Tensor):
                            other = ensure_tensor(other)

                        result = _cls.apply(self, other).data

                        np.copyto(dst=self.data, src=result)

                        return self

                    return inplace_method

                if "method" in inplace_cfg:
                    inplace_method_name = inplace_cfg["method"]
                    if not hasattr(tensor_cls, inplace_method_name):
                        setattr(
                            tensor_cls,
                            inplace_method_name,
                            make_inplace_function(op, name),
                        )

                if "dunder" in inplace_cfg:
                    inplace_dunder_name = inplace_cfg["dunder"]
                    if not hasattr(tensor_cls, inplace_dunder_name):
                        setattr(
                            tensor_cls,
                            inplace_dunder_name,
                            make_inplace_function(op, name),
                        )

        logger.debug("All operations were successfully registered ✅")

    except Exception as e:
        exception_lines = [line for line in traceback.format_exception(e)]
        logger.error(
            "An error occurred during the loading of operations in the module.\n\n"
        )
        print(*exception_lines)
        raise
