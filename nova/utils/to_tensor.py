from __future__ import annotations
import traceback
from typing import Any, Optional
from nova.utils.log_config import logger
from numpy import ndarray

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype


def ensure_tensor(
    obj: Any, dtype: Optional[Dtype] = None, requires_grad: Optional[bool] = None
) -> Tensor:

    from nova import Tensor

    try:

        # Case 1: Already a Tensor
        if isinstance(obj, Tensor):

            # If it's already a tensor and no parameters were specified, return unchanged
            # This is important for efficiency: we avoid unnecessary copies

            if dtype is None and requires_grad is None:

                return obj

            # If parameters were specified, we need to create a new tensor
            # because modifying the original would be an unexpected in-place operation

            new_dtype = dtype if dtype is not None else obj.dtype

            new_requires_grad = (
                requires_grad if requires_grad is not None else obj.requires_grad
            )

            return Tensor(obj.data, dtype=new_dtype, requires_grad=new_requires_grad)

        # Case 2: Is a NumPy ndarray

        elif isinstance(obj, ndarray):

            # Preserve the NumPy array's dtype unless otherwise specified
            inferred_dtype = dtype if dtype is not None else obj.dtype
            inferred_requires_grad = (
                requires_grad if requires_grad is not None else False
            )

            return Tensor(
                obj, dtype=inferred_dtype, requires_grad=inferred_requires_grad
            )

            # Case 3: It's a Python scalar or list

        else:

            # For scalars and lists, use float32 as the default if not specified
            # This is consistent with PyTorch and avoids precision issues

            inferred_dtype = dtype if dtype is not None else None
            inferred_requires_grad = (
                requires_grad if requires_grad is not None else False
            )

            return Tensor(
                obj, dtype=inferred_dtype, requires_grad=inferred_requires_grad
            )
    except Exception as e:
        exception_lines = [line for line in traceback.format_exception(e)]
        logger.error(
            "An error occurred during the conversion; please verify that the data type is as expected.\n\n"
        )
        print(*exception_lines)
