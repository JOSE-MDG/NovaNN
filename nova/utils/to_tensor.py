from __future__ import annotations
import traceback
import nova
from typing import Any, Optional
from nova.utils.logger import get_logger
from numpy import ndarray

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype

logger = get_logger()


def ensure_tensor(
    obj: Any, dtype: Optional[Dtype] = None, requires_grad: Optional[bool] = None
) -> Tensor:
    """
    Convert an arbitrary object to a NovaNN Tensor, preserving or overriding dtype and gradient settings.

    This function handles multiple input types and ensures that the result is a Tensor
    compatible with NovaNN's autograd system. It is safe and efficient:
    - Returns the original tensor if no modifications are needed.
    - Creates a new tensor if `dtype` or `requires_grad` differ from the input.
    - Converts numpy arrays, Python scalars, and lists to tensors.

    Supported input types:
        - `Tensor`: returned as-is if no changes requested, otherwise copied with new settings.
        - `numpy.ndarray`: converted to a Tensor, preserving dtype unless overridden.
        - Python scalar (`int`, `float`, `bool`) or list/tuple: converted to a Tensor with appropriate dtype.

    Args:
        obj (Any): Object to convert. Can be Tensor, numpy.ndarray, Python scalar, or list/tuple.
        dtype (Optional[Dtype]): Desired data type for the resulting Tensor.
            If None, inferred from the input object.
        requires_grad (Optional[bool]): Whether the resulting Tensor should track gradients.
            If None, defaults to False for non-Tensor objects, or preserves the input Tensor's setting.

    Returns:
        Tensor: A NovaNN Tensor corresponding to the input object, with specified dtype and gradient tracking.

    Raises:
        Logs an error and prints a traceback if conversion fails.

    Examples:
        >>> import numpy as np
        >>> import nova
        >>> from nova.utils import ensure_tensor
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> t = ensure_tensor(x, dtype=nova.float32, requires_grad=True)
        >>> isinstance(t, nova.Tensor)
        True
        >>> t.type()
        'nova.float32'
        >>> t.requires_grad
        True

        >>> y = ensure_tensor(5)
        >>> y.data
        array(5)
        >>> y.type()
        'nova.int64'

        >>> z = nova.tensor([1, 2, 3])
        >>> z2 = ensure_tensor(z, requires_grad=False)
        >>> z2 is not z
        True
        >>> z2.requires_grad
        False
    """
    from nova import Tensor

    try:
        # Case 1: Already a Tensor
        if isinstance(obj, Tensor):
            if dtype is None and requires_grad is None:
                return obj
            new_dtype = dtype if dtype is not None else obj.dtype
            new_requires_grad = (
                requires_grad if requires_grad is not None else obj.requires_grad
            )
            return Tensor(obj.data, dtype=new_dtype, requires_grad=new_requires_grad)

        # Case 2: NumPy ndarray
        elif isinstance(obj, ndarray):
            inferred_dtype = dtype if dtype is not None else obj.dtype
            inferred_requires_grad = (
                requires_grad if requires_grad is not None else False
            )
            return Tensor(
                obj, dtype=inferred_dtype, requires_grad=inferred_requires_grad
            )

        # Case 3: Python scalar or list/tuple
        else:
            base_dtype = dtype or nova.float32
            if isinstance(obj, bool):
                base_dtype = nova.bool
            elif isinstance(obj, int):
                base_dtype = nova.long
            elif isinstance(obj, float):
                base_dtype = dtype or nova.float32
            return Tensor(obj, dtype=base_dtype, requires_grad=requires_grad or False)

    except Exception as e:
        exception_lines = [line for line in traceback.format_exception(e)]
        logger.error(
            "An error occurred during the conversion; please verify that the data type is as expected.\n\n"
        )
        print(*exception_lines)
