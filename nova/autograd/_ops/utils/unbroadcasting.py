from typing import TYPE_CHECKING
from numpy import ndarray

if TYPE_CHECKING:
    from nova._typing import Size


def unbroadcasting(grad: ndarray, shape: Size) -> ndarray:
    """
    Reduce a broadcasted gradient back to the original tensor shape.
    """
    # Remove extra leading dimensions
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Find axes where broadcasting occurred (size 1 in original)
    axes_to_sum = []
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            axes_to_sum.append(i)

    # Sum all at once to avoid index shifting
    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)

    return grad
