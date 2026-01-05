from typing import TYPE_CHECKING
from numpy import ndarray

if TYPE_CHECKING:
    from nova._typing import Size


def unbroadcasting(grad: ndarray, shape: Size) -> ndarray:
    """
    Reduce a broadcasted gradient back to the original tensor shape.

    This function reverses NumPy broadcasting effects applied during
    the forward pass by summing gradients along the broadcasted dimensions.

    Forward (conceptual example):
        input shape  = (3, 1)
        other shape  = (3, 4)

        Broadcasting expands input to (3, 4)

    Backward:
        grad_output shape = (3, 4)
        grad_input must be reduced back to (3, 1)

    Example:
        >>> input = np.ones((3, 1))
        >>> other = np.ones((3, 4))
        >>> out = input + other        # shape (3, 4)
        >>> grad_output = np.ones((3, 4))

        >>> grad_input = unbroadcasting(grad_output, input.shape)
        >>> grad_input.shape
        (3, 1)

        >>> grad_input
        array([[4.],
               [4.],
               [4.]])

    Explanation:
        - Extra leading dimensions are summed.
        - Dimensions with size 1 in the original shape are summed
          along that axis to collapse the broadcast.

    This is essential for correctly computing gradients of
    element-wise operations involving broadcasting.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
