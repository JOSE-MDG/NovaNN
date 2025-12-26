from numpy import ndarray


def unbroadcasting(grad: ndarray, shape: tuple[int, ...]):
    # Delete extra dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Sum where the original dimension was 1
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
