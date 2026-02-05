from __future__ import annotations
import nova
import numpy as np
from enum import Enum
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Size


class OpCategory(Enum):
    """Categorías de operaciones según su signatura"""

    UNARY = "unary"  # f(x)
    BINARY = "binary"  # f(x, y)
    REDUCTION = "reduction"  # f(x, dim=None, keepdims=False)
    SHAPE = "shape"  # f(x, *size)
    SPECIAL = "special"  # Requiere handling especial


# Ops by category
OPERATIONS = {
    # Unarys - f(x)
    OpCategory.UNARY.value: [
        "exp",
        "log",
        "sqrt",
        "abs",
        "relu",
        "sigmoid",
        "gelu",
        "sin",
        "cos",
        "tan",
        "tanh",
        "clone",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
        "cot",
        "sec",
        "csc",
        "arccot",
        "arcsec",
        "arccsc",
    ],
    # Binarys - f(x, y)
    OpCategory.BINARY.value: [
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "matmul",
        "dot",
        "maximum",
        "minimum",
        "atan2",
    ],
    # Reductions - f(x, dim=None, keepdims=False)
    OpCategory.REDUCTION.value: [
        "sum",
        "mean",
        "var",
        "max",
        "min",
    ],
    # Shape - f(x, *args)
    OpCategory.SHAPE.value: [
        "reshape",
        "view",
        "permute",
        "squeeze",
        "unsqueeze",
        "tile",
        "repeat",
        "pad",
    ],
    # Specials - require custom handling
    OpCategory.SPECIAL.value: [
        "det",
        "inv",  # requires square matrices
        "norm",
        "diag",  # special args
        "split",
        "clamp",
        "extend",
        "trace",  # Reduce dimentions, itsn't simple unary
        "sign",
        "ceil",  # They have no gradients or require special handling
    ],
}


# Reverse mapping for quick lookup: op_name -> category
OP_TO_CATEGORY = {}
for category, ops in OPERATIONS.items():
    for op in ops:
        OP_TO_CATEGORY[op] = category

# Operations that should NOT be tested with gradient checking
SKIP_GRAD_CHECK = {
    "getitem",
    "setitem",  # Non-differentiable
    "as_strided",  # Unsafe
    "argmax",
    "argmin",
    "argsort",  # Discrete outputs
    "sign",
    "ceil",  # No useful gradients (grad = 0 almost everywhere)
    "split",  # Return list, not single tensor
    "pow",  # NaNs with negative numbers
    "tile",
    "norm",  # Requiere a carefull implementation
}


def make_test_input(op_name: str, shape: Size, requires_grad: bool = False) -> Tensor:
    """
    Generates appropriate input data for a given operation,
    ensuring values stay within valid mathematical domains.
    """
    if op_name in ["acosh", "arcsec", "arccsc"]:
        data = np.random.uniform(1.5, 11.5, size=shape)
        return nova.tensor(data.astype(np.float32), requires_grad=requires_grad)

    elif op_name in ["atanh", "arcsin", "arccos"]:
        data = np.random.uniform(-0.9, 0.9, size=shape)
        return nova.tensor(data.astype(np.float32), requires_grad=requires_grad)

    elif op_name == "asinh":
        data = np.random.uniform(0.1, 10.0, size=shape)
        return nova.tensor(data.astype(np.float32), requires_grad=requires_grad)

    elif op_name in ["log", "sqrt"]:
        data = np.random.rand(*shape) + 0.5
        return nova.tensor(data.astype(np.float32), requires_grad=requires_grad)

    elif op_name in ["det", "inv", "trace"]:
        n = min(shape) if len(shape) > 1 else shape[0]
        return nova.randn(n, n, requires_grad=requires_grad, dtype=nova.float32)

    # Default
    return nova.randn(*shape, requires_grad=requires_grad, dtype=nova.float32)


def create_op_wrapper(op_name: str) -> Callable[[Tensor], Tensor]:
    """
    Creates a wrapper that calls the operation correctly.

    Args:
        op_name: Name of the operation

    Returns:

        Function that takes a tensor and executes the operation
    """
    category = OP_TO_CATEGORY.get(op_name, OpCategory.SPECIAL.value)

    # UNARY: Simply call method
    if category == OpCategory.UNARY.value:
        return lambda x: getattr(x, op_name)()

    # BINARY: Use clone as second argument
    if category == OpCategory.BINARY.value:
        if op_name == "matmul":
            # matmul needs compatible shapes
            return lambda x: x @ x.T
        elif op_name == "dot":
            # dot needs vectors or 2D matrices
            return lambda x: x.flatten().dot(x.flatten())
        else:
            # Op element-wise
            return lambda x: getattr(x, op_name)(x.clone() * 0.5)

    # REDUCTION: Reduce on first dimension
    if category == OpCategory.REDUCTION.value:
        return lambda x: getattr(x, op_name)(dim=0, keepdims=False)

    # SHAPE: Specific operations
    if category == OpCategory.SHAPE.value:
        if op_name == "reshape":
            return lambda x: x.reshape(x.numel(), 1)
        elif op_name == "view":
            return lambda x: x.view(-1)
        elif op_name == "permute":
            return lambda x: x.permute(*reversed(range(x.dim())))
        elif op_name == "squeeze":
            return lambda x: x.unsqueeze(0).squeeze(0)
        elif op_name == "unsqueeze":
            return lambda x: x.unsqueeze(0)
        elif op_name == "tile":
            return lambda x: x.tile((2, 2))
        elif op_name == "repeat":
            return lambda x: x.repeat(2, dim=0)
        elif op_name == "pad":
            return lambda x: x.pad(((1, 1), (1, 1)), mode="constant")

    # SPECIAL: Specific cases
    if op_name == "det":
        return lambda x: x.det()
    elif op_name == "inv":
        # Add identity to ensure invertibility
        return lambda x: (x + nova.eye(x.shape[0]) * 2.0).inv()
    elif op_name == "norm":
        return lambda x: x.norm(ord=2, dim=None, keepdims=False)
    elif op_name == "diag":
        return lambda x: x.diag(diagonal=0)
    elif op_name == "clamp":
        # clamp expects min/max as positional arguments or kwargs
        return lambda x: x.clamp(-0.5, 0.5)
    elif op_name == "split":
        # `split` returns a list; we take the first element
        # and sum it to get a scalar output
        return lambda x: x.split(2, dim=0)[0].sum()
    elif op_name == "extend":
        return lambda x: x.extend(1, *x.shape)
    elif op_name == "trace":
        return lambda x: x.trace()
    elif op_name == "sign":
        return lambda x: x.sign()
    elif op_name == "ceil":
        return lambda x: x.ceil()

    # Fallback:
    return lambda x: getattr(x, op_name)()


# List of all operations to be tested (excluding those without a grad)
ALL_TESTABLE_OPS = [
    op for ops in OPERATIONS.values() for op in ops if op not in SKIP_GRAD_CHECK
]
