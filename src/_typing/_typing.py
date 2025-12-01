from typing import Tuple, Any, Callable, List, Optional, Union, TYPE_CHECKING
import numpy as np

# At runtime, this block is ignored.
if TYPE_CHECKING:
    from src.module import Parameters

# Tensor shape definition (batch_size, channels, height, width, ...)
Shape = Tuple[int, ...]

# Weight initialization function signature
InitFn = Callable[[Shape], np.ndarray]

# List of trainable parameters
# We use a forward reference (string literal "Parameters") since the class
# is not available at runtime, but the type checker will recognize it.
ListOfParameters = List["Parameters"]

# Alternative initialization function signature (for compatibility)
InitFnArg = Callable[[Shape], Any]

# Flexible dimension specification (single value or tuple)
IntOrPair = Union[int, Tuple[int, int]]

# Activation function configuration (name, parameter)
ActivAndParams = Tuple[Optional[str], Optional[float]]

# train, validation and test sets
TrainTestEvalSets = Tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]

# Convolution-specific types
KernelSize = Union[int, Tuple[int, int]]
Stride = Union[int, Tuple[int, int]]
Padding = Union[int, Tuple[int, int], str]
