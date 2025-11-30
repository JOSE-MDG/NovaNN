from typing import Tuple, Any, Callable, List, Optional, Union
import numpy as np
from src.module import Parameters

# Tensor shape definition (batch_size, channels, height, width, ...)
Shape = Tuple[int, ...]

# Weight initialization function signature
InitFn = Callable[[Shape], np.ndarray]

# List of trainable parameters
ListOfParameters = List[Parameters]

# Alternative initialization function signature (for compatibility)
InitFnArg = Callable[[Shape], Any]  # Consider replacing with InitFn

# Flexible dimension specification (single value or tuple)
IntOrPair = Union[int, Tuple[int, int]]

# Activation function configuration (name, parameter)
ActivAndParams = Tuple[Optional[str], Optional[float]]

# Convolution-specific types
KernelSize = Union[int, Tuple[int, int]]
Stride = Union[int, Tuple[int, int]]
Padding = Union[int, Tuple[int, int], str]
