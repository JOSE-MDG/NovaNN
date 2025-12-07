from typing import Tuple, Any, Callable, List, Optional, Union, TYPE_CHECKING, Iterator
import numpy as np

# --- Conditional imports for type hints only ---
if TYPE_CHECKING:
    from novann.optim import Adam, SGD, RMSprop
    from novann.losses import CrossEntropyLoss, BinaryCrossEntropy, MAE, MSE
    from novann.module import Parameters

# Tensor shape definition (batch_size, channels, height, width, ...)
Shape = Tuple[int, ...]

# Weight initialization function signature
InitFn = Callable[[Shape], np.ndarray]

# List of trainable parameters
ListOfParameters = List["Parameters"]


# Activation function configuration (name, parameter)
ActivAndParams = Tuple[Optional[str], Optional[float]]

# train, validation and test sets
TrainTestEvalSets = Tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]

# Optimizer type alias
Optimizer = "Adam | SGD | RMSprop"

# Loss function type alias
LossFunc = "BinaryCrossEntropy | CrossEntropyLoss | MAE | MSE"

# Date loader type
Loader = Iterator[Tuple[np.ndarray, np.ndarray]]

# Convolution-specific types
KernelSize = Union[int, Tuple[int, int]]
Stride = Union[int, Tuple[int, int], None]
Padding = Union[int, Tuple[int, int], str]
