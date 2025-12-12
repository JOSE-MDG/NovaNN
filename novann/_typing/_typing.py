import numpy as np
from typing import (
    Tuple,
    Callable,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
    Iterator,
    Iterable,
)

# --- Conditional imports for type hints only ---
if TYPE_CHECKING:
    from novann.optim import Adam, SGD, RMSprop
    from novann import CrossEntropyLoss, BinaryCrossEntropy, MAE, MSE
    from novann.module import Parameters, Layer
    from novann.layers.activations import Activation

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

# Array or extras
ArrayAndExtras = np.ndarray | Tuple[np.ndarray, ...]

# Modules aliases
Modules = "Layer | Activation"

# iterable object of parameters
IterableParameters = Iterable["Parameters"]

# beta coefficients for the optimizers
BetaCoefficients = Tuple[float, float] | float

# Optimizer type alias
Optimizer = "Adam | SGD | RMSprop | AdamW"

# Loss function type alias
LossFunc = "BinaryCrossEntropy | CrossEntropyLoss | MAE | MSE"

# Date loader type
Loader = Iterator[Tuple[np.ndarray, np.ndarray]]

# Convolution-specific types
KernelSize = Union[int, Tuple[int, int]]
Stride = Union[int, Tuple[int, int], None]
Padding = Union[int, Tuple[int, int], str]
