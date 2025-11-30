import numpy as np
from typing import Optional, Callable
from src._typing import ListOfParameters, InitFn

from src.module.module import Parameters
from src.module.layer import Layer
from src.core import config


class Linear(Layer):
    """Fully connected (linear) layer.

    Computes a linear transformation: y = x W^T + b

    This class stores weights and optional bias as Parameters objects so the
    training utilities in the project can inspect and update them.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): If True, include an additive bias term. Default: True.
        init (Optional[Callable]): Optional initializer function that receives a
            shape tuple (out_features, in_features) and returns an array. If
            None, falls back to config.DEFAULT_NORMAL_INIT_MAP["default"].

    Attributes:
        in_features (int)
        out_features (int)
        weight (Parameters): Weight matrix with shape (out_features, in_features).
        bias (Optional[Parameters]): Bias row vector with shape (1, out_features) or None.
        b (bool): Original bias flag.
        init_fn (Optional[Callable]): Stored initializer function.
        _cache_input (Optional[np.ndarray]): Cached input used in backward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)

        # Parameters are created in reset_parameters()
        self.weight: Optional[Parameters] = None
        self.bias: Optional[Parameters] = None

        self.b: bool = bool(bias)
        self.init_fn: Optional[InitFn] = init

        # Cached input for computing gradients in backward()
        self._cache_input: Optional[np.ndarray] = None

        self.reset_parameters()

    def reset_parameters(self, initializer: Optional[InitFn] = None) -> None:
        """(Re)initialize weight and bias Parameters.

        Args:
            initializer: Optional callable to create initial arrays. If None,
                the layer uses self.init_fn or the project's default initializer.
        """
        if initializer is not None:
            init = initializer
        elif self.init_fn is not None:
            init = self.init_fn
        else:
            init = config.DEFAULT_NORMAL_INIT_MAP["default"]
        w = init((self.out_features, self.in_features))
        self.weight = Parameters(np.asarray(w))
        self.weight.name = "weight"

        if self.b:
            self.bias = Parameters(np.zeros((1, self.out_features)))
            self.bias.name = "bias"
        else:
            self.bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: linear transformation.

        Args:
            x (np.ndarray): Input array with shape (batch_size, in_features).

        Returns:
            np.ndarray: Output array with shape (batch_size, out_features).
        """
        self._cache_input = x
        out = x @ self.weight.data.T
        if self.bias is not None:
            out = out + self.bias.data
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass: compute gradients for parameters and return grad wrt input.

        Updates:
            - self.weight.grad shaped (out_features, in_features)
            - self.bias.grad shaped (1, out_features) if bias exists

        Args:
            grad (np.ndarray): Gradient w.r.t. the layer output,
                               shape (batch_size, out_features).

        Returns:
            np.ndarray: Gradient w.r.t. the layer input,
                        shape (batch_size, in_features).
        """
        x = self._cache_input

        # weight.grad: sum over batch of outer products between output-grad and input
        # grad.T @ x -> (out_features, in_features)
        self.weight.grad = grad.T @ x

        if self.bias is not None:
            # bias gradient is per-output summed over batch
            self.bias.grad = np.sum(grad, axis=0, keepdims=True)

        # gradient w.r.t. input: grad @ W
        grad_input = grad @ self.weight.data
        return grad_input

    def parameters(self) -> ListOfParameters:
        """Return a list of Parameters owned by this layer.

        Returns:
            List[Parameters]: [weight] or [weight, bias] if bias is present.
        """
        params: ListOfParameters = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
