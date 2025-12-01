import numpy as np
from typing import Optional
from novann._typing import ListOfParameters, InitFn

from novann.module import Parameters, Layer
from novann.core import DEFAULT_NORMAL_INIT_MAP


class Linear(Layer):
    """A fully connected (linear) layer.

    Computes a linear transformation: y = x W^T + b.

    Args:
        in_features (int): Number of input features (D_in).
        out_features (int): Number of output features (D_out).
        bias (bool): If True, an additive bias term is included. Default: True.
        init (InitFn, optional): Weight initialization function. If None,
            falls back to config.DEFAULT_NORMAL_INIT_MAP["default"].

    Attributes:
        weight (Parameters): Weight matrix with shape (out_features, in_features).
        bias (Optional[Parameters]): Bias vector with shape (1, out_features) or None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: Optional[InitFn] = None,
    ) -> None:
        """Initialize the Linear layer with given dimensions and optional bias."""
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.b: bool = bias  # Flag to track if bias was requested
        self.init_fn: Optional[InitFn] = init

        self.weight: Optional[Parameters] = None
        self.bias: Optional[Parameters] = None
        self._cache_input: Optional[np.ndarray] = None

        self.reset_parameters()

    def reset_parameters(self, initializer: Optional[InitFn] = None) -> None:
        """(Re)initialize weight and bias Parameters."""
        if initializer is not None:
            init = initializer
        elif self.init_fn is not None:
            init = self.init_fn
        else:
            # Fallback to default initializer if none specified
            init = DEFAULT_NORMAL_INIT_MAP["default"]

        # Weight shape: (out_features, in_features)
        w = init((self.out_features, self.in_features))
        self.weight = Parameters(np.asarray(w, dtype=np.float32))
        self.weight.name = "linear weight"

        if self.b:
            # Bias shape: (1, out_features)
            b = np.zeros((1, self.out_features), dtype=np.float32)
            self.bias = Parameters(np.asarray(b, dtype=np.float32))
            self.bias.name = "linear bias"
        else:
            self.bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: computes the linear transformation y = x W^T + b.

        The input is cached for the backward pass.

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, in_features).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, out_features).
        """
        x = x.astype(np.float32, copy=False)
        self._cache_input = x

        # Matrix multiplication: (N, D_in) @ (D_in, D_out) -> (N, D_out)
        out = x @ self.weight.data.T
        if self.bias is not None:
            out = out + self.bias.data
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass: computes gradients for parameters and returns grad w.r.t input.

        Args:
            grad (np.ndarray): Gradient w.r.t. the layer output,
                               shape (batch_size, out_features).

        Returns:
            np.ndarray: Gradient w.r.t. the layer input,
                        shape (batch_size, in_features).
        """
        grad = grad.astype(np.float32, copy=False)
        x = self._cache_input

        # Gradient w.r.t. Weights (W): grad.T @ x -> (D_out, D_in)
        self.weight.grad = grad.T @ x

        # Gradient w.r.t. Bias (b): sum over batch dimension
        if self.bias is not None:
            self.bias.grad = np.sum(grad, axis=0, keepdims=True)

        # Gradient w.r.t. Input (x): grad @ W -> (N, D_in)
        grad_input = grad @ self.weight.data
        return grad_input

    def parameters(self) -> ListOfParameters:
        """Return a list of Parameters owned by this layer."""
        params: ListOfParameters = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
