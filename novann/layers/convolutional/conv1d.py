import numpy as np
import novann.functional as F
from novann.module import Layer, Parameters
from novann._typing import InitFn, ListOfParameters, KernelSize, Stride, Padding
from novann.core import DEFAULT_UNIFORM_INIT_MAP
from typing import Optional


class Conv1d(Layer):
    """Applies a 1D convolution over an input signal composed of several input planes.

    The input must have shape (N, C_in, L_in).

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (KernelSize): Size of the convolving kernel.
        stride (Stride): Stride of the convolution. Default: 1.
        padding (Padding): Zero-padding added to both sides of the input. Default: 0.
        bias (bool): If True, adds a learnable bias to the output. Default: True.
        padding_mode (str): 'zeros', 'reflect', 'replicate', or 'circular'. Default: 'zeros'.
        init (InitFn, optional): Weight initialization function. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize,
        stride: Stride = 1,
        padding: Padding = 0,
        bias: bool = True,
        padding_mode: str = "zeros",
        init: InitFn = None,
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.K: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding
        self.pm: str = padding_mode

        self.init_fn: InitFn = init
        self.use_bias: bool = bias
        self.weight: Optional[Parameters] = None
        self.bias: Optional[Parameters] = None
        self._cache = {}

        self.reset_parameters()

    def reset_parameters(self, initializer: Optional[InitFn] = None) -> None:
        """(Re)initialize weight and bias Parameters."""
        if initializer is not None:
            init = initializer
        elif self.init_fn is not None:
            init = self.init_fn
        else:
            # Note: Assuming 'relu' is a key in DEFAULT_UNIFORM_INIT_MAP
            init = DEFAULT_UNIFORM_INIT_MAP["relu"]
        # Shape: (out_channels, in_channels, kernel_size)
        w = init((self.out_channels, self.in_channels, self.K))
        self.weight = Parameters(np.asarray(w, dtype=np.float32))
        self.weight.name = "conv1d weight"
        if self.use_bias:
            # Shape: (out_channels, 1)
            self.bias = Parameters(np.zeros((self.out_channels, 1), dtype=np.float32))
            self.bias.name = "conv1d bias"
        else:
            self.bias = None

    def _col2im(self, col: np.ndarray) -> np.ndarray:
        """Transforms column matrix back to padded image shape for gradient computation.

        Args:
            col: Gradient w.r.t. the col matrix, shape (C*K, N*L_out).

        Returns:
            Gradient w.r.t. the original unpadded input (N, C, L).
        """
        N, C, L = self._cache["x_shape"]
        L_out = self._cache["L_out"]
        x_p_shape = self._cache["x_p_shape"]

        cols_reshaped = col.reshape(C, self.K, N, L_out).transpose(2, 0, 1, 3)

        grad_input = np.zeros(x_p_shape, dtype=np.float32)
        grad_window = cols_reshaped.transpose(0, 1, 3, 2)  # (N, C, L_out, K)

        # Sum contributions of gradient windows back to the padded input
        for n in range(N):
            for l_out in range(L_out):
                start = l_out * self.stride
                grad_input[n, :, start : start + self.K] += grad_window[n, :, l_out, :]

        if self.padding == 0:
            return grad_input

        # Remove padding
        return grad_input[:, :, self.padding : self.padding + L]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 1D convolution.

        Args:
            x (np.ndarray): Input array with shape (N, C_in, L_in).

        Returns:
            np.ndarray: Output array with shape (N, C_out, L_out).
        """
        x = x.astype(np.float32, copy=False)

        # Reshape back to (N, C_out, L_out)
        out, col, w_col, L_out, x_p_shape = F.conv1d(
            x,
            self.weight,
            self.K,
            self.stride,
            self.padding,
            padding_mode=self.pm,
            bias=self.bias,
            extras=True,
        )

        self._cache["col"] = col
        self._cache["w_col"] = w_col
        self._cache["x_shape"] = x.shape
        self._cache["L_out"] = L_out
        self._cache["x_p_shape"] = x_p_shape

        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for 1D convolution.

        Args:
            grad (np.ndarray): Gradient w.r.t. the layer output, shape (N, C_out, L_out).

        Returns:
            np.ndarray: Gradient w.r.t. the layer input, shape (N, C_in, L_in).
        """
        grad = grad.astype(np.float32, copy=False)
        col = self._cache["col"]
        w_col = self._cache["w_col"]

        # Reshape gradient for matrix multiplication: (out_c, N*L_out)
        grad_reshaped = grad.transpose(1, 0, 2).reshape(self.out_channels, -1)

        # Gradient w.r.t. Weights (kernel): (out_c, N*L_out) @ (N*L_out, C*K) -> (out_c, C*K)
        self.weight.grad = (grad_reshaped @ col.T).reshape(self.weight.data.shape)

        # Gradient w.r.t. Bias: sum over batch and length dimension
        if self.bias is not None:
            self.bias.grad = np.sum(grad_reshaped, axis=1).reshape(-1, 1)

        # Gradient w.r.t. Col matrix: (C*K, out_c) @ (out_c, N*L_out) -> (C*K, N*L_out)
        grad_col = w_col.T @ grad_reshaped
        grad_input = self._col2im(grad_col)

        return grad_input

    def parameters(self) -> ListOfParameters:
        """Return a list of Parameters owned by this layer."""
        params: ListOfParameters = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        return f"Conv1d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.K}, stride={self.stride}, padding={self.padding}, bias={self.bias})"
