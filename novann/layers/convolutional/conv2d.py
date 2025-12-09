import numpy as np
from numpy.lib.stride_tricks import as_strided
from novann.module import Layer, Parameters
from typing import Optional, Tuple
from novann._typing import InitFn, Shape, ListOfParameters, KernelSize, Padding, Stride
from novann.core import DEFAULT_UNIFORM_INIT_MAP


class Conv2d(Layer):
    """Applies a 2D convolution over an input signal composed of several input planes.

    The input must have shape (N, C_in, H_in, W_in).

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (KernelSize): Size of the convolving kernel (H, W).
        stride (Stride): Stride of the convolution (H, W). Default: 1.
        padding (Padding): Zero-padding added to both sides (H, W) or 'valid'. Default: 0.
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KH, self.KW = self._pair(kernel_size)

        stride_val = stride if stride is not None else 1
        self.sh, self.sw = self._pair(stride_val)

        self.ph, self.pw = self._pair(padding)
        self.pm: str = padding_mode

        self.init_fn = init
        self.use_bias = bias
        self.weight = None
        self.bias = None

        self.reset_parameters()
        self._cache = {}

    def reset_parameters(self, initializer: Optional[InitFn] = None):
        """(Re)initialize weight and bias Parameters."""
        if initializer is not None:
            init = initializer
        elif self.init_fn is not None:
            init = self.init_fn
        else:
            init = DEFAULT_UNIFORM_INIT_MAP["relu"]
        # Shape: (out_channels, in_channels, KH, KW)
        w = init((self.out_channels, self.in_channels, self.KH, self.KW))
        self.weight = Parameters(np.asarray(w, dtype=np.float32))
        self.weight.name = "conv2d weight"
        if self.use_bias:
            # Shape: (out_channels, 1)
            self.bias = Parameters(np.zeros((self.out_channels, 1), dtype=np.float32))
            self.bias.name = "conv2d bias"
        else:
            self.bias = None

    def _pair(self, x: Padding) -> Tuple[int, int]:
        """Converts integer, tuple, or valid/same string to a (H, W) pair."""
        if isinstance(x, int):
            return (x, x)
        if isinstance(x, str):
            if x == "valid":
                return (0, 0)
            if x == "same":
                raise ValueError(f"The 'same' value is not currently supported")
            else:
                raise ValueError(f"Unsupported value '{x}'")
        return tuple(x)

    def _calc_out_size(self, height: int, width: int) -> Tuple[int, int]:
        """Calculates the output height and width (H_out, W_out)."""
        out_height = (height + 2 * self.ph - self.KH) // self.sh + 1
        out_width = (width + 2 * self.pw - self.KW) // self.sw + 1
        return out_height, out_width

    def _build_inidices(
        self, x_shape: Shape
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds indices (k, i, j) for the col2im operation."""
        _, C, H, W = x_shape
        out_height, out_width = self._calc_out_size(H, W)

        i0 = np.repeat(np.arange(self.KH), self.KW)
        j0 = np.tile(np.arange(self.KW), self.KH)

        i0 = np.tile(i0, C)  # length = C*KH*KW
        j0 = np.tile(j0, C)

        i1 = self.sh * np.repeat(
            np.arange(out_height), out_width
        )  # length = out_h*out_w
        j1 = self.sw * np.tile(np.arange(out_width), out_height)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)  # shape (K,L) where L = out_h*out_w
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # shape (K,L)
        k = np.repeat(np.arange(C), self.KH * self.KW).reshape(-1, 1)

        return k, i, j

    def _add_padding(self, x: np.ndarray) -> np.ndarray:
        """Applies padding to the input tensor."""
        pad_width = ((0, 0), (0, 0), (self.ph, self.ph), (self.pw, self.pw))
        # zeros, reflect, replicate, circular -> constant, reflect, edge, wrap in numpy
        padding_modes = ("zeros", "reflect", "replicate", "circular")
        if self.pm in padding_modes:
            if self.pm == "zeros":
                mode = "constant"
            elif self.pm == "reflect":
                mode = "reflect"
            elif self.pm == "replicate":
                mode = "edge"
            else:
                mode = "wrap"
        else:
            raise ValueError(
                f"padding_mode Only accept {padding_modes} not '{self.pm}'"
            )

        return np.pad(array=x, pad_width=pad_width, mode=mode)

    def _im2col(self, x: np.ndarray, x_shape: Shape) -> Tuple[np.ndarray, int, int]:
        """Performs im2col transformation for 2D convolution.

        Args:
            x: Input tensor (N, C, H, W).
            x_shape: Original shape of x.

        Returns:
            Tuple: (col matrix, output height, output width).
        """
        N, C, H, W = x_shape
        x_padded = self._add_padding(x)

        out_height, out_width = self._calc_out_size(H, W)

        # Use as_strided to create sliding windows
        shape = (N, C, out_height, out_width, self.KH, self.KW)
        sN, sC, sH, sW = x_padded.strides
        strides = (sN, sC, sH * self.sh, sW * self.sw, sH, sW)
        windows = as_strided(x=x_padded, shape=shape, strides=strides)

        # Reshape to column matrix (C*KH*KW, N*out_h*out_w)
        col = windows.transpose(1, 4, 5, 0, 2, 3).reshape(C * self.KH * self.KW, -1)
        return col, out_height, out_width

    def _col2im(self, col: np.ndarray, x_shape: Shape) -> np.ndarray:
        """Transforms column matrix back to padded image shape for gradient computation.

        Args:
            col: Gradient w.r.t. the col matrix, shape (C*KH*KW, N*out_h*out_w).
            x_shape: Original shape of the input (N, C, H, W).

        Returns:
            Gradient w.r.t. the original unpadded input (N, C, H, W).
        """
        N, C, H, W = x_shape

        H_p, W_p = H + 2 * self.ph, W + 2 * self.pw

        out_height, out_width = self._calc_out_size(H, W)
        k, i, j = self._build_inidices(x_shape=x_shape)

        cols_reshaped = col.reshape(
            C * self.KH * self.KW, N, out_height * out_width
        ).transpose(
            1, 0, 2
        )  # -> (N, C*KH*KW, out_h*out_w)
        grad_input = np.zeros((N, C, H_p, W_p), dtype=np.float32)

        # Efficiently sum contributions to the input gradient using np.add.at
        for n in range(N):
            np.add.at(grad_input[n], (k, i, j), cols_reshaped[n])

        # Remove padding
        if self.ph == 0 and self.pw == 0:
            return grad_input
        return grad_input[:, :, self.ph : self.ph + H, self.pw : self.pw + W]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 2D convolution.

        Args:
            x (np.ndarray): Input array with shape (N, C_in, H_in, W_in).

        Returns:
            np.ndarray: Output array with shape (N, C_out, H_out, W_out).
        """
        x = x.astype(np.float32, copy=False)
        N, _, _, _ = x.shape

        col, out_height, out_width = self._im2col(x=x, x_shape=x.shape)
        w_col = self.weight.data.reshape(self.out_channels, -1)

        # Convolution as Matrix Multiplication: (out_c, C*K*K) @ (C*K*K, N*out_h*out_w) -> (out_c, N*out_h*out_w)
        out = w_col @ col
        if self.bias is not None:
            out += self.bias.data

        # Reshape back to (N, C_out, H_out, W_out)
        out = out.reshape(self.out_channels, N, out_height, out_width).transpose(
            1, 0, 2, 3
        )

        self._cache["x_shape"] = x.shape
        self._cache["col"] = col
        self._cache["w_col"] = w_col

        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for 2D convolution.

        Args:
            grad (np.ndarray): Gradient w.r.t. the layer output, shape (N, C_out, H_out, W_out).

        Returns:
            np.ndarray: Gradient w.r.t. the layer input, shape (N, C_in, H_in, W_in).
        """
        grad = grad.astype(np.float32, copy=False)  # shape -> (N, C_out, H_out, W_out)
        x_shape = self._cache["x_shape"]
        col = self._cache["col"]
        w_col = self._cache["w_col"]

        # Reshape gradient for matrix multiplication: (out_c, N*H_out*W_out)
        grad_reshaped = grad.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)

        # Gradient w.r.t. Weights (kernel): (out_c, N*H_out*W_out) @ (N*H_out*W_out, C*KH*KW) -> (out_c, C*KH*KW)
        self.weight.grad = (grad_reshaped @ col.T).reshape(self.weight.data.shape)

        # Gradient w.r.t. Bias: sum over batch and spatial dimensions
        if self.bias is not None:
            self.bias.grad = np.sum(grad_reshaped, axis=1).reshape(-1, 1)

        # Gradient w.r.t. Col matrix: (C*KH*KW, out_c) @ (out_c, N*H_out*W_out) -> (C*KH*KW, N*H_out*W_out)
        grad_col = w_col.T @ grad_reshaped
        grad_input = self._col2im(grad_col, x_shape=x_shape)

        return grad_input

    def parameters(self) -> ListOfParameters:
        """Return a list of Parameters owned by this layer."""
        params: ListOfParameters = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        return f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={(self.KH, self.KW)}, stride={(self.sh, self.sw)}, padding={(self.ph, self.pw)}, bias={self.use_bias})"
