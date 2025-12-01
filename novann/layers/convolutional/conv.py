import numpy as np
from numpy.lib.stride_tricks import as_strided
from novann.module import Layer, Parameters
from typing import Optional, Tuple
from novann._typing import InitFn, Shape, ListOfParameters, IntOrPair
from novann.core import DEFAULT_UNIFORM_INIT_MAP


class Conv1d(Layer):
    """Applies a 1D convolution over an input signal composed of several input planes.

    The input must have shape (N, C_in, L_in).

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1.
        padding (int): Zero-padding added to both sides of the input. Default: 0.
        bias (bool): If True, adds a learnable bias to the output. Default: True.
        padding_mode (str): 'zeros', 'reflect', 'replicate', or 'circular'. Default: 'zeros'.
        initializer (InitFn, optional): Weight initialization function. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        padding_mode: str = "zeros",
        initializer: InitFn = None,
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.K: int = kernel_size
        self.init_fn: InitFn = initializer
        self.stride: int = stride
        self.padding: int = padding
        self.pm: str = padding_mode

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
        self.weight.name = "kernel1d"
        if self.use_bias:
            # Shape: (out_channels, 1)
            self.bias = Parameters(np.zeros((self.out_channels, 1), dtype=np.float32))
            self.bias.name = "conv1d bias"
        else:
            self.bias = None

    def _calc_out_size(self, L: int) -> int:
        """Calculates the output length L_out."""
        L_out = (L + 2 * self.padding - self.K) // self.stride + 1
        return L_out

    def _add_padding(self, x: np.ndarray) -> np.ndarray:
        """Applies padding to the input tensor."""
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding))
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

    def _im2col(self, x: np.ndarray, x_shape: Shape) -> tuple[np.ndarray, int]:
        """Performs im2col transformation for 1D convolution.

        Args:
            x: Input tensor (N, C, L).
            x_shape: Original shape of x.

        Returns:
            Tuple: (col matrix, output length L_out).
        """
        N, C, L = x_shape
        x_p = self._add_padding(x)
        L_out = self._calc_out_size(L)

        self._cache["x_p_shape"] = x_p.shape

        # Use as_strided to create sliding windows
        shape = (N, C, L_out, self.K)
        sN, sC, sL = x_p.strides
        strides = (sN, sC, sL * self.stride, sL)
        windows = as_strided(x_p, shape=shape, strides=strides)

        # Reshape to column matrix (C*K, N*L_out)
        col = windows.transpose(1, 3, 0, 2).reshape(C * self.K, -1)
        return col, L_out

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
        N, _, _ = x.shape
        col, L_out = self._im2col(x, x.shape)
        w_col = self.weight.data.reshape(self.out_channels, -1)

        # Convolution as Matrix Multiplication: (out_c, C*K) @ (C*K, N*L_out) -> (out_c, N*L_out)
        out = w_col @ col
        if self.bias is not None:
            out += self.bias.data

        # Reshape back to (N, C_out, L_out)
        out = out.reshape(self.out_channels, N, L_out).transpose(1, 0, 2)

        self._cache["col"] = col
        self._cache["w_col"] = w_col
        self._cache["x_shape"] = x.shape
        self._cache["L_out"] = L_out

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

        self._cache.clear()
        return grad_input

    def parameters(self) -> ListOfParameters:
        """Return a list of Parameters owned by this layer."""
        params: ListOfParameters = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Conv2d(Layer):
    """Applies a 2D convolution over an input signal composed of several input planes.

    The input must have shape (N, C_in, H_in, W_in).

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (IntOrPair): Size of the convolving kernel (H, W).
        stride (IntOrPair): Stride of the convolution (H, W). Default: 1.
        padding (IntOrPair | str): Zero-padding added to both sides (H, W) or 'valid'. Default: 0.
        bias (bool): If True, adds a learnable bias to the output. Default: True.
        padding_mode (str): 'zeros', 'reflect', 'replicate', or 'circular'. Default: 'zeros'.
        initializer (InitFn, optional): Weight initialization function. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair] = 1,
        padding: IntOrPair | str = 0,
        bias: bool = True,
        padding_mode: str = "zeros",
        initializer: InitFn = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KH, self.KW = self._pair(kernel_size)

        # Corregir lógica de stride: si es None, debe usar el default (1) o el kernel_size (si fuera pooling),
        # pero para conv, el default es 1. La línea original estaba asignando al KH/KW en su lugar.
        # Si stride es 1, self._pair(1) retorna (1, 1). Si es None, lo manejamos como 1 por defecto.
        stride_val = stride if stride is not None else 1
        self.sh, self.sw = self._pair(stride_val)

        self.ph, self.pw = self._pair(padding)
        self.pm: str = padding_mode

        self.init_fn = initializer
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
            # Note: Assuming 'relu' is a key in DEFAULT_UNIFORM_INIT_MAP
            init = DEFAULT_UNIFORM_INIT_MAP["relu"]
        # Shape: (out_channels, in_channels, KH, KW)
        w = init((self.out_channels, self.in_channels, self.KH, self.KW))
        self.weight = Parameters(np.asarray(w))
        self.weight.name = "kernel2d"
        if self.use_bias:
            # Shape: (out_channels, 1) - Note: reshaped during forward/backward
            self.bias = Parameters(np.zeros((self.out_channels, 1), dtype=np.float32))
            self.bias.name = "conv2d bias"
        else:
            self.bias = None

    def _pair(self, x: IntOrPair | str) -> Tuple[int, int]:
        """Converts integer, tuple, or valid/same string to a (H, W) pair."""
        if isinstance(x, int):
            return (x, x)
        if isinstance(x, str):
            if x == "valid":
                return (0, 0)
            if x == "same":
                raise ValueError(f"The 'same' value is not currently supported")
            else:
                # Corregir error ortográfico en el mensaje de error: 'Unsopported' -> 'Unsupported'
                raise ValueError(f"Unsupported value '{x}'")
        # Assuming tuple input (IntOrPair)
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

        # Convolution as Matrix Multiplication: (out_c, C*K*K) @ (C*K*K, N*L_out) -> (out_c, N*L_out)
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
        grad = grad.astype(np.float32, copy=False)
        x_shape = self._cache["x_shape"]
        col = self._cache["col"]
        w_col = self._cache["w_col"]

        # Reshape gradient for matrix multiplication: (out_c, N*L_out)
        grad_reshaped = grad.reshape(self.out_channels, -1)

        # Gradient w.r.t. Weights (kernel): (out_c, N*L_out) @ (N*L_out, C*K*K) -> (out_c, C*K*K)
        self.weight.grad = (grad_reshaped @ col.T).reshape(self.weight.data.shape)

        # Gradient w.r.t. Bias: sum over batch and spatial dimensions
        if self.bias is not None:
            self.bias.grad = np.sum(grad_reshaped, axis=1).reshape(-1, 1)

        # Gradient w.r.t. Col matrix: (C*K*K, out_c) @ (out_c, N*L_out) -> (C*K*K, N*L_out)
        grad_col = w_col.T @ grad_reshaped
        grad_input = self._col2im(grad_col, x_shape=x_shape)

        self._cache.clear()
        return grad_input

    def parameters(self) -> ListOfParameters:
        """Return a list of Parameters owned by this layer."""
        params: ListOfParameters = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
