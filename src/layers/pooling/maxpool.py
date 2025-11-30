import numpy as np
from numpy.lib.stride_tricks import as_strided
from src._typing import IntOrPair
from typing import Optional, Tuple
from src.module import Layer


class MaxPool1d(Layer):
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.K: int = int(kernel_size)
        self.stride: int = int(stride) if stride is not None else self.K
        self.padding: int = int(padding)
        self.pm: str = padding_mode
        self._cache: dict = {}

    def _add_padding(self, x: np.ndarray) -> np.ndarray:
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

    def _calc_out_length(self, L: int) -> int:
        return (L + 2 * self.padding - self.K) // self.stride + 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)  # (N, C, L)
        N, C, L = x.shape
        x_p = self._add_padding(x)
        L_out = self._calc_out_length(L)
        sN, sC, sL = x_p.strides
        shape = (N, C, L_out, self.K)
        strides = (sN, sC, sL * self.stride, sL)
        windows = as_strided(x_p, shape=shape, strides=strides)  # (N,C,L_out,K)
        out = windows.max(axis=3)  # (N,C,L_out)
        self._cache = {
            "windows": windows,
            "x_shape": x.shape,
            "padded_shape": x_p.shape,
            "L_out": L_out,
        }
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_output = grad_output.astype(np.float32, copy=False)  # (N,C,L_out)

        windows = self._cache["windows"]  # (N,C,L_out,K)
        L_out = self._cache["L_out"]
        _, _, L = self._cache["x_shape"]
        padded_shape = self._cache["padded_shape"]

        max_vals = windows.max(axis=3, keepdims=True)  # (N,C,L_out,1)
        mask = windows == max_vals  # bool
        counts = mask.sum(axis=3, keepdims=True).astype(np.float32)
        counts[counts == 0] = 1.0

        go = grad_output[..., np.newaxis]  # (N,C,L_out,1)
        grad_windows = mask.astype(np.float32) * (go / counts)  # (N,C,L_out,K)
        grad_padded = np.zeros(padded_shape, dtype=np.float32)

        for i in range(L_out):
            start = i * self.stride
            end = start + self.K
            grad_padded[:, :, start:end] += grad_windows[:, :, i, :]

        if self.padding == 0:
            return grad_padded
        return grad_padded[:, :, self.padding : self.padding + L]


class MaxPool2d(Layer):
    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair | None] = None,
        padding: IntOrPair = 0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.KH, self.KW = self._pair(kernel_size)
        if stride is None:
            self.sh, self.sw = self.KH, self.KW
        self.sh, self.sw = self._pair(stride)
        self.ph, self.pw = self._pair(padding)
        self.pm = padding_mode
        self._cache = {}

    def _pair(self, x: IntOrPair | str) -> Tuple[int, int]:
        if isinstance(x, int):
            return (x, x)
        if isinstance(x, str):
            if x == "valid":
                return (0, 0)
            if x == "same":
                raise ValueError(f"The 'same' value is not currently supported")
            else:
                raise ValueError(f"Unsopported value '{x}'")
        return tuple(x)

    def _add_padding(self, x: np.ndarray) -> np.ndarray:
        pad_width = ((0, 0), (0, 0), (self.ph, self.ph), (self.pw, self.pw))
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

    def calculate_output_size(self, height: int, width: int) -> Tuple[int, int]:
        out_height = (height + 2 * self.ph - self.KH) // self.sh + 1
        out_width = (width + 2 * self.pw - self.KW) // self.sw + 1
        return out_height, out_width

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        N, C, H, W = x.shape
        x_padded = self._add_padding(x)

        out_h, out_w = self.calculate_output_size(H, W)

        sN, sC, sH, sW = x_padded.strides
        shape = (N, C, out_h, out_w, self.KH, self.KW)
        strides = (sN, sC, sH * self.sh, sW * self.sw, sH, sW)
        window = as_strided(x_padded, shape=shape, strides=strides)

        out = window.max(axis=(4, 5))

        self._cache["windows"] = window

        self._cache["x_shape"] = x.shape
        self._cache["padded_shape"] = x_padded.shape
        self._cache["out_h"] = out_h
        self._cache["out_w"] = out_w

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # grad_output -> shape (N,C,out_h, out_w)
        grad_output = grad_output.astype(np.float32, copy=False)
        window = self._cache["windows"]
        x_shape = self._cache["x_shape"]
        padded_shape = self._cache["padded_shape"]
        out_h = self._cache["out_h"]
        out_w = self._cache["out_w"]
        _, _, H, W = x_shape

        max_vals = window.max(axis=(4, 5), keepdims=True)  # (N,C,out_h,out_w,1,1)
        mask = (window == max_vals).astype(np.float32)
        counts = mask.sum(axis=(4, 5), keepdims=True)
        counts[counts == 0] = 1.0

        grad_output = grad_output[:, :, :, :, np.newaxis, np.newaxis]
        grad_windows = mask * (grad_output / counts)
        grad_input_padded = np.zeros(padded_shape, dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.sh
                h_end = h_start + self.sh
                w_start = j + self.sw
                w_end = w_start + self.sw

                grad_input_padded[:, :, h_start:h_end, w_start:w_end] += grad_windows[
                    :, :, i, j, :, :
                ]

        if self.ph == 0 and self.pw == 0:
            grad_input = grad_input_padded
        else:
            grad_input = grad_input_padded[
                :, :, self.ph : self.ph + H, self.pw : self.pw + W
            ]
        self._cache.clear()
        return grad_input
