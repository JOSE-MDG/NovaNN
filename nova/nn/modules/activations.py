from __future__ import annotations
import nova
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype, Dim


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope)


class GeLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


class PReLU(Module):
    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        *,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()

        self.weight: Parameter = Parameter(
            nova.tensor(num_parameters, dtype=dtype).fill_(init)
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, self.weight)


class Tanh(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


class Softmax(Module):
    def __inti__(self, dim: Dim = 1) -> None:
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, dim=self.dim)


class LogSoftmax(Module):
    def __init__(self, dim: Dim = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, dim=self.dim)
