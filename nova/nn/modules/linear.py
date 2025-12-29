from __future__ import annotations
import nova
import math
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight: Parameter = Parameter(
            nova.empty((out_features, in_features), dtype=dtype)
        )
        if bias:
            self.bias: Parameter = Parameter(nova.empty((1, out_features), dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = init.get_fans(self.weight.size, mode="fan_in")
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"
