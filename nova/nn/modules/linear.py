from __future__ import annotations
import nova
import math
import nova.nn.init as init
import nova.nn.functional as F
from nova.nn.parameter import UninitializedParameter
from nova.nn.modules.lazy import LazyModuleMixin
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype


class Linear(Module):

    weight: Parameter
    bias: Optional[Parameter]

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

        self.weight = Parameter(nova.empty((out_features, in_features), dtype=dtype))
        if bias:
            self.bias = Parameter(nova.empty((1, out_features), dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = init.get_fans(self.weight, mode="fan_in")
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class LazyLinear(LazyModuleMixin, Linear):

    weight: UninitializedParameter
    bias: Optional[UninitializedParameter]

    def __init__(
        self, out_features: int, bias: bool = True, dtype: Optional[Dtype] = None
    ):
        Module.__init__(self)
        self.out_features = out_features
        self.use_bias = bias
        self.dtype = dtype

        self.weight = UninitializedParameter()
        if bias:
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        if not self.has_uninitialized_params():
            super().reset_parameters()

    def initialize_parameters(self, input: Tensor) -> None:
        """
        Infers ``in_features`` based on ``input`` and initializes parameters.
        """
        if self.has_uninitialized_params():
            with nova.no_grad():
                self.in_features = input.shape[-1]
                self.weight = self.weight.materialize(
                    (self.out_features, self.in_features), dtype=self.dtype
                )
                if self.use_bias:
                    self.bias = self.bias.materialize(
                        (1, self.out_features), dtype=self.dtype
                    )
                self.reset_parameters()
