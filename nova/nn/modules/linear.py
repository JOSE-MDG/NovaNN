from __future__ import annotations
import nova
import nova.nn.init as init
from typing import TYPE_CHECKING
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight: Parameter = Parameter(nova.rand(out_features, in_features))
        if bias:
            self.bias: Parameter = Parameter(nova.zeros((1, out_features)))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.use_bias:
            out = out + self.bias

        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"
