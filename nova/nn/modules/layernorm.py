from __future__ import annotations
import nova
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Size, Dtype


class LayerNorm(Module):

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        normalized_shape: Size,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(nova.empty((normalized_shape,)), dtype=dtype)
            self.bias = Parameter(nova.empty((normalized_shape,)), dtype=dtype)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

    def extra_repr(self) -> str:
        return "normalized_shape={normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )
