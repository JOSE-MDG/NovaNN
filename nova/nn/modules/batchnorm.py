from __future__ import annotations
import nova
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module, LazyModuleMixin
from nova.nn.parameter import (
    Parameter,
    Buffer,
    UninitializedParameter,
    UninitializedBuffer,
)

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype


"""
Batch Normalization Behavior Table:

| training | track_running_stats | Result | Layer Behavior                                         |
|:---------|:--------------------|:-------|:-------------------------------------------------------|
| True     | True                | True   | Trains and updates running statistics (memory).        |
| True     | False               | True   | Trains using only the current batch (no memory).       |
| False    | False               | True   | Evaluates using current batch (because no memory).     |
| False    | True                | False  | Evaluates using saved memory (Standard production mode).|
"""

"""
Batch Normalization Usage Cheat Sheet:

| Parameter                 | When to change it           | Technical Reason                                         |
|:--------------------------|:----------------------------|:---------------------------------------------------------|
| momentum low (0.01)       | Noisy dataset or Fine-tuning| You want "memory" to be very stable and change slowly.   |
| momentum high (0.5)       | Clean/homogeneous dataset   | You want the model to adapt very quickly to the data.    |
| momentum = None           | Very small batches          | Uses num_batches_tracked to average history equally.     |
| track_running_stats=False | Siamese Nets / Meta-learning| Identical normalization in train/eval (input-based only).|
"""


class _BatchNorm(Module):

    running_mean: Buffer
    running_var: Buffer
    num_batches_tracked: Buffer

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        num_features: int,
        momentum: Optional[float] = 0.1,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()

        self.track_running_stats = track_running_stats
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        if self.track_running_stats:
            self.running_mean = Buffer(nova.empty((num_features,), dtype=dtype))
            self.running_var = Buffer(nova.empty((num_features,)), dtype=dtype)
            self.num_batches_tracked = Buffer(nova.empty(()), dtype=nova.long)
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        if self.affine:
            self.weight = Parameter(nova.empty((num_features,)), dtype=dtype)
            self.bias = Parameter(nova.empty((num_features,)), dtype=dtype)
            self.weight.is_bn_param = True
            self.bias.is_bn_param = True
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:

        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.ones_()
            self.num_batches_tracked.zero_()

        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:

        self._check_input_dim(input)

        exp_avg_factor = 0.0

        if self._training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exp_avg_factor = 1.0 / float(self.num_batches_tracked.item())
                else:
                    exp_avg_factor = self.momentum

        return F.batch_norm(
            input=input,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=self._training or not self.track_running_stats,
            momentum=exp_avg_factor,
            eps=self.eps,
        )

    def _check_input_dim(self, input: Tensor) -> None | Exception:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return "{num_features}, momentum={momentum}, eps={eps}, affine={affine}, track_running_stats={track_running_stats}".format(
            **self.__dict__
        )


class _LayzyNormBase(LazyModuleMixin, _BatchNorm):

    running_mean: UninitializedBuffer
    running_var: UninitializedBuffer
    num_batches_tracked: UninitializedBuffer

    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(
        self,
        momentum=0.1,
        eps=0.00001,
        affine=True,
        track_running_stats=True,
        dtype=None,
    ):
        Module.__init__(self)

        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.affine = affine
        self.dtype = dtype
        self.eps = eps

        if self.track_running_stats:
            self.running_mean = UninitializedBuffer()
            self.running_var = UninitializedBuffer()
            self.num_batches_tracked = UninitializedBuffer()
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        if self.affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
            self.weight.is_bn_param = True
            self.bias.is_bn_param = True
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        if not self.has_uninitialized_params():
            super().reset_parameters()

    def initialize_parameters(self, input: Tensor) -> None:
        """
        Infers ``num_features`` based on ``input`` and initializes parameters.
        """
        if self.has_uninitialized_params():

            self.num_features = input.shape[1]
            if self.track_running_stats:

                self.running_mean = self.running_mean.materialize(
                    (self.num_features,), dtype=self.dtype
                )
                self.running_var = self.running_var.materialize(
                    (self.num_features,), dtype=self.dtype
                )
                self.num_batches_tracked = self.num_batches_tracked.materialize(
                    (self.num_features,), dtype=nova.long
                )

            if self.affine:
                self.weight = self.weight.materialize(
                    (self.num_features,), dtype=self.dtype
                )
                self.bias = self.bias.materialize(
                    (self.num_features,), dtype=self.dtype
                )

            self.reset_parameters()


class BatchNorm1d(_BatchNorm):
    """Applies Batch Normalization over a 2D or 3D input"""

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class LazyBatchNorm1d(_LayzyNormBase):

    def _check_input_dim(self, input) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class BatchNorm2d(_BatchNorm):
    """Applies Batch Normalization over a 4D input"""

    def _check_input_dim(self, input) -> None:
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class LazyBatchNorm2d(_LayzyNormBase):

    def _check_input_dim(self, input) -> None:
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class BatchNorm3d(_BatchNorm):
    """Applies Batch Normalization over a 5D input"""

    def _check_input_dim(self, input) -> None:
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")


class LazyBatchNorm3d(_LayzyNormBase):

    def _check_input_dim(self, input) -> None:
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")
