from __future__ import annotations
import nova
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module
from nova.nn.parameter import Parameter, Buffer

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


class BatchNorm1d(Module):
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
            self.running_mean: Buffer = Buffer(nova.empty((num_features,), dtype=dtype))
            self.running_var: Buffer = Buffer(nova.empty((num_features,), dtype=dtype))
            self.num_batches_tracked: Buffer = Buffer(nova.empty((), dtype=nova.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        if self.affine:
            self.weight: Parameter = Parameter(nova.empty((num_features,), dtype=dtype))
            self.bias: Parameter = Parameter(nova.empty((num_features,), dtype=dtype))
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
        exp_avg_factor = 0.0

        if self._training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exp_avg_factor = 1.0 / self.num_batches_tracked.to(nova.double)
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


class BatchNorm2d(Module):
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
            self.running_mean: Buffer = Buffer(nova.empty((num_features,), dtype=dtype))
            self.running_var: Buffer = Buffer(nova.empty((num_features,), dtype=dtype))
            self.num_batches_tracked: Buffer = Buffer(nova.empty(0, dtype=nova.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        if self.affine:
            self.weight: Parameter = Parameter(nova.empty((num_features,), dtype=dtype))
            self.bias: Parameter = Parameter(nova.empty((num_features,), dtype=dtype))
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
        exp_avg_factor = 0.0

        if self._training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exp_avg_factor = 1.0 / self.num_batches_tracked.to(nova.double)
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


class BatchNorm3d(Module):
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
            self.running_mean: Buffer = Buffer(nova.empty((num_features,), dtype=dtype))
            self.running_var: Buffer = Buffer(nova.empty((num_features,), dtype=dtype))
            self.num_batches_tracked: Buffer = Buffer(nova.empty(0, dtype=nova.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        if self.affine:
            self.weight: Parameter = Parameter(nova.empty((num_features,), dtype=dtype))
            self.bias: Parameter = Parameter(nova.empty((num_features,), dtype=dtype))
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
        exp_avg_factor = 0.0

        if self._training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exp_avg_factor = 1.0 / self.num_batches_tracked.to(nova.double)
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
