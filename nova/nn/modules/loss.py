from __future__ import annotations
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import LossReducton


class MSELoss(Module):
    def __init__(
        self, reduction: LossReducton = "mean", weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, self.weight, self.reduction)

    def extra_repr(self) -> str:
        return "reduction={reduction}".format(**self.__dict__)


class L1Loss(Module):
    def __init__(
        self, reduction: LossReducton = "mean", weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, self.weight, self.reduction)

    def extra_repr(self) -> str:
        return "reduction={reduction}".format(**self.__dict__)


class SmoothL1Loss(Module):
    def __init__(
        self,
        reduction: LossReducton = "mean",
        beta: float = 0.1,
        weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.beta = beta
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> str:
        return F.smooth_l1_loss(input, target, self.beta, self.reduction, self.weight)

    def extra_repr(self) -> str:
        return "reduction={reduction}, beta={beta}".format(**self.__dict__)


class BCELoss(Module):
    def __init__(
        self,
        reduction: LossReducton = "mean",
        weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, self.weight, self.reduction)

    def extra_repr(self) -> str:
        return "reduction={reduction}".format(**self.__dict__)


class BCEWithLogitsLoss(Module):
    def __init__(
        self,
        reduction: LossReducton = "mean",
        weight: Optional[Tensor] = None,
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            input, target, self.weight, self.reduction, self.pos_weight
        )

    def extra_repr(self) -> str:
        return "reduction={reduction}".format(**self.__dict__)


class NLLLoss(Module):
    def __init__(
        self, reduction: LossReducton = "mean", weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input, target, self.weight, self.reduction)

    def extra_repr(self):
        return "reduction={reduction}".format(**self.__dict__)


class CrossEntropyLoss(Module):
    def __init__(
        self, reduction: LossReducton = "mean", weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, self.weight)

    def extra_repr(self):
        return "reduction={reduction}".format(**self.__dict__)


class KLDivLoss(Module):
    def __init__(self, reduction: LossReducton = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(
            input,
            target,
            reduction=self.reduction,
        )

    def extra_repr(self):
        return "reduction={reduction}".format(**self.__dict__)
