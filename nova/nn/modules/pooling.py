from __future__ import annotations
import nova
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor


class AdaptiveAvgPool1d(Module): ...


class AdaptiveAvgPool2d(Module): ...


class AdaptiveAvgPool3d(Module): ...


class AvgPool1d(Module): ...


class AvgPool2d(Module): ...


class AvgPool3d(Module): ...


class MaxPool1d(Module): ...


class MaxPool2d(Module): ...


class MaxPool3d(Module): ...
