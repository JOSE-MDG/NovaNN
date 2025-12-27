from __future__ import annotations
import nova
import nova.nn.init as init
from typing import TYPE_CHECKING
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor


class Flatten(Module): ...
