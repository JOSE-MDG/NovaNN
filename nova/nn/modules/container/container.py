from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Type
from nova.nn.modules import Module

if TYPE_CHECKING:
    from nova import Tensor


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for module in modules:
            if not isinstance(module, Module):
                raise ValueError(
                    f"Only Module types can be registered in the sequential container, got {type(module)}"
                )

            self.register_module(module.__class__.__name__, module)

    def forward(self, x: Tensor):
        for module in self._modules.values():
            x = module(x)
        return x
