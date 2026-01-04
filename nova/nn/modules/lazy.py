from __future__ import annotations
import itertools
from nova.utils.decorators.registry import registry_class
from nova.nn.parameter import is_lazy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor


@registry_class
class LazyModuleMixin:

    def initialize_parameters(self, input: Tensor) -> None:
        r"""Initialize parameters according to the input batch properties.

        This adds an interface to isolate parameter initialization from the
        forward pass when doing parameter shape inference.
        """
        raise NotImplementedError(
            f"initialize_parameters is not implemented for {self.__class__.__name__}"
        )

    def has_uninitialized_params(self) -> bool:
        params = iter(self.parameters(recurse=False))
        buffers = iter(self.buffers(recurse=False))
        for param in itertools.chain(params, buffers):
            if is_lazy(param):
                return True

        return False

    def forward(self, input: Tensor) -> Tensor:
        if self.has_uninitialized_params():
            self.initialize_parameters(input)
        return super().forward(input)
