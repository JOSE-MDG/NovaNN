from __future__ import annotations
import itertools
from nova.utils.decorators.registry import registry_class
from nova.nn.parameter import is_lazy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor


@registry_class
class LazyModuleMixin:
    """Mixin class for modules with lazy parameter initialization.

    This mixin provides a standard interface for modules that defer parameter
    initialization until the first forward pass. This is particularly useful when
    building models where input dimensions are unknown at construction time.

    Modules that inherit from this mixin must implement the ``initialize_parameters``
    method to define how parameters should be initialized based on the input tensor.

    The lazy initialization pattern allows for more flexible model construction,
    as you don't need to know the exact shape of your data ahead of time. The
    parameters are automatically materialized with the correct shapes when the
    first batch of data passes through the module.

    Examples::

        >>> class MyLazyLayer(LazyModuleMixin, Module):
        ...     def __init__(self, out_features):
        ...         Module.__init__(self)
        ...         self.out_features = out_features
        ...         self.weight = UninitializedParameter()
        ...
        ...     def initialize_parameters(self, input):
        ...         with nova.no_grad():
        ...             self.in_features = input.shape[-1]
        ...             self.weight = self.weight.materialize(
        ...                 (self.out_features, self.in_features)
        ...             )
        ...
        >>> layer = MyLazyLayer(128)
        >>> print(layer.has_uninitialized_params())  # True
        >>> x = nova.randn(32, 64)
        >>> output = layer(x)  # Parameters initialized here
        >>> print(layer.has_uninitialized_params())  # False
        >>> print(layer.in_features)  # 64

    Note:
        The ``forward`` method is automatically overridden to check for uninitialized
        parameters and call ``initialize_parameters`` if needed before proceeding
        with the actual forward computation.
    """

    def initialize_parameters(self, input: Tensor) -> None:
        """Initialize parameters according to the input batch properties.

        This method provides an interface to isolate parameter initialization from
        the forward pass when doing parameter shape inference. It is called
        automatically during the first forward pass if the module has uninitialized
        parameters.

        Args:
            input: Input tensor used to infer parameter shapes

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Examples::

            >>> class MyLazyLayer(LazyModuleMixin, Module):
            ...     def initialize_parameters(self, input):
            ...         with nova.no_grad():
            ...             in_features = input.shape[-1]
            ...             self.weight = self.weight.materialize((self.out_features, in_features))
        """
        raise NotImplementedError(
            f"initialize_parameters is not implemented for {self.__class__.__name__}"
        )

    def has_uninitialized_params(self) -> bool:
        """Check if the module has any uninitialized parameters or buffers.

        Iterates through all parameters and buffers (non-recursively) to determine
        if any are still in an uninitialized state.

        Returns:
            ``True`` if any parameters or buffers are uninitialized, ``False`` otherwise

        Examples::

            >>> layer = LazyLinear(128)
            >>> print(layer.has_uninitialized_params())  # True
            >>> x = nova.randn(32, 64)
            >>> _ = layer(x)
            >>> print(layer.has_uninitialized_params())  # False
        """
        params = iter(self.parameters(recurse=False))
        buffers = iter(self.buffers(recurse=False))
        for param in itertools.chain(params, buffers):
            if is_lazy(param):
                return True

        return False

    def forward(self, input: Tensor) -> Tensor:
        """Performs the forward pass with automatic lazy initialization.

        If the module has uninitialized parameters, this method calls
        ``initialize_parameters`` to materialize them before proceeding with
        the actual forward computation defined in the parent class.

        Args:
            input: Input tensor

        Returns:
            Output tensor from the parent class's forward method

        Note:
            This method overrides the parent's forward to add lazy initialization
            logic. The actual computation is delegated to ``super().forward()``.
        """
        if self.has_uninitialized_params():
            self.initialize_parameters(input)
        return super().forward(input)
