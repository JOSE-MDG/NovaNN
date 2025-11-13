from typing import Optional

from src.module.layer import Layer


class Activation(Layer):
    """Base class for activation layers.

    This class centralizes common behaviour for activation layers used by the
    project (e.g. relu, tanh, sigmoid). It provides a stable `name` used as an
    initialization key and a flag (`affect_init`) to indicate whether the
    activation should influence default weight initialization.

    Attributes:
        name (str): Lowercased class name used as an identifier for init maps.
        affect_init (bool): If True, the activation influences weight init.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name: str = self.__class__.__name__.lower()
        self.affect_init: bool = True

    def get_init_key(self) -> Optional[str]:
        """Return the initialization key for this activation.

        Returns:
            The activation identifier (lowercased class name) if the activation
            should affect weight initialization; otherwise, None.
        """
        if not self.affect_init:
            return None
        return self.name

    @property
    def init_key(self) -> Optional[str]:
        """Convenience alias for `get_init_key()`."""
        return self.get_init_key()
