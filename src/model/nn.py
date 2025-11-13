import numpy as np
import inspect
import re
from src.module.layer import Layer
from src.layers.activations.activations import Activation
from src.layers.linear.linear import Linear
from src.core import config
from src.module.module import Parameters
from src.core.logger import logger

from typing import Iterable, Optional, Tuple, Callable, Any


class Sequential(Layer):
    """A sequential container for modules.

    Modules will be added to it in the order they are passed in the constructor.
    The forward pass will apply each module sequentially.
    """

    def __init__(self, *modules: Layer) -> None:
        super().__init__()
        self._layers = modules
        self._aply_initializer_for_linear_layers()

    def _is_activation(self, layer: Layer) -> bool:
        """Return True if `layer` is an activation layer."""
        return isinstance(layer, Activation)

    def _find_next_activation(
        self, start_idx: int
    ) -> Tuple[Optional[str], Optional[float]]:
        """Find the next activation after `start_idx` and return its init key (name) and param.

        Returns:
            Tuple of (init_key, activation_param) where either can be None.
        """
        for i in range(start_idx + 1, len(self._layers)):
            layer = self._layers[i]
            if self._is_activation(layer):
                key = layer.init_key
                if key is not None:
                    activation_params = getattr(layer, "activation_param", None)
                    logger.debug(f"Current activation is: '{key}'")
                    return key, activation_params
        return None, None

    def _find_last_activation(
        self, last_idx: int
    ) -> Tuple[Optional[str], Optional[float]]:
        """Find the last activation before `last_idx` and return its init key and param.

        Returns:
            Tuple of (init_key, activation_param) where either can be None.
        """
        for i in reversed(range(0, last_idx)):
            layer = self._layers[i]
            if self._is_activation(layer):
                key = layer.init_key
                if key is not None:
                    activation_params = getattr(layer, "activation_param", None)
                    logger.debug(f"Last activation was: {key}")
                    return key, activation_params
        return None, None

    def _aply_initializer_for_linear_layers(self) -> None:
        """Apply sensible default initializers to Linear layers based on nearby activations.

        Walks the contained layers and sets a seed initializer on Linear layers that
        do not already have one. Chooses initializers according to the next (or
        previous for final linear) activation's init key.
        """
        for idx, layer in enumerate(self._layers):
            if isinstance(layer, Linear):
                if getattr(layer, "init_fn", None) is not None:
                    continue

                init_key, activation_param = self._find_next_activation(idx)
                if init_key is not None:
                    init_fn_base = config.DEFAULT_NORMAL_INIT_MAP.get(
                        init_key, config.DEFAULT_NORMAL_INIT_MAP["default"]
                    )

                    if activation_param is not None:
                        init_fn = self._create_custom_init_fn(
                            init_fn_base, activation_param, init_key
                        )
                        logger.debug(
                            f"Initializing '{init_key}' is '{self.__get_lambda_name(init_fn)}' <- (with params)"
                        )
                    else:
                        init_fn = init_fn_base
                        logger.debug(
                            f"Initializing '{init_key}' with '{self.__get_lambda_name(init_fn)}' <- (without parameters)"
                        )

                elif layer == self._layers[-1]:

                    init_key, activation_param = self._find_last_activation(idx)

                    if init_key is not None:
                        init_fn_base = config.DEFAULT_NORMAL_INIT_MAP.get(
                            init_key, config.DEFAULT_NORMAL_INIT_MAP["default"]
                        )

                        if activation_param is not None:
                            init_fn = self._create_custom_init_fn(
                                init_fn_base, activation_param, init_key
                            )
                            # log the actual initializer selected (init_fn), not the base
                            logger.debug(
                                f"Prev initialization was '{init_key}', so the initialization is '{self.__get_lambda_name(init_fn)}' <- (with params)"
                            )
                        else:
                            init_fn = init_fn_base
                            logger.debug(
                                f"Prev initialization was '{init_key}', so the initialization is '{self.__get_lambda_name(init_fn)}' <- (without params)"
                            )
                else:
                    init_fn = config.DEFAULT_NORMAL_INIT_MAP["default"]
                    logger.debug(
                        f"Initializing by default '{self.__get_lambda_name(init_fn)}' <- (without activations)"
                    )

                layer.reset_parameters(init_fn)

    def __get_lambda_name(self, lambda_fn: Callable[..., Any]) -> str:
        """Try to infer a readable name for an initializer (works for simple lambdas)."""
        try:
            source_code = inspect.getsource(lambda_fn)
            match = re.search(r"lambda\s+shape\s*:\s*([a-zA-Z_]\w*)\s*\(", source_code)
            if match:
                return str(match.group(1))
            else:
                # fall back to callable __name__ (lambda -> '<lambda>')
                return f"({getattr(lambda_fn, '__name__', str(type(lambda_fn)))})"
        except Exception:
            return "unable to retrieve source"

    def _create_custom_init_fn(
        self,
        init_fn_base: Callable[[Tuple[int, int]], Any],
        a: float,
        nonlinearity: str,
    ) -> Callable[[Tuple[int, int]], Any]:
        """Wrap a base initializer to inject nonlinearity-specific parameters.

        Currently only special-cases leakyrelu (uses kaiming_normal_ with slope `a`).
        """
        from src.core.init import kaiming_normal_

        def custom_init(shape: Tuple[int, int]) -> Any:
            if nonlinearity == "leakyrelu":
                return kaiming_normal_(shape, a=a, nonlinearity=nonlinearity)
            else:
                return init_fn_base(shape)

        return custom_init

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Makes the class instance callable as a function, aliasing the forward pass.

        Args:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The output of the forward pass.
        """
        return self.forward(x)

    def train(self) -> None:
        """Sets all modules in the container to training mode."""
        for m in self._layers:
            m.train()

    def eval(self) -> None:
        """Sets all modules in the container to evaluation mode."""
        for m in self._layers:
            m.eval()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs a forward pass through the layers in order.

        Args:
            x (np.ndarray): The input tensor for the first layer.

        Returns:
            np.ndarray: The output tensor from the last layer.
        """
        out = x
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs a backward pass through the layers in reverse order.

        Args:
            grad (np.ndarray): The gradient of the loss with respect to the
                               output of the final layer.

        Returns:
            np.ndarray: The gradient with respect to the input of the first layer.
        """
        grad_input = grad
        for layer in reversed(self._layers):
            grad_input = layer.backward(grad_input)
        return grad_input

    def parameters(self) -> Iterable[Parameters]:
        """Gathers and returns all parameters from the layers in the container.

        It iterates through each layer and collects its parameters.

        Returns:
            Iterable[Parameters]: An iterable of all parameters in the model.
        """
        parameters = []
        for layer in self._layers:
            parameters.extend(layer.parameters())
        return parameters

    def zero_grad(self) -> None:
        """Zero gradients for all submodules' parameters."""
        for layer in self._layers:
            layer.zero_grad()
