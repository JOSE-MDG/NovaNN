import numpy as np
import inspect
import re
from novann.layers.activations import Activation
from novann.layers import Linear, Conv2d, Conv1d
from novann.core import config
from novann.module import Parameters, Layer
from novann.core.logger import logger

from novann._typing import InitFn, ActivAndParams
from typing import Iterable, Tuple, Any


class Sequential(Layer):
    """A sequential container for neural network modules.

    Modules are added in the order they are passed to the constructor.
    The forward pass applies each module sequentially, and the backward
    pass applies the gradient in reverse order.

    This class attempts to automatically apply appropriate weight initializations
    to Linear and Conv layers based on the nearest activation function.

    Args:
        *modules (Layer): Variable-length list of Layer instances to chain
                          sequentially.
    """

    def __init__(self, *modules: Layer) -> None:
        """Initializes the Sequential module and applies automatic weight initialization."""
        super().__init__()
        self._layers = modules
        self._aply_initializer_for_linear_layers()

    def _is_activation(self, layer: Layer) -> bool:
        """Returns True if `layer` is an activation layer."""
        return isinstance(layer, Activation)

    def _find_next_activation(self, start_idx: int) -> ActivAndParams:
        """Finds the next activation after `start_idx`.

        Used to determine the initialization key and parameter required for
        Linear/Conv layers.

        Returns:
            Tuple[str | None, float | None]: The initialization key (name)
                                             and its parameter (e.g., LeakyReLU slope).
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

    def _find_last_activation(self, last_idx: int) -> ActivAndParams:
        """Finds the last activation before `last_idx`.

        Used to initialize the final Linear/Conv layer based on the preceding activation.

        Returns:
            Tuple[str | None, float | None]: The initialization key (name)
                                             and its parameter.
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

    def _is_initializable(self, layer: Layer) -> bool:
        """Returns True if the layer is a weighted layer (Linear, Conv) that requires initialization."""
        instances = (Linear, Conv1d, Conv2d)
        if isinstance(layer, instances):
            return True

    def _aply_initializer_for_linear_layers(self) -> None:
        """Applies default initializers to Linear/Conv layers based on nearby activations.

        Searches for the next activation (or the last one for the final layer) to
        select the appropriate initialization function (e.g., Kaiming for ReLU).
        """
        for idx, layer in enumerate(self._layers):
            if self._is_initializable(layer=layer):
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

    def __get_lambda_name(self, lambda_fn: InitFn) -> str:
        """Attempts to infer a readable name for an initializer function for logging purposes."""
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
        init_fn_base: InitFn,
        a: float,
        nonlinearity: str,
    ) -> InitFn:
        """Wraps a base initializer to inject non-linearity specific parameters (e.g., slope 'a' for LeakyReLU)."""
        from novann.core.init import kaiming_normal_

        def custom_init(shape: Tuple[int, int]) -> Any:
            if nonlinearity == "leakyrelu":
                return kaiming_normal_(shape, a=a, nonlinearity=nonlinearity)
            else:
                return init_fn_base(shape)

        return custom_init

    def train(self) -> None:
        """Sets all contained modules in the container to training mode."""
        for m in self._layers:
            m.train()

    def eval(self) -> None:
        """Sets all contained modules in the container to evaluation mode."""
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
            out = layer(out)
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

        Returns:
            Iterable[Parameters]: An iterable of all trainable parameters in the model.
        """
        parameters = []
        for layer in self._layers:
            parameters.extend(layer.parameters())
        return parameters

    def zero_grad(self) -> None:
        """Zeros the gradients for all submodules' parameters."""
        for layer in self._layers:
            layer.zero_grad()

    def __repr__(self):
        lines = []
        lines.append(f"{self.__class__.__name__}(")

        for i, layer in enumerate(self._layers):
            layer_repr = repr(layer)
            layer_repr = self._add_indent(layer_repr, 2)
            lines.append(f"  ({i}): {layer_repr}")

        lines.append(")")
        return "\n".join(lines)

    def _add_indent(self, s_, numSpaces):
        """Auxiliary method to make it look nice if there are nested layers"""
        s = s_.split("\n")
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        return first + "\n" + s
