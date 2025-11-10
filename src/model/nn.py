import numpy as np
import inspect
import re
from src.module.layer import Layer
from src.layers.activations.activations import Activation
from src.layers.linear.linear import Linear
from src.core import config
from src.module.module import Parameters
from src.core.logger import logger


from typing import Iterable


class Sequential(Layer):
    """A sequential container for modules.

    Modules will be added to it in the order they are passed in the constructor.
    The forward pass will apply each module sequentially.
    """

    def __init__(self, *modules: Layer):
        super().__init__()
        self._layers = modules
        self._aply_initializer_for_linear_layers()

    def _is_activation(self, layer: Activation):
        return isinstance(layer, Activation)

    def _find_next_activation(self, start_idx: int):
        for i in range(start_idx + 1, len(self._layers)):
            layer = self._layers[i]
            if self._is_activation(layer):
                key = layer.get_init_key()
                if key is not None:
                    activation_params = getattr(layer, "activation_param", None)
                    logger.debug(f"Current activation is: '{key}'")
                    return key, activation_params
        return None, None

    def _find_last_activation(self, last_idx: int):
        for i in reversed(range(0, last_idx)):
            layer = self._layers[i]
            if self._is_activation(layer):
                key = layer.get_init_key()
                if key is not None:
                    activation_params = getattr(layer, "activation_param", None)
                    logger.debug(f"Last activation was: {key}")
                    return key, activation_params
        return None, None

    def _aply_initializer_for_linear_layers(self):
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
                            logger.debug(
                                f"Prev Initialization was '{init_key}' so the initialization is {self.__get_lambda_name(init_fn_base)} <- (with params)"
                            )
                        else:
                            init_fn = init_fn_base
                            logger.debug(
                                f"Prev Initialization was '{init_key}' so the initialization is {self.__get_lambda_name(init_fn_base)} <- (without params)"
                            )
                else:
                    init_fn = config.DEFAULT_NORMAL_INIT_MAP["default"]
                    logger.debug(
                        f"Initializing by default '{self.__get_lambda_name(init_fn)}' <- (without activations)"
                    )

                layer.reset_parameters(init_fn)

    def __get_lambda_name(self, lambda_fn):
        try:
            source_code = inspect.getsource(lambda_fn)
            match = re.search(r"lambda\s+shape\s*:\s*([a-zA-Z_]\w*)\s*\(", source_code)
            if match:
                return str(match.group(1))
            else:
                return f"({lambda_fn.__name__})"
        except:
            return f"The resource could not be found."

    def _create_custom_init_fn(self, init_fn_base, a, nonlinearity):
        from src.core.init import kaiming_normal_

        def custom_init(shape):
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

    def train(self):
        """Sets all modules in the container to training mode."""
        for m in self._layers:
            m.train()

    def eval(self):
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
        input = x
        # Apply each layer sequentially
        for layer in self._layers:
            input = layer.forward(input)
        return input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs a backward pass through the layers in reverse order.

        Args:
            grad (np.ndarray): The gradient of the loss with respect to the
                               output of the final layer.

        Returns:
            np.ndarray: The gradient with respect to the input of the first layer.
        """
        grad_input = grad
        # Apply backpropagation through each layer in reverse order
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

    def zero_grad(self):
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params
