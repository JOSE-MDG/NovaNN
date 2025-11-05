import numpy as np
from src.module.layer import Layer
from src.module.module import Parameters
from typing import List, Iterable


class Sequential(Layer):
    """A sequential container for modules.

    Modules will be added to it in the order they are passed in the constructor.
    The forward pass will apply each module sequentially.
    """

    def __init__(self, modules: List[Layer]):
        """Initializes the Sequential container.

        Args:
            modules (List[Layer]): An ordered list of layers/modules to be
                                   executed in sequence.
        """
        super().__init__()
        self._layers = modules

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
        for m in self.layers:
            m.train()

    def eval(self):
        """Sets all modules in the container to evaluation mode."""
        for m in self.layers:
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

        It iterates through each layer and collects its parameters if the layer
        is marked as having them.

        Returns:
            Iterable[Parameters]: An iterable of all parameters in the model.
        """
        parameters = []
        for layer in self.layers:
            # Check if the layer has parameters to contribute
            if layer.__getattribute__("_have_params"):
                parameters.extend(list(layer.parameters()))
        return parameters

    def zero_grad(self):
        """Resets the gradients of all parameters in the model to zero."""
        for layer in self._layers:
            # It's good practice to let the layer handle its own parameters.
            # This check ensures we only call zero_grad on layers that have it implemented
            # and are expected to have parameters.
            if layer.__getattribute__("_have_params"):
                layer.zero_grad()
