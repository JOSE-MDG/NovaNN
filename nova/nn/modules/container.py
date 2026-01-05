from __future__ import annotations
from typing import TYPE_CHECKING
from nova.nn.modules import Module

if TYPE_CHECKING:
    from nova import Tensor


class Sequential(Module):
    """A sequential container for organizing modules in order.

    Modules will be added to it in the order they are passed in the constructor.
    The ``forward()`` method of ``Sequential`` accepts any input and forwards it
    to the first module it contains. It then chains the outputs sequentially to
    the inputs of the subsequent modules, finally returning the output of the
    last module.

    The value a ``Sequential`` provides over manually calling a sequence of modules
    is that it allows treating the whole container as a single module, such that
    performing a transformation on the ``Sequential`` applies to each of the modules
    it stores (which are each a registered submodule of the ``Sequential``).

    Args:
        *modules: Variable number of Module instances to be added to the container

    Raises:
        ValueError: If any argument is not an instance of Module

    Examples::

        >>> # Using Sequential to create a simple MLP
        >>> model = Sequential(
        ...     Linear(784, 256),
        ...     ReLU(),
        ...     Linear(256, 128),
        ...     ReLU(),
        ...     Linear(128, 10)
        ... )
        >>> x = nova.randn(32, 784)
        >>> output = model(x)
        >>> print(output.shape)
        (32, 10)

        >>> # Sequential automatically chains module outputs
        >>> model = Sequential(
        ...     Conv2d(3, 64, kernel_size=3),
        ...     BatchNorm2d(64),
        ...     ReLU(),
        ...     MaxPool2d(2)
        ... )
        >>> x = nova.randn(1, 3, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        (1, 64, 15, 15)

        >>> # Access modules by index
        >>> print(len(model))  # 4
        >>> first_layer = model._modules['0']

        >>> # Sequential with activation functions
        >>> model = Sequential(
        ...     Linear(100, 50),
        ...     Tanh(),
        ...     Linear(50, 10),
        ...     Softmax(dim=1)
        ... )

        >>> # Empty Sequential
        >>> model = Sequential()
        >>> print(model)  # Sequential()

    Note:
        ``Sequential`` is particularly useful for building models where data flows
        linearly through layers. For more complex architectures with branching or
        skip connections, consider subclassing ``Module`` directly.
    """

    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            if not isinstance(module, Module):
                raise ValueError(
                    f"Only Module types can be registered in the sequential container, got {type(module)}"
                )

            self.register_module(str(i), module)

    def forward(self, input: Tensor) -> Tensor:
        """Sequentially applies each module to the input.

        Passes the input through each module in order, using the output of
        each module as the input to the next one.

        Args:
            input: Input tensor to be processed by the sequential modules

        Returns:
            Output tensor after passing through all modules in sequence

        Examples::

            >>> model = Sequential(
            ...     Linear(10, 20),
            ...     ReLU(),
            ...     Linear(20, 5)
            ... )
            >>> x = nova.randn(32, 10)
            >>> y = model.forward(x)  # Equivalent to model(x)
            >>> print(y.shape)
            (32, 5)
        """
        for module in self._modules.values():
            input = module(input)
        return input

    def __len__(self):
        """Returns the number of modules in the Sequential container.

        Returns:
            Number of modules stored in this Sequential

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> print(len(model))
            3
        """
        return len(self._modules)

    def _addindent(self, s_: str, numSpaces):
        """Adds indentation to a string representation.

        Helper method for formatting the string representation of the Sequential
        container by adding spaces at the beginning of each line.

        Args:
            s_: String to be indented
            numSpaces: Number of spaces to add as indentation

        Returns:
            Indented string
        """
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    def __repr__(self) -> str:
        """Returns a string representation of the Sequential container.

        Creates a formatted representation that compresses repeated module
        representations for better readability. When multiple consecutive
        modules are identical, they are shown as a range (e.g., "(0-3): 4 x Linear").

        Returns:
            Formatted string representation of the Sequential container

        Examples::

            >>> # Simple sequential
            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> print(model)
            Sequential(
              (0): Linear(in_features=10, out_features=20, bias=True)
              (1): ReLU()
              (2): Linear(in_features=20, out_features=5, bias=True)
            )

            >>> # With repeated modules (compressed representation)
            >>> model = Sequential(
            ...     Linear(10, 10),
            ...     Linear(10, 10),
            ...     Linear(10, 10),
            ...     ReLU()
            ... )
            >>> print(model)
            Sequential(
              (0-2): 3 x Linear(in_features=10, out_features=10, bias=True)
              (3): ReLU()
            )

            >>> # Empty sequential
            >>> model = Sequential()
            >>> print(model)
            Sequential()
        """
        list_of_reprs = [repr(item) for item in self._modules.values()]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._get_name() + "("
        for (start_id, end_id), b in zip(
            start_end_indices, repeated_blocks, strict=True
        ):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = self._addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str
