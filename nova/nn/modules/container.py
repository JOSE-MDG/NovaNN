from __future__ import annotations
import operator
from itertools import islice
from collections import OrderedDict
from nova.nn.modules import Module
from typing import TYPE_CHECKING, Iterable, Iterator, TypeVar, overload, Self

if TYPE_CHECKING:
    from nova import Tensor


def _addindent(s_: str, numSpaces):
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


_V = TypeVar("_V")


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

    _modules: dict[str, Module]

    @overload
    def __init__(self, *args: Module) -> None: ...

    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args: Module):  # pyright: ignore[reportInconsistentOverload]
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, module in args[0].items():
                self.register_module(name, module)
        else:
            for idx, module in enumerate(args):
                self.register_module(str(idx), module)

    def _get_item_by_idx(self, iterator: Iterable[_V], idx: int) -> _V:
        """Retrieves an item from an iterator by its index.

        Internal helper method that handles index normalization and retrieval
        from an iterable using islice.

        Args:
            iterator: Iterable to retrieve item from
            idx: Index of the item to retrieve (supports negative indexing)

        Returns:
            Item at the specified index

        Raises:
            IndexError: If index is out of range
        """
        idx = operator.index(idx)
        self._check_idx(idx)

        idx %= len(self._modules)
        return next(islice(iterator, idx, None))

    def _check_idx(self, idx: int) -> None:
        """Validates that an index is within valid range.

        Args:
            idx: Index to validate

        Raises:
            IndexError: If index is out of range [-n, n) where n is the number of modules
        """
        n = len(self._modules)
        if not -n <= idx < n:
            raise IndexError(f"Index {idx} is out of range")

    def __getitem__(self, idx: slice | int) -> Sequential | Module:
        """Retrieves a module or sub-sequence by index or slice.

        Supports both integer indexing (to get a single module) and slice notation
        (to get a new Sequential containing a subset of modules).

        Args:
            idx: Integer index or slice object

        Returns:
            Single Module if idx is int, or Sequential if idx is slice

        Raises:
            IndexError: If integer index is out of range

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> first_layer = model[0]
            >>> print(first_layer)
            Linear(in_features=10, out_features=20, bias=True)

            >>> # Negative indexing
            >>> last_layer = model[-1]
            >>> print(last_layer)
            Linear(in_features=20, out_features=5, bias=True)

            >>> # Slicing
            >>> sub_model = model[0:2]
            >>> print(sub_model)
            Sequential(
              (0): Linear(in_features=10, out_features=20, bias=True)
              (1): ReLU()
            )

            >>> # Step slicing
            >>> model = Sequential(Linear(10, 10), ReLU(), Linear(10, 10), ReLU())
            >>> linear_only = model[::2]  # Gets every other layer
        """
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        """Replaces a module at the specified index.

        Args:
            idx: Index of the module to replace (supports negative indexing)
            module: New module to set at the specified position

        Raises:
            IndexError: If index is out of range

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> model[1] = Tanh()  # Replace ReLU with Tanh
            >>> print(model)
            Sequential(
              (0): Linear(in_features=10, out_features=20, bias=True)
              (1): Tanh()
              (2): Linear(in_features=20, out_features=5, bias=True)
            )

            >>> # Replace using negative indexing
            >>> model[-1] = Linear(20, 10)
        """
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: slice | int) -> None:
        """Deletes a module or range of modules from the container.

        After deletion, remaining modules are automatically renumbered to maintain
        sequential integer keys starting from 0.

        Args:
            idx: Integer index or slice specifying which module(s) to delete

        Raises:
            IndexError: If integer index is out of range

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 10), Tanh())
            >>> del model[1]  # Remove ReLU
            >>> print(model)
            Sequential(
              (0): Linear(in_features=10, out_features=20, bias=True)
              (1): Linear(in_features=20, out_features=10, bias=True)
              (2): Tanh()
            )

            >>> # Delete multiple modules with slice
            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 10), Tanh())
            >>> del model[1:3]  # Remove ReLU and Linear
            >>> print(model)
            Sequential(
              (0): Linear(in_features=10, out_features=20, bias=True)
              (1): Tanh()
            )

            >>> # Delete using negative indexing
            >>> del model[-1]  # Remove last module
        """
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(
            zip(str_indices, self._modules.values(), strict=True)
        )

    def __iter__(self) -> Iterator[Module]:
        """Returns an iterator over the modules in the container.

        Returns:
            Iterator over Module instances

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> for module in model:
            ...     print(type(module).__name__)
            Linear
            ReLU
            Linear

            >>> # Use in list comprehension
            >>> layer_types = [type(m).__name__ for m in model]
        """
        return iter(self._modules.values())

    def pop(self, key: int | slice):
        """Removes and returns a module or slice of modules from the container.

        The container is automatically renumbered after removal to maintain
        sequential integer keys.

        Args:
            key: Integer index or slice specifying which module(s) to remove

        Returns:
            Removed Module (if key is int) or Sequential (if key is slice)

        Raises:
            IndexError: If integer index is out of range

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> activation = model.pop(1)
            >>> print(activation)
            ReLU()
            >>> print(len(model))
            2

            >>> # Pop last element
            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> last = model.pop(-1)
            >>> print(last)
            Linear(in_features=20, out_features=5, bias=True)

            >>> # Pop multiple modules with slice
            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 10), Tanh())
            >>> middle = model.pop(1:3)
            >>> print(middle)
            Sequential(
              (0): ReLU()
              (1): Linear(in_features=20, out_features=10, bias=True)
            )
        """
        v = self[key]
        del self[key]
        return v

    def append(self, module: Module) -> Self:
        """Appends a given module to the end of the container.

        Args:
            module: Module to append

        Returns:
            Self for method chaining

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU())
            >>> model.append(Linear(20, 5))
            >>> print(model)
            Sequential(
              (0): Linear(in_features=10, out_features=20, bias=True)
              (1): ReLU()
              (2): Linear(in_features=20, out_features=5, bias=True)
            )

            >>> # Method chaining
            >>> model = Sequential(Linear(10, 20))
            >>> model.append(ReLU()).append(Linear(20, 5)).append(Softmax(dim=1))
        """
        self.register_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module) -> Self:
        """Inserts a module into the Sequential container at the specified index.

        All modules at and after the insertion index are shifted to the right.
        Supports negative indexing.

        Args:
            index: The index to insert the module (supports negative indexing)
            module: The module to be inserted

        Returns:
            Self for method chaining

        Raises:
            AssertionError: If module is not an instance of Module
            IndexError: If index is out of range

        Examples::

            >>> model = Sequential(Linear(10, 20), Linear(20, 5))
            >>> model.insert(1, ReLU())
            >>> print(model)
            Sequential(
              (0): Linear(in_features=10, out_features=20, bias=True)
              (1): ReLU()
              (2): Linear(in_features=20, out_features=5, bias=True)
            )

            >>> # Insert at beginning
            >>> model = Sequential(Linear(10, 20), ReLU())
            >>> model.insert(0, BatchNorm1d(10))

            >>> # Negative indexing
            >>> model = Sequential(Linear(10, 20), Linear(20, 5))
            >>> model.insert(-1, ReLU())  # Insert before last layer

            >>> # Method chaining
            >>> model.insert(0, Dropout(0.5)).insert(2, Dropout(0.3))
        """
        if not isinstance(module, Module):
            raise AssertionError(f"module should be of type: {Module}")

        n = len(self._modules)
        self._check_idx(index)

        if index < 0:
            index += n

        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential: Iterable[Module]) -> Self:
        """Extends the current Sequential container with modules from an iterable.

        Appends all modules from the provided iterable to the end of the current
        container in order.

        Args:
            sequential: An iterable of Module instances to be added to the container

        Returns:
            Self for method chaining

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU())
            >>> other = Sequential(Linear(20, 10), Tanh())
            >>> model.extend(other)
            >>> print(model)
            Sequential(
              (0): Linear(in_features=10, out_features=20, bias=True)
              (1): ReLU()
              (2): Linear(in_features=20, out_features=10, bias=True)
              (3): Tanh()
            )

            >>> # Extend with a list of modules
            >>> model = Sequential(Linear(10, 20))
            >>> model.extend([ReLU(), Dropout(0.5), Linear(20, 5)])

            >>> # Method chaining
            >>> model.extend(encoder_layers).extend(decoder_layers)
        """
        for layer in sequential:
            self.append(layer)
        return self

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

            >>> # The forward pass chains outputs automatically
            >>> model = Sequential(Linear(784, 256), ReLU(), Linear(256, 10))
            >>> x = nova.randn(64, 784)
            >>> # x -> Linear -> ReLU -> Linear -> output
            >>> output = model(x)
        """
        for module in self:
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

            >>> # Empty Sequential
            >>> model = Sequential()
            >>> print(len(model))
            0

            >>> # Check if container is empty
            >>> if len(model) == 0:
            ...     print("No modules in container")
        """
        return len(self._modules)

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
        list_of_reprs = [repr(item) for item in self]
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

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    # TODO: Implement arithmetic operations between containers
