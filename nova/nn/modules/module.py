from __future__ import annotations
from typing import Iterable, Self, Optional, TYPE_CHECKING
from collections import OrderedDict
from nova.utils import registry_class
from nova.nn.parameter import (
    UninitializedParameter,
    UninitializedBuffer,
    is_lazy,
    Parameter,
    Buffer,
)

if TYPE_CHECKING:
    from nova import Tensor


def _addindent(s_: str, numSpaces: int):
    """Adds indentation to a multi-line string.

    Helper method for formatting the string representation by adding spaces
    to each line.

    Args:
        s_: String to indent
        numSpaces: Number of spaces to add to each line

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


class ModuleMeta(type):
    """
    Metaclass for automatic registration of Module subclasses.

    This metaclass automatically registers all classes that inherit from
    Module in the serialization registry. This enables safe deserialization
    of any Module subclass without requiring explicit @registry_class
    decorators on each class.

    The registration happens at class definition time, ensuring that all
    Module subclasses (Linear, Conv2d, ReLU, etc.) can be safely loaded
    with weights_only=True.

    Note:
        The Module base class itself is not registered to avoid redundancy.
        Only its subclasses are automatically registered.

    Examples::

        >>> from nova.nn import Module
        >>>
        >>> # This class is automatically registered
        >>> class CustomLayer(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        >>> # Can be safely serialized and deserialized
        >>> import nova
        >>> layer = CustomLayer()
        >>> nova.save(layer, "custom_layer.pth")
        >>> loaded = nova.load("custom_layer.pth", weights_only=True)
        ✅ Successfully loaded from custom_layer.pth
        >>>
        >>> # All built-in modules are also auto-registered
        >>> from nova.nn import Linear
        >>> model = Linear(10, 5)
        >>> nova.save(model, "model.pth")
        >>> loaded_model = nova.load("model.pth", weights_only=True)

    See Also:
        - :func:`nova.utils.decorators.registry.registry_class`: Manual registration
        - :class:`nova.nn.Module`: Base class for all neural network modules
        - :func:`nova.serialization.load`: Safe deserialization with weights_only
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Create a new Module subclass and register it automatically.

        This method is called when a new class inheriting from Module is
        defined. It creates the class normally and then registers it in
        the serialization registry for safe deserialization.

        Args:
            name: Name of the class being created
            bases: Tuple of base classes
            namespace: Dictionary containing class attributes and methods
            **kwargs: Additional keyword arguments passed to type.__new__

        Returns:
            The newly created and registered class

        Note:
            This is called automatically by Python when defining classes.
            Users should not call this method directly.
        """
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Auto-register the class (except Module itself)
        if name != "Module":
            registry_class(cls)

        return cls


class Module(metaclass=ModuleMeta):
    """Base class for all neural network modules.

    Your models should subclass this class. Modules can contain other Modules,
    allowing to nest them in a tree structure. You can assign the submodules as
    regular attributes.

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`train`, :meth:`eval`, etc.

    The Module class automatically tracks three types of objects:

    - **Parameters**: Learnable tensors that are updated during training
    - **Buffers**: Non-learnable tensors (e.g., running statistics in BatchNorm)
    - **Submodules**: Other Module instances that form the model hierarchy

    When you assign a Parameter, Buffer, or Module to an attribute of your Module
    subclass, it is automatically registered and will be included in
    :meth:`parameters`, :meth:`buffers`, or :meth:`modules` respectively.

    Examples::

        >>> # Simple custom module
        >>> class MyModule(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = Linear(10, 20)
        ...         self.activation = ReLU()
        ...
        ...     def forward(self, x):
        ...         x = self.linear(x)
        ...         x = self.activation(x)
        ...         return x
        ...
        >>> model = MyModule()
        >>> x = nova.randn(5, 10)
        >>> output = model(x)
        >>> print(output.shape)
        (5, 20)

        >>> # Module with custom parameters
        >>> class MyLinear(Module):
        ...     def __init__(self, in_features, out_features):
        ...         super().__init__()
        ...         self.weight = Parameter(nova.randn(out_features, in_features))
        ...         self.bias = Parameter(nova.zeros(out_features))
        ...
        ...     def forward(self, x):
        ...         return x @ self.weight.T + self.bias
        ...
        >>> linear = MyLinear(10, 5)
        >>> print(len(list(linear.parameters())))  # 2 (weight and bias)

        >>> # Nested modules
        >>> class CNN(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = Conv2d(3, 64, 3)
        ...         self.conv2 = Conv2d(64, 128, 3)
        ...         self.fc = Linear(128, 10)
        ...
        ...     def forward(self, x):
        ...         x = self.conv1(x)
        ...         x = self.conv2(x)
        ...         x = self.fc(x)
        ...         return x
        ...
        >>> model = CNN()
        >>> # All parameters from submodules are accessible
        >>> print(sum(p.numel() for p in model.parameters()))

    Note:
        The :meth:`forward` method must be implemented by all subclasses. It defines
        the computation performed at every call and should not be called directly.
        Instead, call the module instance itself (which invokes ``__call__``).
    """

    def __init__(self):
        """Initializes internal Module state.

        Sets up the internal dictionaries for tracking parameters, buffers, and
        submodules, and initializes the training mode flag.
        """
        self._initialized: bool = False
        self._parameters: dict[str, Parameter] = {}
        self._buffers: dict[str, Buffer] = {}
        self._modules: dict[str, Module] = {}
        self._training: bool = True
        self._initialized: bool = True

    def __call__(self, *args, **kwargs) -> Tensor:
        """Calls the forward method when the module is called as a function.

        This allows you to use the module instance as a callable, which internally
        invokes the :meth:`forward` method.

        Args:
            *args: Variable length argument list passed to forward
            **kwargs: Arbitrary keyword arguments passed to forward

        Returns:
            Output from the forward method

        Examples::

            >>> m = Linear(10, 5)
            >>> x = nova.randn(3, 10)
            >>> output = m(x)  # Calls m.forward(x)
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Should be overridden by all subclasses. Although the recipe for forward
        pass needs to be defined within this function, one should call the Module
        instance afterwards instead of this since the former takes care of running
        registered hooks while the latter silently ignores them.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            Output tensor(s) from the forward computation

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Examples::

            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = Linear(10, 5)
            ...
            ...     def forward(self, x):
            ...         return self.linear(x)
        """
        raise NotImplementedError("forward method must be implemented")

    def parameters(self, recurse: bool = True) -> Iterable[Parameter]:
        """Returns an iterator over module parameters.

        This is typically passed to an optimizer to update the model's learnable
        parameters during training.

        Args:
            recurse: If ``True``, yields parameters of this module and all submodules.
                Otherwise, yields only parameters that are direct members of this module.
                Default: ``True``

        Yields:
            Parameter: Module parameters

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> params = list(model.parameters())
            >>> print(len(params))  # 4 (2 weights + 2 biases)

            >>> # Only direct parameters (not from submodules)
            >>> module = Sequential(Linear(10, 20), Linear(20, 5))
            >>> direct_params = list(module.parameters(recurse=False))
            >>> print(len(direct_params))  # 0 (Sequential has no direct parameters)

            >>> # Typical usage with optimizer
            >>> optimizer = SGD(model.parameters(), lr=0.01)
        """
        for param in self._parameters.values():
            if param is not None:
                yield param

        if recurse:
            for module in self._modules.values():
                if module is not None:
                    yield from module.parameters(recurse=True)

    def buffers(self, recurse: bool = True) -> Iterable[Buffer]:
        """Returns an iterator over module buffers.

        Buffers are non-learnable tensors that are part of the module's state but
        are not updated by the optimizer. Common examples include running statistics
        in BatchNorm layers.

        Args:
            recurse: If ``True``, yields buffers of this module and all submodules.
                Otherwise, yields only buffers that are direct members of this module.
                Default: ``True``

        Yields:
            Buffer: Module buffers

        Examples::

            >>> bn = BatchNorm2d(64)
            >>> buffers = list(bn.buffers())
            >>> print(len(buffers))  # 3 (running_mean, running_var, num_batches_tracked)

            >>> # Buffers are not parameters
            >>> print(len(list(bn.parameters())))  # 2 (weight, bias)
        """
        for buf in self._buffers.values():
            if buf is not None:
                yield buf

        if recurse:
            for module in self._modules.values():
                if module is not None:
                    yield from module.buffers(recurse=True)

    def modules(self, recurse: bool = True) -> Iterable[Module]:
        """Returns an iterator over all modules in the network.

        This yields all submodules contained in this module, which is useful for
        operations that need to be applied to all layers (e.g., initialization,
        mode switching, or inspection).

        Args:
            recurse: If ``True``, yields modules of this module and all submodules
                recursively. Otherwise, yields only direct child modules.
                Default: ``True``

        Yields:
            Module: Child modules

        Note:
            Unlike ``parameters()`` and ``buffers()``, this method does NOT yield
            ``self``. It only yields the submodules contained in ``self._modules``.

        Examples::

            >>> model = Sequential(
            ...     Linear(10, 20),
            ...     ReLU(),
            ...     BatchNorm1d(20),
            ...     Linear(20, 5)
            ... )
            >>> modules = list(model.modules())
            >>> print(len(modules))  # 4 (all submodules)
            >>> for m in modules:
            ...     print(type(m).__name__)
            # Linear
            # ReLU
            # BatchNorm1d
            # Linear

            >>> # Only direct children (not nested)
            >>> class TwoLayerNet(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.block1 = Sequential(Linear(10, 20), ReLU())
            ...         self.block2 = Linear(20, 5)
            ...
            >>> model = TwoLayerNet()
            >>> direct = list(model.modules(recurse=False))
            >>> print(len(direct))  # 2 (block1, block2)
            >>> all_modules = list(model.modules(recurse=True))
            >>> print(len(all_modules))  # 4 (block1, Linear, ReLU, block2)

            >>> # Apply operation to all modules
            >>> for m in model.modules():
            ...     if isinstance(m, Linear):
            ...         nn.init.xavier_normal_(m.weight)
        """
        for m in self._modules.values():
            if m is not None:
                yield m

        if recurse:
            for m in self._modules.values():
                if m is not None:
                    yield from m.modules(recurse=True)

    def register_buffer(self, name: str, buffer: Optional[Buffer]) -> None:
        """Adds a buffer to the module.

        This is typically used to register a buffer that should be part of the
        module's state but should not be considered a model parameter (e.g.,
        running statistics in BatchNorm).

        The buffer can be accessed as an attribute using the given name.

        Args:
            name: Name of the buffer. The buffer can be accessed from this module
                using the given name
            buffer: Buffer to be registered. If ``None``, no buffer is registered

        Raises:
            ValueError: If buffer is not a Buffer instance and not lazy

        Examples::

            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer('running_mean', Buffer(nova.zeros(10)))
            ...
            >>> m = MyModule()
            >>> print(m.running_mean)  # Access as attribute
            >>> print(list(m.buffers()))  # Shows in buffers()

            >>> # Registering None
            >>> m.register_buffer('optional_buffer', None)
            >>> print(m.optional_buffer)  # None
        """
        if buffer is None:
            self._buffers[name] = None
        elif not isinstance(buffer, Buffer) and not is_lazy(buffer):
            raise ValueError("Only Buffer types can be registered.")
        else:
            self._buffers[name] = buffer

        setattr(self, name, buffer)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Adds a parameter to the module.

        The parameter can be accessed as an attribute using the given name.
        This is typically used when you want to register a parameter that should
        be updated by the optimizer.

        Args:
            name: Name of the parameter. The parameter can be accessed from this
                module using the given name
            param: Parameter to be registered. If ``None``, no parameter is registered

        Raises:
            ValueError: If param is not a Parameter instance and not lazy

        Examples::

            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_parameter('weight', Parameter(nova.randn(5, 10)))
            ...         self.register_parameter('bias', Parameter(nova.zeros(5)))
            ...
            >>> m = MyModule()
            >>> print(m.weight.shape)  # (5, 10)
            >>> print(len(list(m.parameters())))  # 2

            >>> # Registering None (useful for optional parameters)
            >>> m.register_parameter('optional_param', None)
            >>> print(m.optional_param)  # None
        """
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter) and not is_lazy(param):
            raise ValueError("Only Parameter types can be registered.")
        else:
            self._parameters[name] = param

        setattr(self, name, param)

    def register_module(self, name: str, module: Optional[Module]) -> None:
        """Adds a child module to the current module.

        The module can be accessed as an attribute using the given name. This is
        typically used internally when you assign a Module to an attribute, but
        can also be called explicitly.

        Args:
            name: Name of the child module. The module can be accessed from this
                module using the given name
            module: Child module to be registered. If ``None``, no module is registered

        Raises:
            ValueError: If module is not a Module instance

        Examples::

            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_module('linear', Linear(10, 5))
            ...
            >>> m = MyModule()
            >>> print(m.linear)  # Access as attribute
            >>> print(list(m.modules()))  # Shows in modules()

            >>> # Usually done automatically via assignment
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = Linear(10, 5)  # Automatically registered
        """
        if module is None:
            self._modules[name] = None
        elif not isinstance(module, Module):
            raise ValueError("Only Module types can be registered.")
        else:
            self._modules[name] = module

        setattr(self, name, module)

    def zero_grad(self, set_to_none: bool = True):
        """Zero out the gradients of all parameters in the module.

        This method is typically called before the backward pass to clear
        gradients from the previous iteration. Setting gradients to None
        instead of zero can be more memory efficient.

        Args:
            set_to_none: If True, sets gradients to None instead of zero.
                This is more memory-efficient as it avoids allocation.
                Defaults to True.

        Examples:
            Basic usage in training loop:

            >>> model = MyModel()
            >>> optimizer = SGD(model.parameters(), lr=0.01)
            >>> for inputs, targets in dataloader:
            ...     model.zero_grad()  # Clear previous gradients
            ...     outputs = model(inputs)
            ...     loss = criterion(outputs, targets)
            ...     loss.backward()
            ...     optimizer.step()

            Using set_to_none=False for debugging:

            >>> model.zero_grad(set_to_none=False)
            >>> # Gradients are now zero tensors instead of None
            >>> # Useful when you need to inspect gradient values

            Manual gradient accumulation:

            >>> accumulation_steps = 4
            >>> model.zero_grad()
            >>> for i, (inputs, targets) in enumerate(dataloader):
            ...     outputs = model(inputs)
            ...     loss = criterion(outputs, targets) / accumulation_steps
            ...     loss.backward()
            ...     if (i + 1) % accumulation_steps == 0:
            ...         optimizer.step()
            ...         model.zero_grad()
        """
        for param in self.parameters():
            param.zero_grad(set_to_none=set_to_none)

    def __setattr__(
        self,
        name: str,
        value: (
            Parameter | Module | Buffer | UninitializedBuffer | UninitializedParameter
        ),
    ):
        """Intercepts attribute assignment to register parameters, buffers, and modules.

        This method is called whenever you assign an attribute to the module. It
        automatically detects if the value is a Parameter, Buffer, or Module and
        registers it appropriately.

        Args:
            name: Name of the attribute
            value: Value being assigned (Parameter, Buffer, Module, or other)

        Note:
            This method enables the convenient syntax of ``self.linear = Linear(10, 5)``
            automatically registering the Linear module as a submodule.
        """
        if not getattr(self, "_initialized", False):
            object.__setattr__(self, name, value)
            return

        if isinstance(value, (Parameter, UninitializedParameter)):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, (Buffer, UninitializedBuffer)):
            self._buffers[name] = value

        object.__setattr__(self, name, value)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterable[tuple[str, Parameter]]:
        """Returns an iterator over module parameters, yielding both name and parameter.

        This is useful when you need to know the name of each parameter, for example
        when debugging or when you want to apply different learning rates to different
        layers.

        Args:
            prefix: Prefix to prepend to all parameter names. Default: ``""``
            recurse: If ``True``, yields parameters of this module and all submodules.
                Otherwise, yields only parameters that are direct members of this module.
                Default: ``True``

        Yields:
            Tuple[str, Parameter]: Tuple of parameter name and parameter

        Examples::

            >>> model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> for name, param in model.named_parameters():
            ...     print(name, param.shape)
            0.weight (20, 10)
            0.bias (1, 20)
            2.weight (5, 20)
            2.bias (1, 5)

            >>> # With prefix
            >>> for name, param in model.named_parameters(prefix='model'):
            ...     print(name)
            model.0.weight
            model.0.bias
            model.2.weight
            model.2.bias

            >>> # Apply different learning rates
            >>> param_groups = [
            ...     {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'lr': 0.01},
            ...     {'params': [p for n, p in model.named_parameters() if 'weight' in n], 'lr': 0.001}
            ... ]
        """
        for name, param in self._parameters.items():
            if param is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, param

        if recurse:
            for module_name, module in self._modules.items():
                submodule_prefix = f"{prefix}.{module_name}" if prefix else module_name
                yield from module.named_parameters(
                    prefix=submodule_prefix, recurse=True
                )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterable[tuple[str, Buffer]]:
        """Returns an iterator over module buffers, yielding both name and buffer.

        Args:
            prefix: Prefix to prepend to all buffer names. Default: ``""``
            recurse: If ``True``, yields buffers of this module and all submodules.
                Otherwise, yields only buffers that are direct members of this module.
                Default: ``True``

        Yields:
            Tuple[str, Buffer]: Tuple of buffer name and buffer

        Examples::

            >>> model = Sequential(BatchNorm2d(64), Conv2d(64, 128, 3))
            >>> for name, buffer in model.named_buffers():
            ...     print(name, buffer.shape)
            0.running_mean (64,)
            0.running_var (64,)
            0.num_batches_tracked ()
        """
        for name, buf in self._buffers.items():
            if buf is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, buf

        if recurse:
            for module_name, module in self._modules.items():
                submodule_prefix = f"{prefix}.{module_name}" if prefix else module_name
                yield from module.named_buffers(prefix=submodule_prefix, recurse=True)

    def named_modules(self, prefix: str = "") -> Iterable[tuple[str, Module]]:
        """Returns an iterator over all modules in the network, yielding both name and module.

        This includes the module itself as well as all descendant modules. This is useful
        for inspecting the model structure or applying operations to specific modules.

        Args:
            prefix: Prefix to prepend to all module names. Default: ``""``

        Yields:
            Tuple[str, Module]: Tuple of module name and module

        Examples::

            >>> model = Sequential(
            ...     Linear(10, 20),
            ...     ReLU(),
            ...     Linear(20, 5)
            ... )
            >>> for name, module in model.named_modules():
            ...     print(name, type(module).__name__)
             Sequential
            0 Linear
            1 ReLU
            2 Linear

            >>> # Apply operation to specific module types
            >>> for name, module in model.named_modules():
            ...     if isinstance(module, Linear):
            ...         print(f"Linear layer: {name}")
        """
        yield prefix, self

        for name, module in self._modules.items():
            if module is not None:
                module_prfix = f"{prefix}.{name}" if prefix else name
                yield from module.named_modules(prefix=module_prfix)

    def train(self, mode: bool = True) -> Self[Module]:
        """Sets the module in training mode.

        This has an effect on certain modules like Dropout and BatchNorm that behave
        differently during training and evaluation. Calling ``train()`` recursively
        sets all submodules to training mode as well.

        Args:
            mode: Whether to set training mode (``True``) or evaluation mode (``False``).
                Default: ``True``

        Returns:
            Self: Returns self for method chaining

        Examples::

            >>> model = Sequential(Linear(10, 20), Dropout(0.5), Linear(20, 5))
            >>> model.train()  # Enable training mode
            >>> # Dropout is now active
            >>> output = model(x)

            >>> model.eval()  # Disable training mode
            >>> # Dropout is now disabled
            >>> output = model(x)

            >>> # Method chaining
            >>> model.train().forward(x)
        """
        self._training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode=mode)
        return self

    def eval(self, mode: bool = False) -> Self[Module]:
        """Sets the module in evaluation mode.

        This is equivalent to ``self.train(False)``. It affects modules like Dropout
        and BatchNorm that behave differently during training and evaluation.

        Args:
            mode: Opposite of training mode. If ``False`` (default), sets training mode.
                This parameter exists for consistency but is typically not used.

        Returns:
            Self: Returns self for method chaining

        Examples::

            >>> model = Sequential(BatchNorm2d(64), Dropout(0.5))
            >>> model.eval()  # Disable training mode
            >>> # BatchNorm uses running statistics, Dropout is disabled
            >>> output = model(x)

            >>> # Common pattern
            >>> model.train()  # Training
            >>> for epoch in range(num_epochs):
            ...     # training loop
            ...     pass
            >>> model.eval()  # Evaluation
            >>> # evaluation loop
        """
        self.train(mode)
        return self

    def state_dict(
        self, destination: Optional[OrderedDict | dict] = None, prefix: str = ""
    ) -> dict:
        """Returns a dictionary containing the whole state of the module.

        Both parameters and persistent buffers (e.g., running averages in BatchNorm)
        are included. Keys are the corresponding parameter and buffer names. Parameters
        and buffers set to ``None`` are not included.

        The returned dictionary can be used to restore the module's state later using
        :meth:`load_state_dict`.

        Args:
            destination: If provided, the state of module will be updated into this dict
                and the same object is returned. Otherwise, an OrderedDict is created
                and returned. Default: ``None``
            prefix: Prefix added to parameter and buffer names. Default: ``""``

        Returns:
            Dict containing the complete state of the module

        Examples::

            >>> model = Sequential(Linear(10, 20), Linear(20, 5))
            >>> state = model.state_dict()
            >>> print(state.keys())
            odict_keys(['0.weight', '0.bias', '1.weight', '1.bias'])

            >>> # Save to disk
            >>> nova.save(model.state_dict(), 'model.pth')

            >>> # With BatchNorm (includes buffers)
            >>> model = Sequential(Conv2d(3, 64, 3), BatchNorm2d(64))
            >>> state = model.state_dict()
            >>> # Includes running_mean, running_var, etc.
        """
        if destination is None:
            destination = OrderedDict()

        for name, param in self._parameters.items():
            destination[prefix + name] = param.detach() if param is not None else None

        for name, buf in self._buffers.items():
            destination[prefix + name] = buf.detach() if buf is not None else None

        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".")

        return destination

    def load_state_dict(self, state_dict: OrderedDict | dict, prefix: str = "") -> None:
        """Copies parameters and buffers from ``state_dict`` into this module and its descendants.

        If a key in ``state_dict`` corresponds to a parameter or buffer in this module,
        the corresponding parameter or buffer is updated with the value from ``state_dict``.
        This is typically used to restore a saved model.

        Args:
            state_dict: Dictionary containing parameters and persistent buffers
            prefix: Prefix for parameter and buffer names in the state_dict. Default: ``""``

        Examples::

            >>> model = Sequential(Linear(10, 20), Linear(20, 5))
            >>> # Save state
            >>> state = model.state_dict()
            >>> nova.save(state, 'model.pth')

            >>> # Create new model and load state
            >>> new_model = Sequential(Linear(10, 20), Linear(20, 5))
            >>> loaded_state = nova.load('model.pth')
            >>> new_model.load_state_dict(loaded_state)

            >>> # Transfer learning: load partial state
            >>> pretrained_state = nova.load('pretrained.pth')
            >>> model.load_state_dict(pretrained_state)
            >>> # Only matching keys are loaded
        """
        for name, param in self._parameters.items():
            key = prefix + name
            if key in state_dict:
                if param is not None:
                    param.copy_(state_dict[key]) if param is not None else None

        for name, buf in self._buffers.items():
            key = prefix + name
            if key in state_dict:
                buf.copy_(state_dict[key]) if buf is not None else None

        for name, module in self._modules.items():
            module.load_state_dict(state_dict=state_dict, prefix=prefix + name + ".")

    def __repr__(self) -> str:
        """Returns a string representation of the module.

        The string representation shows the module's structure, including all
        submodules and their configurations. This is useful for debugging and
        understanding the model architecture.

        Returns:
            String representation of the module

        Examples::

            >>> model = Sequential(
            ...     Linear(10, 20),
            ...     ReLU(),
            ...     Linear(20, 5)
            ... )
            >>> print(model)
            Sequential(
              (0): Linear(in_features=10, out_features=20, bias=True)
              (1): ReLU()
              (2): Linear(in_features=20, out_features=5, bias=True)
            )

            >>> # Custom module
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv = Conv2d(3, 64, 3)
            ...         self.bn = BatchNorm2d(64)
            ...
            ...     def extra_repr(self):
            ...         return 'custom_param=42'
            ...
            >>> m = MyModule()
            >>> print(m)
            MyModule(
              custom_param=42
              (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=True)
              (bn): BatchNorm2d(64, momentum=0.1, eps=1e-05, affine=True, track_running_stats=True)
            )
        """
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def _get_name(self) -> str:
        """Returns the name of the module class.

        Returns:
            Class name as string
        """
        return self.__class__.__name__

    def extra_repr(self) -> str:
        """Sets the extra representation of the module.

        To print customized extra information, you should re-implement this method
        in your own modules. Both single-line and multi-line strings are acceptable.

        Returns:
            String containing extra information about the module

        Examples::

            >>> class MyModule(Module):
            ...     def __init__(self, param1, param2):
            ...         super().__init__()
            ...         self.param1 = param1
            ...         self.param2 = param2
            ...
            ...     def extra_repr(self):
            ...         return f'param1={self.param1}, param2={self.param2}'
            ...
            >>> m = MyModule(10, 20)
            >>> print(m)
            MyModule(param1=10, param2=20)
        """
        return ""
