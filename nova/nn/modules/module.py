from typing import Iterable, Self, Optional
from collections import OrderedDict
from nova.nn import Parameter, Buffer
from nova.utils import registry_class


@registry_class
class Module:
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._buffers: dict[str, Buffer] = {}
        self._modules: dict[str, Module] = {}
        self._training: bool = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method must be implemented")

    def parameters(self, recurse: bool = True) -> Iterable[Parameter]:

        for param in self._parameters.values():
            if param is not None:
                yield param

        if recurse:
            for module in self._modules.values():
                if module is not None:
                    yield from module.parameters(recurse=True)

    def buffers(self, recurse: bool = True) -> Iterable[Buffer]:

        for buf in self._buffers.values():
            if buf is not None:
                yield buf

        if recurse:
            for module in self._modules.values():
                if module is not None:
                    yield from module.buffers(recurse=True)

    def register_buffer(self, name: str, buffer: Optional[Buffer]) -> None:

        if buffer is None:
            self._buffers[name] = None
        if not isinstance(buffer, Buffer):
            raise ValueError("Only Buffer types can be registered.")
        else:
            self._buffers[name] = buffer

        setattr(self, name, buffer)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:

        if param is None:
            self._parameters[name] = None
        if not isinstance(param, Parameter):
            raise ValueError("Only Parameter types can be registered.")
        else:
            self._parameters[name] = param

        setattr(self, name, param)

    def register_module(self, name: str, module: Optional[Module]) -> None:

        if module is None:
            self._modules[name] = None
        if not isinstance(module, Module):
            raise ValueError("Only Module types can be registered.")
        else:
            self._modules[name] = module

        setattr(self, name, module)

    def __setattr__(self, name: str, value: Parameter | Module | Buffer):

        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Buffer):
            self._buffers[name] = value

        object.__setattr__(self, name, value)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterable[tuple[str, Parameter]]:

        for name, param in self._parameters.items():
            if param is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, param

        if recurse:
            for moduel_name, module in self._modules.items():
                submdule_prefix = f"{moduel_name}.{prefix}" if prefix else moduel_name
                yield from module.named_parameters(prefix=submdule_prefix, recurse=True)

    def named_modules(self, prefix: str = "") -> Iterable[tuple[str, Module]]:

        yield prefix, self

        for name, module in self._modules.items():
            if module is not None:
                module_prfix = f"{prefix}.{name}" if prefix else name
                yield from module.named_modules(prefix=module_prfix)

    def train(self, mode: bool = True) -> Self[Module]:

        self._training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode=mode)
        return self

    def eval(self, mode: bool = False) -> Self[Module]:

        self.train(mode)

        return self

    def state_dict(
        self, destination: Optional[OrderedDict | dict] = None, prefix: str = ""
    ):
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

    def load_state_dict(self, state_dict: OrderedDict | dict, prefix: str = ""):

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

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]

        for i, module in enumerate(self._modules.values()):
            mod_str = repr(module)
            mod_str = self._addindent(mod_str, 2)
            lines.append(f"  ({i}): {mod_str}")

        lines.append(")")
        return "\n".join(lines)

    @staticmethod
    def _addindent(s: str, numSpaces: int):
        lines = s.split("\n")
        if len(lines) == 1:
            return s
        first = lines[0]
        rest = "\n".join(" " * numSpaces + line for line in lines[1:])
        return first + "\n" + rest
