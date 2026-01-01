from __future__ import annotations
from typing import Iterable, Self, Optional, TYPE_CHECKING
from collections import OrderedDict
from nova.nn import Parameter, Buffer
from nova.utils import registry_class

if TYPE_CHECKING:
    from nova import Tensor


@registry_class
class Module:
    def __init__(self):
        self._initialized: bool = False
        self._parameters: dict[str, Parameter] = {}
        self._buffers: dict[str, Buffer] = {}
        self._modules: dict[str, Module] = {}
        self._training: bool = True
        self._initialized: bool = True

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
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
        elif not isinstance(buffer, Buffer):
            raise ValueError("Only Buffer types can be registered.")
        else:
            self._buffers[name] = buffer

        setattr(self, name, buffer)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise ValueError("Only Parameter types can be registered.")
        else:
            self._parameters[name] = param

        setattr(self, name, param)

    def register_module(self, name: str, module: Optional[Module]) -> None:

        if module is None:
            self._modules[name] = None
        elif not isinstance(module, Module):
            raise ValueError("Only Module types can be registered.")
        else:
            self._modules[name] = module

        setattr(self, name, module)

    def __setattr__(self, name: str, value: Parameter | Module | Buffer):

        if not getattr(self, "_initialized", False):
            object.__setattr__(self, name, value)
            return

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
                submdule_prefix = f"{prefix}.{moduel_name}" if prefix else moduel_name
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

    def __repr__(self) -> str:
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = self._addindent(mod_str, 2)
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
        return self.__class__.__name__

    def extra_repr(self) -> str:
        return ""

    @staticmethod
    def _addindent(s_: str, numSpaces: int):
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s
