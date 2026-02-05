from typing import Optional, overload
from numpy import ndarray
from nova import Tensor

@overload
def grad(
    outputs: Tensor,
    inputs: Tensor,
    grad_outputs: Optional[Tensor | ndarray | list[Tensor | ndarray]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> ndarray: ...
@overload
def grad(
    outputs: Tensor | list[Tensor],
    inputs: list[Tensor],
    grad_outputs: Optional[Tensor | ndarray | list[Tensor | ndarray]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> list[ndarray]: ...
@overload
def grad(
    outputs: list[Tensor],
    inputs: Tensor,
    grad_outputs: Optional[Tensor | ndarray | list[Tensor | ndarray]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> ndarray: ...
@overload
def grad(
    outputs: list[Tensor],
    inputs: list[Tensor],
    grad_outputs: Optional[Tensor | ndarray | list[Tensor | ndarray]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> list[ndarray]: ...

# Implementation signature (catch-all)
def grad(
    outputs: Tensor | list[Tensor],
    inputs: Tensor | list[Tensor],
    grad_outputs: Optional[Tensor | ndarray | list[Tensor | ndarray]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> list[ndarray] | ndarray: ...
