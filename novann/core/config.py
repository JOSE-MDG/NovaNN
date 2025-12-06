import os
from typing import Dict
from novann._typing import InitFn


from .constants import *
from .init import (
    calculate_gain,
    kaiming_normal_,
    kaiming_uniform_,
    random_init_,
    xavier_normal_,
    xavier_uniform_,
)


"""Core configuration constants and default initialization maps.

Exposes dictionaries that map activation 
names to default weight initialization functions.
"""

# Default initialization maps
DEFAULT_NORMAL_INIT_MAP: Dict[str, InitFn] = {
    "relu": lambda shape: kaiming_normal_(shape, a=0.0, nonlinearity="relu"),
    "leakyrelu": lambda shape: kaiming_normal_(shape, a=0.01, nonlinearity="leakyrelu"),
    "tanh": lambda shape: xavier_normal_(shape, gain=calculate_gain("tanh")),
    "sigmoid": lambda shape: xavier_normal_(shape, gain=calculate_gain("sigmoid")),
    "default": lambda shape: random_init_(shape),
}

DEFAULT_UNIFORM_INIT_MAP: Dict[str, InitFn] = {
    "relu": lambda shape: kaiming_uniform_(shape, a=0.0, nonlinearity="relu"),
    "leakyrelu": lambda shape: kaiming_uniform_(
        shape, a=0.01, nonlinearity="leakyrelu"
    ),
    "tanh": lambda shape: xavier_uniform_(shape, gain=calculate_gain("tanh")),
    "sigmoid": lambda shape: xavier_uniform_(shape, gain=calculate_gain("sigmoid")),
    "default": lambda shape: random_init_(shape),
}
