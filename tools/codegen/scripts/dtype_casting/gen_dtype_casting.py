"""Generate scalar dtype casting functions for the CPU backend.

Reads a JSON rules file defining every dtype cast pair, renders each
rule through a Jinja2 template, and writes the combined C output and
header files to the appropriate locations in the source tree.
"""

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment

from tools.codegen.engine import (
    DEFAULT_LOADER,
    PROJECT_ROOT,
    RULES_DIR,
    ClangFormatter,
    CodeGenEngine,
    Engine,
    EnvSpec,
    Renders,
    register_engine,
)

# Paths
_RULES_PATH = RULES_DIR / "dtype_casting" / "dtype_casting_rules.json"

_TARGET_PATH0 = (
    PROJECT_ROOT / "ncore" / "native" / "cpu" / "dtype" / "DTypeCasting.c"
)

_TARGET_PATH1 = (
    PROJECT_ROOT
    / "ncore"
    / "include"
    / "ncore"
    / "native"
    / "cpu"
    / "dtype"
    / "casting.h"
)

# Jinja2 environment
_TEMPLATE0 = str(Path("dtype_casting") / "DTypeCasting.jinja")
_TEMPLATE1 = str(Path("dtype_casting") / "DTypeCasting.h.jinja")

_jinja_env = Environment(loader=DEFAULT_LOADER, keep_trailing_newline=True)
_jinja_env.filters["split"] = lambda s, sep=None: s.split(sep)

# Rules
if not _RULES_PATH.exists():
    raise FileNotFoundError(f"rules file not found: {_RULES_PATH}")

with open(_RULES_PATH) as _f:
    _rules: dict[Any, Any] = json.load(_f)

# Engine
_env_specs: list[EnvSpec] = [
    EnvSpec(id=0, env=_jinja_env, name="env_0"),
]

_renders: list[list[Renders]] = [
    [
        Renders(
            template_name=_TEMPLATE0,
            render_path=_TARGET_PATH0,
            formatter=ClangFormatter(target=_TARGET_PATH0),
            data=_rules,
        ),
        Renders(
            template_name=_TEMPLATE1,
            render_path=_TARGET_PATH1,
            formatter=ClangFormatter(target=_TARGET_PATH1),
            data=_rules,
        ),
    ]
]

_codegen_engine = CodeGenEngine(_env_specs)
_codegen_engine.add_rendering_templates(_env_specs, _renders)

register_engine(
    Engine(
        engine=_codegen_engine,
        name="Dtype Casting Implementations and Declarations",
        id=0,
    )
)
