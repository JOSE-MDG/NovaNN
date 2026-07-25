"""Generate lookup tables for CPU dtype casting dispatch.

Reads a JSON rules file containing the cast-table definitions for every
dtype cast pair, renders each rule through a Jinja2 template, and writes
the combined C implementation and header files to the appropriate
locations in the source tree.
"""

from __future__ import annotations

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
_RULES_PATH = RULES_DIR / "dtype_casting" / "cast_tables_rules.json"

_TARGET_PATH0 = (
    PROJECT_ROOT / "ncore" / "src" / "core" / "tables" / "cast_tables.c"
)

_TARGET_PATH1 = (
    PROJECT_ROOT / "ncore" / "include" / "ncore" / "tables" / "cast_tables.h"
)

# Jinja2 environment
_TEMPLATE0 = (Path("dtype_casting") / "CastTables.jinja").as_posix()
_TEMPLATE1 = (Path("dtype_casting") / "CastTables.h.jinja").as_posix()

_jinja_env = Environment(loader=DEFAULT_LOADER, keep_trailing_newline=True)

# Rules
if not _RULES_PATH.exists():
    raise FileNotFoundError(f"rules file not found: {_RULES_PATH}")

with open(_RULES_PATH) as _f:
    _rules: dict[Any, Any] = json.load(_f)

# Engine
_env_specs: list[EnvSpec] = [
    EnvSpec(id=1, env=_jinja_env, name="env_1"),
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
        name="Cast Tables Implementation and Declarations",
        id=1,
    )
)
