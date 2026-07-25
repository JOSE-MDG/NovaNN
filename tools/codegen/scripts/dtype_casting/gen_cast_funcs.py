"""Generate the public cast dispatch header for dtype conversions.

Reads a JSON rules file derived from the cast tables and dtype casting
definitions. Each entry contains a dispatch function name, its lookup
table reference, ISA guard requirements, pre-computed C condition
expressions, and variant indices. Renders them through a Jinja2 template
and writes the combined header-only cast.h file to the source tree.
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
_RULES_PATH = RULES_DIR / "dtype_casting" / "cast_funcs_rules.json"
_TARGET_PATH = (
    PROJECT_ROOT / "ncore" / "include" / "ncore" / "headeronly" / "cast.h"
)
_TEMPLATE = (Path("dtype_casting") / "CastFuncs.h.jinja").as_posix()

# Jinja2 environment
_jinja_env = Environment(loader=DEFAULT_LOADER, keep_trailing_newline=True)
_jinja_env.filters["split"] = lambda s, sep=None: s.split(sep)

# Rules
if not _RULES_PATH.exists():
    raise FileNotFoundError(f"rules file not found: {_RULES_PATH}")

with open(_RULES_PATH) as _f:
    _rules: dict[Any, Any] = json.load(_f)

# Engine
_env_specs: list[EnvSpec] = [
    EnvSpec(id=2, env=_jinja_env, name="env_2"),
]

_render: list[Renders] = [
    Renders(
        template_name=_TEMPLATE,
        render_path=_TARGET_PATH,
        formatter=ClangFormatter(target=_TARGET_PATH),
        data=_rules,
    ),
]

_codegen_engine = CodeGenEngine(_env_specs)
_codegen_engine.add_rendering_templates(_env_specs, [_render])

register_engine(
    Engine(
        engine=_codegen_engine,
        name="Cast Functions Declarations",
        id=2,
    )
)
