from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from jinja2 import (
    Environment,
    FileSystemLoader,
    Template,
    TemplateError,
    TemplateNotFound,
)
from tqdm import tqdm

"""Core classes for Jinja2-based template-driven code generation.

Provides CodeGenEngine for rendering templates with JSON/YAML rules
files, EngineManager for orchestrating multiple generation stages,
the data classes Renders, EnvSpec and Engine, and module-level
constants for project paths, formatter binaries and the default
template loader.
"""


class Renders(NamedTuple):
    """Hold the data needed to render a single template.

    Attributes:
        template_name: Name of the Jinja2 template file (located
            under templates/ with a .jinja suffix).
        render_path: Destination path for the generated file.
        formatter: Optional formatter to apply after rendering
            (e.g. ``ClangFormatter`` or ``RuffFormatter``).
        data: Template variables injected during rendering,
            loaded from a JSON rules file.
    """

    template_name: str
    render_path: Path
    formatter: ClangFormatter | RuffFormatter | None
    data: dict[Any, Any]


class EnvSpec(NamedTuple):
    """Specify a Jinja2 environment to register.

    Attributes:
        id: Unique identifier for the environment.
        env: The Jinja2 Environment instance.
        name: Expected to follow the pattern 'env_{id}'.
    """

    id: int
    env: Environment
    name: str


class Engine(NamedTuple):
    """Wrap a CodeGenEngine with a name and id for the manager.

    Each engine represents one logical stage in the generation
    pipeline (e.g. processing a rules file to produce outputs
    from a template).

    Attributes:
        engine: The underlying CodeGenEngine instance.
        name: Human-readable name for this stage.
        id: Unique identifier for the engine.
    """

    engine: CodeGenEngine
    name: str
    id: int


class _EnvEntry:
    """Contain a registered environment and its rendering state."""

    def __init__(self, env_obj: EnvSpec) -> None:
        self.id = env_obj.id
        self.name = env_obj.name
        self.env = env_obj.env
        self.templates: list[Template] = []
        self.renders: list[Renders] = []


class EngineRunResult(NamedTuple):
    """Outcome of running a single engine within EngineManager.run().

    Attributes:
        engine: The engine that was run.
        ok: True if generation completed without raising.
        error: The exception raised, if any (None when ok is True).
    """

    engine: Engine
    ok: bool
    error: Exception | None


@dataclass(frozen=True)
class Formatter:
    """Base configuration for a code formatter binary.

    Attributes:
        name: Human-readable formatter name (e.g. ``"clang-format"``).
        bin: Absolute path to the formatter executable.
        file: Path to the formatter configuration file (e.g.
            ``.clang-format``, ``ruff.toml``).
    """

    name: str
    bin: Path
    file: Path


# Root of the project repository (three levels up from this file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Absolute path to ``tools/codegen/`` within the project.
CODEGEN_DIR = PROJECT_ROOT / "tools" / "codegen"
# Directory holding JSON/YAML rules files that feed the templates.
RULES_DIR = CODEGEN_DIR / "rules"
# Directory holding Python scripts that register engine configurations.
SCRIPTS_DIR = CODEGEN_DIR / "scripts"
# Directory holding Jinja2 template files (``.jinja``).
TEMPLATES_DIR = CODEGEN_DIR / "templates"

# Default `FileSystemLoader` instance pointing to
# `TEMPLATES_DIR`. Used by engines that do not provide a custom loader.
DEFAULT_LOADER: FileSystemLoader = FileSystemLoader(TEMPLATES_DIR)


def get_clang_format_path() -> Path:
    """Locate the clang-format binary on the system.

    Uses shutil.which() first (cross-platform), then falls back to
    common installation paths based on the current OS.

    Returns:
        Absolute path to the clang-format binary.

    Raises:
        FileNotFoundError: If clang-format cannot be found.
    """
    if (path := shutil.which("clang-format")) is not None:
        return Path(path)

    if sys.platform == "win32" and platform.machine() == "AMD64":
        candidates = [
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
            / "LLVM/bin/clang-format.exe",
        ]
    elif sys.platform == "linux":
        candidates = [
            Path("/usr/bin/clang-format"),
            Path("/usr/local/bin/clang-format"),
            Path.home() / ".local/bin/clang-format",
        ]
    else:
        raise OSError(
            f"Unsupported platform '{sys.platform}' for clang-format lookup."
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "clang-format not found. Install LLVM/Clang or add it to PATH."
    )


def get_ruff_path() -> Path:
    """Locate the ruff binary on the system.

    Uses shutil.which() first (cross-platform), then falls back to
    common installation paths based on the current OS.

    Returns:
        Absolute path to the ruff binary.

    Raises:
        FileNotFoundError: If ruff cannot be found.
    """
    if (path := shutil.which("ruff")) is not None:
        return Path(path)

    if sys.platform == "win32" and platform.machine() == "AMD64":
        local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates = [
            local_app_data / "Programs/Python/Python313/Scripts/ruff.exe",
            Path.home() / "AppData/Roaming/Python/Python313/Scripts/ruff.exe",
        ]
    elif sys.platform == "linux":
        candidates = [
            Path("/usr/bin/ruff"),
            Path("/usr/local/bin/ruff"),
            Path.home() / ".local/bin/ruff",
        ]
    else:
        raise OSError(f"Unsupported platform '{sys.platform}' for ruff lookup.")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "ruff not found. Install with 'uv add ruff' or add it to PATH."
    )


# Absolute path to the ``ruff`` binary, resolved once at import time.
RUFF_PATH = get_ruff_path()
# Absolute path to the ``clang-format`` binary, resolved once at import
# time.
CLANG_FORMATTER_PATH = get_clang_format_path()
# User home directory, used as a fallback target for formatters when no
# specific output file has been set.
HOME = Path.home()


@dataclass(frozen=True)
class ClangFormatter(Formatter):
    """Clang-format formatter configuration.

    Locates the ``clang-format`` binary and the project-level
    ``.clang-format`` config file.  The ``target`` field specifies
    the file to format in-place.

    Attributes:
        name: Formatter identifier, defaults to ``"clang-format"``.
        bin: Auto-detected path to the ``clang-format`` binary.
        file: Path to the project ``.clang-format`` config.
        target: File to format.
    """

    name: str = "clang-format"
    bin: Path = CLANG_FORMATTER_PATH
    file: Path = PROJECT_ROOT / ".clang-format"
    target: Path = HOME


@dataclass(frozen=True)
class RuffFormatter(Formatter):
    """Ruff formatter/linter configuration.

    Locates the ``ruff`` binary and the project-level ``ruff.toml``
    config file.  When ``check`` is ``True``, ``ruff check`` is
    executed after ``ruff format``.

    Attributes:
        name: Formatter identifier, defaults to ``"ruff"``.
        bin: Auto-detected path to the ``ruff`` binary.
        file: Path to the project ``ruff.toml`` config.
        target: File to format.
        check: If ``True``, also run ``ruff check`` after formatting.
    """

    name: str = "ruff"
    bin: Path = RUFF_PATH
    file: Path = PROJECT_ROOT / "ruff.toml"
    target: Path = HOME
    check: bool = False


class EngineManager:
    """Register, track, and execute engines as a singleton manager.

    Provides a centralized registry for Engine instances and supports
    running code generation across all registered engines with optional
    exclusion of specific engines by id.

    This is typically used to group multiple generation stages so they
    can all be triggered from a single entry point (e.g. generate.py).
    Each stage may read a different rules file and produce a different
    output file from the same or a different template.

    Example:
        Register two engines from generation scripts and run them:

        >>> from pathlib import Path
        >>> from jinja2 import Environment, FileSystemLoader
        >>> from tools.codegen.engine import (
        ...     ClangFormatter, CodeGenEngine, Engine, EnvSpec, Renders,
        ...     EngineManager,
        ... )
        >>>
        >>> templates_dir = Path('templates/')
        >>> jinja_env = Environment(loader=FileSystemLoader(templates_dir))
        >>> env_spec = [EnvSpec(id=0, env=jinja_env, name='env_0')]
        >>>
        >>> # Load rules and define renders for scalar dtype casting
        >>> rules = json.loads((Path('rules/rules.json')).read_text())
        >>> output = Path('output.c')
        >>> renders = [
        ...     Renders(
        ...         template_name='template.jinja',
        ...         render_path=output,
        ...         formatter=ClangFormatter(target=output),
        ...         data=rules,
        ...     ),
        ... ]
        >>>
        >>> cg = CodeGenEngine(env_spec)
        >>> cg.add_rendering_templates(env_spec, [renders])
        >>> engine = Engine(engine=cg, name='engine', id=0)
        >>>
        >>> manager = EngineManager()
        >>> manager.register_engine(engine)
        >>> manager.run()                # generate
        >>> manager.run(exclude_id=0)    # skip engine 0
        >>> manager.pop(0)               # remove engine 0
    """

    _engines: ClassVar[OrderedDict[int, Engine]] = OrderedDict()
    _instance: EngineManager | None = None
    _initialized: bool = False

    def __new__(cls) -> EngineManager:
        """Return the singleton EngineManager instance."""
        if cls._instance is None:
            cls._instance = object.__new__(cls)
            return cls._instance
        return cls._instance

    def __init__(self, engines: list[Engine] | None = None) -> None:
        """Initialize the engines manager singleton.

        Args:
            engines: Optional list of engines to register at startup.
        """
        if not self._initialized:
            if engines is not None:
                for engine in engines:
                    self.register_engine(engine)
            self._initialized = True

    def register_engine(self, engine: Engine) -> None:
        """Register a single engine.

        Args:
            engine: The engine to register.

        Raises:
            ValueError: If an engine with the same id is already registered.
        """
        if engine.id in self._engines:
            raise ValueError(f"Duplicate engine {engine.name}")

        self._engines[engine.id] = engine

    def pop(self, engine_id: int) -> Engine:
        """Remove and return an engine by its id.

        Args:
            engine_id: The id of the engine to remove.

        Returns:
            The removed engine.

        Raises:
            KeyError: If no engine with the given id is registered.
        """
        if engine_id not in self._engines:
            raise KeyError(f"Id '{engine_id}' not found")

        return self._engines.pop(engine_id)

    def run(
        self,
        exclude_id: int | list[int] | None = None,
        *,
        stop_on_error: bool = True,
        run_formatters: bool = True,
        verbose: bool = False,
    ) -> list[EngineRunResult]:
        """Run code generation for all engines, optionally excluding some.

        Args:
            exclude_id: Optional engine id or list of engine ids to skip.
                        If None, all engines are executed.
            stop_on_error: If True (default), the first engine failure
                raises immediately and no further engines run. If False,
                every selected engine is attempted regardless of earlier
                failures, and the per-engine outcomes are returned so the
                caller can decide how to report them.
            run_formatters: If True (default), execute file formatters
                (clang-format, ruff) on each rendered output. If False,
                skip formatting entirely.
            verbose: If True, print a log message per engine before
                execution, showing the engine name and id.

        Engines are always executed in ascending id order regardless of
        the order in which they were registered.

        Returns:
            A list of EngineRunResult, one per engine that was attempted,
            in id order. Only meaningful when stop_on_error=False;
            when stop_on_error=True and everything succeeds, this is the
            full list of successful results.

        Raises:
            ValueError: If an id in exclude_id is not found in the registry.
            TypeError: If exclude_id is not None, int, or list[int].
            RuntimeError: If stop_on_error=True and an engine fails; wraps
                the original exception.
        """
        if exclude_id is None:
            engines_to_run = sorted(self._engines.values(), key=lambda e: e.id)
        elif isinstance(exclude_id, int):
            if exclude_id not in self._engines:
                raise ValueError(
                    f"Engine id '{exclude_id}' not found in registry"
                )
            engines_to_run = sorted(
                (e for e in self._engines.values() if e.id != exclude_id),
                key=lambda e: e.id,
            )
        elif isinstance(exclude_id, list):
            exclude_set = set(exclude_id)
            for eid in exclude_set:
                if eid not in self._engines:
                    raise ValueError(f"Engine id '{eid}' not found in registry")
            engines_to_run = sorted(
                (e for e in self._engines.values() if e.id not in exclude_set),
                key=lambda e: e.id,
            )
        else:
            raise TypeError(
                f"Expected int, list[int], or None, got {type(exclude_id).__name__}"
            )

        results: list[EngineRunResult] = []

        total_renders = sum(
            len(entry.renders)
            for engine in engines_to_run
            for entry in engine.engine._registry.values()
        )

        with tqdm(total=total_renders, desc="Rendering", unit="render") as pbar:
            for engine in engines_to_run:
                if verbose:
                    tqdm.write(
                        f"Executing {engine.name}... (Engine ID: {engine.id})"
                    )
                try:
                    engine.engine.generate(
                        run_formatters=run_formatters, pbar=pbar
                    )
                except Exception as exc:
                    if stop_on_error:
                        raise RuntimeError(
                            f"Engine '{engine.name}' (id={engine.id}) failed: {exc}"
                        ) from exc
                    results.append(
                        EngineRunResult(engine=engine, ok=False, error=exc)
                    )
                    continue

                results.append(
                    EngineRunResult(engine=engine, ok=True, error=None)
                )

        return results


class CodeGenEngine:
    """Manage code generation across multiple template environments.

    Maintains a registry of Jinja2 environments and their associated
    templates.  Data loaded from JSON rules files is injected into
    .jinja templates and the rendered output is written to disk.

    Each engine is tied to one or more environments identified by a
    unique id.  Templates are registered per-environment together with
    the data and the destination path for the output.

    Example:
        Load rules and render a scalar dtype casting template:

        >>> from pathlib import Path
        >>> from jinja2 import Environment, FileSystemLoader
        >>> from tools.codegen.engine import (
        ...     ClangFormatter, CodeGenEngine, EnvSpec, Renders,
        ... )
        >>>
        >>> templates_dir = Path('templates/')
        >>> jinja_env = Environment(loader=FileSystemLoader(templates_dir))
        >>> spec = [EnvSpec(id=1, env=jinja_env, name='env_1')]
        >>>
        >>> rules_path = Path('rules/rules.json')
        >>> rules = json.loads(rules_path.read_text())
        >>>
        >>> output = Path('file.c')
        >>> renders = [
        ...     Renders(
        ...         template_name='other_template.jinja',
        ...         render_path=output,
        ...         formatter=ClangFormatter(target=output),
        ...         data=rules,
        ...     ),
        ... ]
        >>>
        >>> engine = CodeGenEngine(env_spec=spec)
        >>> engine.add_rendering_templates(spec, [renders])
        >>> engine.generate()
    """

    def __init__(self, env_specs: list[EnvSpec] | None = None) -> None:
        """Initialize the code generation engine.

        Args:
            env_specs: Optional list of environment specs to register at startup.
        """
        self._registry: dict[int, _EnvEntry] = {}

        if env_specs is not None:
            for env in env_specs:
                self._register(env)

    def _register(self, env: EnvSpec) -> None:
        """Register an environment, validating uniqueness and name format."""
        if env is None:
            raise ValueError("Environment cannot be None")
        if env.id in self._registry:
            raise ValueError(f"Duplicate environment id: {env.id}")
        if env.name != f"env_{env.id}":
            raise ValueError(
                f"Invalid environment name '{env.name}', expected 'env_{env.id}'"
            )

        self._registry[env.id] = _EnvEntry(env)

    def set_new_env(self, env_specs: list[EnvSpec]) -> None:
        """Register additional environment specs.

        Args:
            env_specs: list of environment specs to register.
        """
        for env in env_specs:
            self._register(env)

    def add_rendering_templates(
        self,
        env_specs: list[EnvSpec],
        renders: list[list[Renders]],
    ) -> None:
        """Add templates and their rendering data for specified environment specs.

        Args:
            env_specs: list of environment specs to add templates to.
            renders: list of render specifications, one per environment.

        Raises:
            ValueError: If lengths mismatch or environment is not registered.
        """
        if len(env_specs) != len(renders):
            raise ValueError(
                f"env_specs and renders must have the same length, "
                f"got {len(env_specs)} and {len(renders)}"
            )

        for spec, templates in zip(env_specs, renders, strict=False):
            if spec.id not in self._registry:
                raise ValueError(f"EnSpec id {spec.id} not registered")

            entry = self._registry[spec.id]
            for template in templates:
                entry.templates.append(
                    spec.env.get_template(template.template_name)
                )
                entry.renders.append(template)

    def _generate_rendered_file(
        self,
        file_path: Path,
        result: str,
        formatter: ClangFormatter | RuffFormatter | None,
    ) -> None:
        """Write rendered content to disk, creating parent dirs as needed.

        If a formatter is provided, it is executed on the target file
        after writing.

        Args:
            file_path: Destination path for the rendered output.
            result: Rendered template string to write.
            formatter: Optional formatter to apply after writing.

        Raises:
            RuntimeError: If writing the file or running the formatter
                fails.
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to create directory or write file '{file_path}': {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error writing rendered output to '{file_path}': {exc}"
            ) from exc

        if formatter is not None:
            binary = str(formatter.bin)
            file = str(formatter.file)
            target = str(formatter.target)
            if isinstance(formatter, ClangFormatter):
                self._run_formatter(
                    formatter,
                    [
                        binary,
                        "-i",
                        f"--style=file:{file}",
                        target,
                    ],
                )
            elif isinstance(formatter, RuffFormatter):
                self._run_formatter(
                    formatter,
                    [binary, "format", target],
                )
                if formatter.check:
                    self._run_formatter(
                        formatter,
                        [binary, "check", target],
                    )
            else:
                raise TypeError(
                    f"Unsupported formatter type: {type(formatter).__name__}. "
                    f"Expected ClangFormatter or RuffFormatter."
                )

    def _run_formatter(
        self, formatter: ClangFormatter | RuffFormatter, command: list[str]
    ):
        """Execute a formatter binary on a target file.

        Args:
            formatter: The formatter configuration containing paths
                to the binary and config file.
            command: Full command-line arguments to pass to
                ``subprocess.run``.

        Raises:
            FileNotFoundError: If the formatter config file or
                target file does not exist.
            RuntimeError: If the formatter exits with a non-zero
                code or an unexpected error occurs.
        """
        if not formatter.file.exists():
            raise FileNotFoundError(
                f"Formatter config file not found: '{formatter.file}'. "
                f"Expected '{formatter.name}' configuration at this path."
            )

        try:
            if not formatter.target.exists() or not os.path.isfile(
                formatter.target
            ):
                raise FileNotFoundError(
                    f"Target file to format not found: '{formatter.target}'."
                )
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Formatter '{formatter.name}' failed with exit code "
                f"{exc.returncode}: {' '.join(command)}"
            ) from exc
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error running formatter '{formatter.name}': {exc}"
            ) from exc

    def generate(
        self,
        run_formatters: bool = True,
        pbar: tqdm | None = None,
    ) -> None:
        """Render all registered templates and write output files.

        Iterates through all registered specs and renders their templates
        using the provided data, writing results to the specified paths.
        When ``run_formatters`` is True, each rendered file is passed
        through its configured formatter (if any).

        Args:
            run_formatters: If True, execute file formatters on each
                rendered output. If False, skip formatting.
            pbar: Optional tqdm progress bar to update after each render.

        Raises:
            jinja2.TemplateNotFound: If a template file does not exist.
            jinja2.TemplateError: If a Jinja2 template fails to render.
            OSError: If writing the output file fails.
        """
        for entry in self._registry.values():
            if not entry.templates:
                continue

            for template, render in zip(
                entry.templates, entry.renders, strict=False
            ):
                try:
                    result = template.render(rules=render.data)
                    self._generate_rendered_file(
                        file_path=render.render_path,
                        result=result,
                        formatter=render.formatter if run_formatters else None,
                    )
                except TemplateNotFound as exc:
                    raise TemplateNotFound(
                        f"Template not found for render '{render.template_name}': {exc}"
                    ) from exc
                except TemplateError as exc:
                    raise TemplateError(
                        f"Failed to render template '{render.template_name}': {exc}"
                    ) from exc
                except OSError as exc:
                    raise OSError(
                        f"Failed to write output to '{render.render_path}': {exc}"
                    ) from exc
                finally:
                    if pbar is not None:
                        pbar.update(1)


# Init manager
manager = EngineManager()


def register_engine(engine: Engine | list[Engine]) -> None:
    """Register one or more engines with the global engine manager.

    This function provides a unified interface to register a single Engine
    instance or a list of Engine instances.

    Args:
        engine: An Engine instance or a list of Engine instances to register.

    Raises:
        ValueError: If engine is None or an invalid type is provided.
    """
    if engine is None:
        raise ValueError("Engine cannot be None")

    if isinstance(engine, list):
        for eng in engine:
            manager.register_engine(eng)
    elif isinstance(engine, Engine):
        manager.register_engine(engine)
    else:
        raise ValueError(
            f"Expected Engine or list[Engine], got {type(engine).__name__}"
        )


def is_manager_empty() -> bool:
    """Returns True if the manager has registered engines; otherwise, returns False."""
    return not manager._engines


def generate(
    exclude_id: int | list[int] | None = None,
    *,
    stop_on_error: bool = True,
    run_formatters: bool = True,
    verbose: bool = False,
) -> list[EngineRunResult]:
    """Run code generation for all registered engines.

    Convenience wrapper around ``manager.run()``.  Accepts an optional
    engine id or list of ids to exclude from the current run.

    Args:
        exclude_id: Optional engine id or list of engine ids to skip.
            If None, all engines are executed.
        stop_on_error: If True (default), stop and raise on the first
            engine failure. If False, run every selected engine and
            return per-engine results instead of raising.
        run_formatters: If True (default), execute file formatters
            (clang-format, ruff) on each rendered output. If False,
            skip formatting entirely.
        verbose: If True, print per-engine log messages during
            generation.

    Returns:
        A list of EngineRunResult describing what happened per engine.

    Raises:
        RuntimeError: If stop_on_error=True and an engine fails.
    """
    return manager.run(
        exclude_id,
        stop_on_error=stop_on_error,
        run_formatters=run_formatters,
        verbose=verbose,
    )
