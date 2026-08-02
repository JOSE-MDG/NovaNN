"""CLI entry point for template-driven code generation.

Discovers and imports all generation scripts under ``scripts/`` to
register their engines, then runs them with optional exclusion of
specific engine ids.
"""

import importlib
import sys
from typing import TYPE_CHECKING, NoReturn

import typer

from tools.codegen import engine
from tools.codegen.engine import (
    PROJECT_ROOT,
    SCRIPTS_DIR,
    generate,
    is_manager_empty,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

app: typer.Typer = typer.Typer(help="NovaNN code generation toolchain.")

type Main = Callable[[], None] | None


def _discover_and_import_scripts() -> None:
    """Import every .py file under ``scripts/`` to register its engines.

    Each script is expected to call ``register_engine()`` at module
    level so its engines are available to the global ``EngineManager``.
    Helper modules are imported too, so they can expose utility entry
    points consumed by registered engines.

    Modules that define a ``BUILD_PRIORITY`` integer are collected and
    their ``main()`` is called in ascending priority order (lower value
    = earlier execution).
    """
    if not SCRIPTS_DIR.is_dir():
        return

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    build_modules: list[tuple[int, ModuleType]] = []

    for py_file in sorted(SCRIPTS_DIR.rglob("*.py")):
        # Build a dotted module name from PROJECT ROOT.
        # e.g. tools/codegen/scripts/.../gen_...
        #   -> tools.codegen.scripts.{...}.gen_...
        module_name = (
            py_file
            .relative_to(PROJECT_ROOT)
            .with_suffix("")
            .as_posix()
            .replace("/", ".")
        )

        if module_name in sys.modules:
            continue

        module: ModuleType = importlib.import_module(module_name)

        # Collect any module with a BUILD_PRIORITY attribute for
        # ordered execution, regardless of its file name pattern.
        priority: int | None = getattr(module, "BUILD_PRIORITY", None)
        if priority is not None:
            build_modules.append((priority, module))

    # Execute build modules in priority order.
    for _, module in sorted(build_modules, key=lambda x: x[0]):
        main: Main = getattr(module, "main", None)
        if callable(main):
            main()


def _parse_exclude(raw: str | None) -> list[int] | None:
    """Parse the ``--exclude`` CLI argument into a list of engine IDs.

    Args:
        raw: Raw comma-separated value from the ``--exclude`` option.

    Returns:
        The parsed engine ids, or None if ``raw`` is None.

    Raises:
        typer.BadParameter: If ``raw`` is empty or contains a token
            that is not an integer.
    """
    if raw is None:
        return None
    parts = [v.strip() for v in raw.split(",") if v.strip()]
    if not parts:
        raise typer.BadParameter(
            "Empty --exclude value. Provide engine IDs as a comma-separated "
            "list, e.g. '--exclude 1,3'."
        )
    parsed: list[int] = []
    for part in parts:
        try:
            parsed.append(int(part))
        except ValueError:
            raise typer.BadParameter(
                f"Invalid engine ID '{part}' in --exclude '{raw}'. "
                "Expected integers separated by commas, e.g. '--exclude 1,3'."
            ) from None
    return parsed


def _fail_no_engines() -> NoReturn:
    """Report that no engines were registered and exit with an error."""
    typer.echo(
        "No engines were registered. Expected gen_*.py scripts under "
        f"'{SCRIPTS_DIR}' that call register_engine() at import time.",
        err=True,
    )
    raise typer.Exit(code=1)


@app.callback()
def callback() -> None:
    """CodeGen code generation toolchain."""
    pass


@app.command()
def gen(
    all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Generate all registered engines. Mutually exclusive with --exclude.",
    ),
    exclude: str | None = typer.Option(
        None,
        "--exclude",
        "-e",
        help="Comma-separated list of engine IDs to exclude (e.g. '1,3').",
        callback=_parse_exclude,
    ),
    keep_going: bool = typer.Option(
        False,
        "--keep-going",
        help="Run every selected engine even if one fails; report all failures at the end.",
    ),
    run_formatters: bool | None = typer.Option(
        None,
        "--run-formatters/--no-run-formatters",
        help="Execute file formatters (clang-format, ruff) on rendered outputs.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show per-engine log messages during generation.",
    ),
    list_outputs: bool = typer.Option(
        False,
        "--list-outputs",
        "-lo",
        help=(
            "Print the paths of the files that would be generated, one per "
            "line, without generating anything."
        ),
    ),
) -> None:
    """Run code generation for all registered engines.

    With ``--list-outputs``, print the paths of the files that would be
    generated (one per line, relative to the project root) without
    generating anything.  The only options allowed alongside it are
    ``--all`` and ``--exclude``.
    """
    if all and exclude is not None:
        raise typer.BadParameter(
            "Cannot combine --all with --exclude. Use '--all' to generate "
            "every engine or '--exclude <ids>' to skip specific engines "
            "(e.g. '--exclude 1,3'), but not both."
        )

    formatters = True if run_formatters is None else run_formatters

    if list_outputs:
        conflicting: list[str] = []
        if keep_going:
            conflicting.append("--keep-going")
        if run_formatters is not None:
            conflicting.append(
                "--run-formatters" if run_formatters else "--no-run-formatters"
            )
        if verbose:
            conflicting.append("--verbose")
        if conflicting:
            raise typer.BadParameter(
                f"Cannot combine --list-outputs with {', '.join(conflicting)}. "
                "These options only affect file generation, which "
                "--list-outputs does not perform. The only options allowed "
                "alongside --list-outputs are --all and --exclude."
            )

        if is_manager_empty():
            _fail_no_engines()

        for path in engine.list_outputs(exclude):  # type: ignore[arg-type]
            typer.echo(path.relative_to(PROJECT_ROOT).as_posix())
        return

    if not all and exclude is None:
        typer.echo(
            "No --all or --exclude given; generating all registered engines. "
            "Use '--exclude <ids>' to skip specific engines or '--list-outputs' "
            "to preview the files that would be generated."
        )

    if is_manager_empty():
        _fail_no_engines()

    if not keep_going:
        generate(
            exclude_id=exclude,  # type: ignore
            run_formatters=formatters,
            verbose=verbose,
        )
        typer.echo("Generation complete.")
        return

    results = generate(
        exclude_id=exclude,  # type: ignore
        stop_on_error=False,
        run_formatters=formatters,
        verbose=verbose,
    )
    failures = [r for r in results if not r.ok]

    for r in results:
        status = "OK" if r.ok else "FAILED"
        typer.echo(f"[{status}] {r.engine.name} (id={r.engine.id})")
        if not r.ok:
            typer.echo(f"    {r.error}", err=True)

    if failures:
        typer.echo(
            f"{len(failures)}/{len(results)} engine(s) failed; fix the "
            f"reported errors above and re-run.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("Generation complete.")


def main() -> None:
    """Entry point for the code generation CLI.

    Discovers all generation scripts under ``scripts/``, then invokes
    the Typer application to parse arguments and run the selected
    engines.
    """
    _discover_and_import_scripts()
    app()


if __name__ == "__main__":
    main()
