"""CLI entry point for template-driven code generation.

Discovers and imports all generation scripts under ``scripts/`` to
register their engines, then runs them with optional exclusion of
specific engine ids.
"""

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from tools.codegen.engine import SCRIPTS_DIR, generate, is_manager_empty

if TYPE_CHECKING:
    from collections.abc import Callable

app: typer.Typer = typer.Typer(help="NovaNN code generation toolchain.")


def _discover_and_import_scripts() -> None:
    """Import every .py file under ``scripts/`` to register its engines.

    Each script is expected to call ``register_engine()`` at module
    level so its engines are available to the global ``EngineManager``.
    Helper modules are imported too, so they can expose utility entry
    points consumed by registered engines.
    """
    if not SCRIPTS_DIR.is_dir():
        return

    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for py_file in sorted(SCRIPTS_DIR.rglob("*.py")):
        # Build a dotted module name from PROJECT ROOT.
        # e.g. tools/codegen/scripts/.../gen_...
        #   -> tools.codegen.scripts.{...}.gen_...
        module_name = (
            py_file.relative_to(project_root)
            .with_suffix("")
            .as_posix()
            .replace("/", ".")
        )

        if module_name in sys.modules:
            continue

        module = importlib.import_module(module_name)

        if py_file.stem.startswith("_"):
            main: Callable[[], None] | None = getattr(module, "main", None)
            if callable(main):
                main()


def _parse_exclude(raw: str | None) -> list[int] | None:
    """Parse the ``--exclude`` CLI argument into a list of engine IDs."""
    if raw is None:
        return None
    parts = [v.strip() for v in raw.split(",") if v.strip()]
    if not parts:
        raise typer.BadParameter("Expected at least one integer.")
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise typer.BadParameter(  # noqa: B904
            f"'{raw}' is not a valid comma-separated list of integers."
        )


@app.callback()
def callback() -> None:
    """CodeGen code generation toolchain."""
    pass


@app.command()
def gen(
    all: bool = typer.Option(
        False,
        "--all",
        help="Generate all registered engines. Mutually exclusive with --exclude.",
    ),
    exclude: str | None = typer.Option(
        None,
        "--exclude",
        help="Comma-separated list of engine IDs to exclude (e.g. '1,3').",
        callback=_parse_exclude,
    ),
    keep_going: bool = typer.Option(
        False,
        "--keep-going",
        help="Run every selected engine even if one fails; report all failures at the end.",
    ),
    run_formatters: bool = typer.Option(
        True,
        "--run-formatters/--no-run-formatters",
        help="Execute file formatters (clang-format, ruff) on rendered outputs.",
    ),
) -> None:
    """Run code generation for all registered engines."""
    if all and exclude is not None:
        raise typer.BadParameter("--all and --exclude cannot be used together.")

    if not all and exclude is None:
        typer.echo(
            "No --all or --exclude given; generating all registered engines."
        )

    if is_manager_empty():
        typer.echo(
            "No engines were registered. Check that scripts/ exists and that "
            "each gen_*.py script calls register_engine() at import time.",
            err=True,
        )
        raise typer.Exit(code=1)

    if not keep_going:
        generate(exclude_id=exclude, run_formatters=run_formatters)  # type: ignore
        typer.echo("Generation complete.")
        return

    results = generate(
        exclude_id=exclude,  # type: ignore
        stop_on_error=False,
        run_formatters=run_formatters,  # type: ignore
    )
    failures = [r for r in results if not r.ok]

    for r in results:
        status = "OK" if r.ok else "FAILED"
        typer.echo(f"[{status}] {r.engine.name} (id={r.engine.id})")
        if not r.ok:
            typer.echo(f"    {r.error}", err=True)

    if failures:
        typer.echo(
            f"{len(failures)}/{len(results)} engine(s) failed.", err=True
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
