#!/usr/bin/env python3
"""Generate project agent files from a YAML spec."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

app: typer.Typer = typer.Typer(
    help="Generate harness agents from spec YAML.",
    add_completion=False,
)


def load_spec(path: Path) -> dict:
    """Load spec YAML."""
    text = path.read_text(encoding="utf-8")
    if yaml:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("spec must be a YAML mapping")
        return data
    raise SystemExit(
        "PyYAML required: uv run --with pyyaml scripts/generate_agent.py --spec ..."
    )


def validate_name(name: str) -> None:
    """Check agent name."""
    if (
        not isinstance(name, str)
        or not (1 <= len(name) <= 64)
        or not NAME_RE.match(name)
    ):
        raise ValueError(
            f"invalid name {name!r}: must match {NAME_RE.pattern}, 1-64 chars"
        )


def render_claude(spec: dict) -> str:
    """Render .claude/agents/<name>.md."""
    name = spec["name"]
    desc = spec["description"].strip().replace('"', '\\"')
    body = spec.get("body", "").strip()
    tools = spec.get("tools_needed", {}).get("claude")
    lines = ["---", f"name: {name}", f'description: "{desc}"']
    if tools:
        lines.append(f"tools: {', '.join(tools)}")
    for k, v in spec.get("claude_extra", {}).items():
        lines.append(f"{k}: {v}")
    lines += ["---", "", body, ""]
    return "\n".join(lines)


def render_opencode(spec: dict) -> str:
    """Render .opencode/agents/<name>.md."""
    desc = spec["description"].strip().replace('"', '\\"')
    body = spec.get("body", "").strip()
    perm = spec.get("tools_needed", {}).get("opencode")
    mode = spec.get("opencode_mode", "subagent")
    lines = ["---", f'description: "{desc}"', f"mode: {mode}"]
    if perm:
        lines.append("permission:")
        for k, v in perm.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    lines.append(
                        f'    "{kk}": {vv}'
                        if " " in kk or "*" in kk
                        else f"    {kk}: {vv}"
                    )
            else:
                lines.append(f"  {k}: {v}")
    for k, v in spec.get("opencode_extra", {}).items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"{k}: {v}")
    lines += ["---", "", body, ""]
    return "\n".join(lines)


def render_codex(spec: dict) -> str:
    """Render .codex/agents/<name>.toml."""
    name = spec["name"]
    desc = spec["description"].strip()
    body = spec.get("body", "").strip()
    sandbox = spec.get("tools_needed", {}).get("codex_sandbox", "read-only")
    safe_body = body.replace('"""', '\\"\\"\\"')
    lines = [
        f'name = "{name}"',
        f'description = "{desc.replace(chr(34), chr(92) + chr(34))}"',
        f'sandbox_mode = "{sandbox}"',
        'developer_instructions = """',
        safe_body,
        '"""',
        "",
    ]
    for k, v in spec.get("codex_extra", {}).items():
        if isinstance(v, dict):
            lines.append(f"[{k}]")
            for kk, vv in v.items():
                lines.append(f'{kk} = "{vv}"')
        else:
            lines.append(f'{k} = "{v}"')
    return "\n".join(lines)


RENDERERS = {
    "claude": (render_claude, ".claude/agents/{name}.md"),
    "opencode": (render_opencode, ".opencode/agents/{name}.md"),
    "codex": (render_codex, ".codex/agents/{name}.toml"),
}


@app.command()
def generate(
    spec: Annotated[
        Path,
        typer.Option(
            "--spec",
            help="Path to agent-spec.yaml",
            show_default=False,
        ),
    ],
    harnesses: Annotated[
        str,
        typer.Option(
            "--harnesses",
            help="Comma-separated: claude,opencode,codex",
        ),
    ] = "claude,opencode,codex",
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Repo root / output prefix",
        ),
    ] = Path("."),
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print without writing",
        ),
    ] = False,
) -> None:
    """Generate harness agents from spec YAML."""
    try:
        data = load_spec(spec)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    except SystemExit as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None

    name = data.get("name")
    if not name:
        typer.echo("spec missing 'name'", err=True)
        raise typer.Exit(code=2)
    try:
        validate_name(name)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    if not data.get("description"):
        typer.echo("spec missing 'description'", err=True)
        raise typer.Exit(code=2)
    if not data.get("body"):
        typer.echo("spec missing 'body' (system prompt)", err=True)
        raise typer.Exit(code=2)

    requested = [h.strip().lower() for h in harnesses.split(",") if h.strip()]
    unknown = [h for h in requested if h not in RENDERERS]
    if unknown:
        typer.echo(
            f"unknown harness(es): {unknown} (choose from {list(RENDERERS)})",
            err=True,
        )
        raise typer.Exit(code=2)

    for h in requested:
        render, pattern = RENDERERS[h]
        content = render(data)
        rel = pattern.format(name=name)
        dest = out / rel
        if dry_run:
            typer.echo(f"--- {rel} (dry-run) ---\n{content}\n")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            typer.echo(f"wrote {dest}")

    if not dry_run:
        for h in requested:
            _, pattern = RENDERERS[h]
            rel = pattern.format(name=name)
            dest = out / rel
            typer.echo(f"validate: {dest}")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
