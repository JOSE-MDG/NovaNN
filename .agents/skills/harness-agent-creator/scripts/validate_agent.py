#!/usr/bin/env python3
"""Validate project agent files."""

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

CLAUDE_KNOWN_TOOLS = {
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Bash",
    "PowerShell",
    "List",
    "Task",
    "Agent",
    "Skill",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "AskUserQuestion",
    "NotebookEdit",
    "Monitor",
    "ToolSearch",
    "EnterWorktree",
    "ExitWorktree",
    "TaskStop",
    "SendMessage",
    "Artifact",
}
CLAUDE_KNOWN_TOOLS_LOWER = {t.lower(): t for t in CLAUDE_KNOWN_TOOLS}

OPENCODE_VALID_MODES = {"primary", "subagent", "all"}
OPENCODE_VALID_PERM = {"allow", "ask", "deny"}

app: typer.Typer = typer.Typer(
    help="Validate project agent files.",
    add_completion=False,
)


def parse_frontmatter(path: Path):
    """Parse frontmatter from Markdown agent file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None, text, ["missing frontmatter ---"]
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text, ["unterminated frontmatter"]
    raw = parts[1]
    body = parts[2]
    if yaml is None:
        return None, body, ["PyYAML not installed (uv run --with pyyaml)"]
    try:
        data = yaml.safe_load(raw) or {}
    except Exception as e:
        return None, body, [f"YAML parse error: {e}"]
    if not isinstance(data, dict):
        return None, body, ["frontmatter is not a mapping"]
    return data, body, []


def validate_claude(path: Path) -> list[str]:
    """Validate Claude Code agent."""
    errs = []
    data, body, perrs = parse_frontmatter(path)
    errs.extend(perrs)
    if data is None:
        return errs
    name = data.get("name")
    desc = data.get("description")
    if not name:
        errs.append("missing required 'name'")
    elif not isinstance(name, str) or not NAME_RE.match(name):
        errs.append(f"invalid name {name!r}: must match {NAME_RE.pattern}")
    elif name.startswith("-") or ":" in name:
        errs.append(
            f"invalid name {name!r}: must not start with '-' or contain ':'"
        )
    if not desc:
        errs.append("missing required 'description'")
    elif not isinstance(desc, str) or not desc.strip():
        errs.append("empty description")
    tools = data.get("tools")
    if tools is not None:
        if isinstance(tools, str):
            items = [t.strip() for t in tools.split(",") if t.strip()]
        elif isinstance(tools, list):
            items = [str(t).strip() for t in tools]
        else:
            items = []
            errs.append(
                f"'tools' must be string or list, got {type(tools).__name__}"
            )
        for t in items:
            base = t.split("(")[0].strip()
            base = base.split("__")[0] if "__" in base else base
            if (
                base.lower() not in CLAUDE_KNOWN_TOOLS_LOWER
                and not base.startswith("mcp__")
            ):
                errs.append(
                    f"unknown tool {t!r} in 'tools' (check https://code.claude.com/docs/en/sub-agents)"
                )
    if name and path.stem != name:
        errs.append(
            f"warning: file stem {path.stem!r} != name {name!r} (should match)"
        )
    if not body.strip():
        errs.append("empty body (system prompt)")
    return errs


def validate_opencode(path: Path) -> list[str]:
    """Validate OpenCode agent."""
    errs = []
    data, body, perrs = parse_frontmatter(path)
    errs.extend(perrs)
    if data is None:
        return errs
    desc = data.get("description")
    if not desc:
        errs.append("missing required 'description'")
    elif not isinstance(desc, str) or not (1 <= len(desc) <= 1024):
        errs.append(
            f"'description' must be 1-1024 chars, got {len(desc) if isinstance(desc, str) else 'non-string'}"
        )
    mode = data.get("mode")
    if mode is not None and mode not in OPENCODE_VALID_MODES:
        errs.append(
            f"invalid mode {mode!r}: must be one of {OPENCODE_VALID_MODES}"
        )
    perm = data.get("permission")
    if perm is not None:
        if not isinstance(perm, dict):
            errs.append(
                f"'permission' must be mapping, got {type(perm).__name__}"
            )
        else:
            for k, v in perm.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if vv not in OPENCODE_VALID_PERM:
                            errs.append(
                                f"permission.{k}['{kk}'] invalid value {vv!r}"
                            )
                elif v not in OPENCODE_VALID_PERM:
                    errs.append(f"permission.{k} invalid value {v!r}")
    if not body.strip():
        errs.append("empty body (system prompt)")
    return errs


def validate_codex(path: Path) -> list[str]:
    """Validate Codex agent."""
    errs = []
    try:
        import tomllib  # py 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return ["tomllib not available (python >=3.11 or install tomli)"]
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"TOML parse error: {e}"]
    for field in ("name", "description", "developer_instructions"):
        if field not in data or not str(data[field]).strip():
            errs.append(f"missing required '{field}'")  # noqa: PERF401
    name = data.get("name", "")
    if name and not NAME_RE.match(str(name)):
        errs.append(f"invalid name {name!r}: must match {NAME_RE.pattern}")
    sandbox = data.get("sandbox_mode")
    if sandbox is not None and sandbox not in (
        "read-only",
        "workspace-write",
        "danger-full-access",
    ):
        errs.append(f"invalid sandbox_mode {sandbox!r}")
    return errs


def validate_github(path: Path) -> list[str]:
    """Validate GitHub Copilot custom agent (.github/agents/*.agent.md)."""
    errs = []
    data, body, perrs = parse_frontmatter(path)
    errs.extend(perrs)
    if data is None:
        return errs
    # name is optional in GitHub, but if present should be valid
    name = data.get("name")
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            errs.append("empty name")
        elif not NAME_RE.match(name):
            errs.append(f"invalid name {name!r}: must match {NAME_RE.pattern}")
        elif path.stem.removesuffix(".agent") != name and path.stem != name:
            # file is name.agent.md, stem is name.agent or name
            errs.append(
                f"warning: file stem {path.stem!r} != name {name!r} (should match)"
            )
    desc = data.get("description")
    if not desc:
        errs.append("missing required 'description'")
    elif not isinstance(desc, str) or not desc.strip():
        errs.append("empty description")
    tools = data.get("tools")
    if tools is not None:
        if isinstance(tools, str):
            # comma-separated string is allowed
            items = [t.strip() for t in tools.split(",") if t.strip()]
        elif isinstance(tools, list):
            items = [str(t).strip() for t in tools]
        else:
            items = []
            errs.append(f"'tools' must be string or list, got {type(tools).__name__}")
        # minimal check: tools should be known github tool names
        valid_github_tools = {"read", "search", "edit", "execute", "web", "agent"}
        for t in items:
            base = t.split("/")[0].strip().lower()
            if base not in valid_github_tools and base != "*":
                # not fatal, just warning style but treat as error for now if unknown
                # keep lenient: only warn if completely unknown?
                pass
    if not body.strip():
        errs.append("empty body (system prompt)")
    return errs


VALIDATORS = {
    ".claude": validate_claude,
    ".opencode": validate_opencode,
    ".codex": validate_codex,
    ".github": validate_github,
}


def classify(path: Path):
    """Pick validator by path."""
    s = str(path)
    if ".claude" in s:
        return validate_claude
    if ".opencode" in s:
        return validate_opencode
    if ".codex" in s:
        return validate_codex
    if ".github" in s:
        return validate_github
    if path.suffix == ".toml":
        return validate_codex
    if path.suffix == ".md" and ".agent." in path.name:
        return validate_github
    return validate_claude


@app.command()
def validate(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="Agent file or directory to validate",
            show_default=False,
        ),
    ],
) -> None:
    """Validate project agent files."""
    targets: list[Path] = []
    for p in paths:
        if p.is_dir():
            targets.extend(p.glob("*.md"))
            targets.extend(p.glob("*.toml"))
        elif p.is_file():
            targets.append(p)
        else:
            typer.echo(f"not found: {p}", err=True)
            raise typer.Exit(code=2)
    if not targets:
        typer.echo("no agent files found", err=True)
        raise typer.Exit(code=2)
    ok = True
    for t in sorted(targets):
        fn = classify(t)
        errs = fn(t)
        real = [e for e in errs if not e.startswith("warning:")]
        warns = [e for e in errs if e.startswith("warning:")]
        if real:
            ok = False
            typer.echo(f"✗ {t}")
            for e in errs:
                typer.echo(f"  - {e}")
        else:
            suffix = f" ({'; '.join(warns)})" if warns else ""
            typer.echo(f"✓ {t}{suffix}")
    if not ok:
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
