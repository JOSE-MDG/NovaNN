# Codex

`https://learn.chatgpt.com/docs/agent-configuration/subagents` — re-fetch before generating.

Project agents: `.codex/agents/<name>.toml`. One file per agent, standalone TOML.

## Required

```toml
name = "code-mapper"
description = "Read-only explorer for locating code paths."
developer_instructions = """
Map the code that owns the failing flow. Identify entry points before editing.
"""
```

`name` should match the filename. Missing any of the three → not loaded.

## Optional

`sandbox_mode` (`read-only` / `workspace-write` / `danger-full-access`), `model`, `model_reasoning_effort`, `[mcp_servers.*]`, `[[skills.config]]`.

Project settings go in `.codex/config.toml` under `[agents]` (`enabled`, `max_concurrent_threads_per_session`). Don't confuse `.codex/agents/*.toml` with `AGENTS.md` — the latter is project instructions, not agents.

## Example

```toml
name = "reviewer"
description = "PR reviewer focused on correctness, security and missing tests."
sandbox_mode = "read-only"
developer_instructions = """
Review like an owner. Prioritize correctness, security and test gaps.
Cite file:line, include reproduction steps when possible.
"""
```
