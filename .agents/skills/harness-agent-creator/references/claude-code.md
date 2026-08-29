# Claude Code

`https://code.claude.com/docs/en/sub-agents` — re-fetch before generating.

Project agents live in `.claude/agents/` (checked in). Frontmatter + body.

```markdown
---
name: code-reviewer
description: Reviews code for quality and best practices
tools: Read, Grep, Glob
---

You are a code reviewer. Provide actionable feedback.
```

Body is the system prompt. The agent only sees this plus cwd, not the full Claude Code prompt.

## Fields

Only `name` and `description` are required. Keep `description` short — it drives delegation.

| Field | Notes |
| --- | --- |
| `name` | `^[a-z0-9]+(-[a-z0-9]+)*$`, no `:` or leading `-`. Should match filename. |
| `description` | When to delegate. Put detail in body. |
| `tools` | Allowlist. Omitted → inherits all available to subagents. |
| `disallowedTools` | Denylist, applied before `tools`. |
| `permissionMode` | `default` / `acceptEdits` / `auto` / `dontAsk` / `bypassPermissions` / `plan` |
| `color` | `red` `blue` `green` `yellow` `purple` `orange` `pink` `cyan` |

Full list: `mcpServers`, `skills`, `hooks`, `memory`, `isolation: worktree`, `maxTurns`, `effort`, `background`, `initialPrompt`.

`tools` accepts `mcp__<server>` patterns. Background agents are further restricted — check the official docs.

## Watcher

Changes in `.claude/agents/` are picked up live. Restart is needed if the directory didn't exist when the session started.

## Example — read-only

```markdown
---
name: explore-nova
description: Fast read-only exploration for NovaNN. Use when searching files without edits.
tools: Read, Grep, Glob, Bash
---

You are a read-only explorer. Search and summarize, don't edit.
```
