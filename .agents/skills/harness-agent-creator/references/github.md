# GitHub Copilot

`https://docs.github.com/en/copilot/reference/custom-agents-configuration` — re-fetch before generating.

Project agents live in `.github/agents/` (checked in). Frontmatter + body. Filename must be `<name>.agent.md` (allowed chars `.` `-` `_` `a-z` `A-Z` `0-9`), `name` should match stem without `.agent`.

```markdown
---
name: code-reviewer
description: Reviews code for quality and best practices
argument-hint: A file or diff to review, e.g. 'review ncore/src/core/tensor.c'
tools: [read, search, web]
user-invocable: true
---

You are a code reviewer. Provide actionable feedback.
```

Body is the system prompt (prompt). Agent only sees this plus cwd, not the full Copilot prompt.

## Fields

Only `description` is required. Keep `description` short — it drives delegation. `name` is optional in GitHub docs but this skill always sets it to match filename for deduplication.

| Field | Notes |
| --- | --- |
| `name` | Optional in docs, `^[a-z0-9]+(-[a-z0-9]+)*$` in this skill. Should match file stem without `.agent`. |
| `description` | Required, when to delegate. Put detail in body. |
| `tools` | Allowlist. Omitted → all tools. Supports `read`, `search`, `edit`, `execute`, `web`, `agent` and `*`. List as `[read, search, web]` or comma string. |
| `argument-hint` | Hint shown in picker, e.g. `A file to review, e.g. 'review ncore/src/core/tensor.c'`. Optional but this skill sets it. |
| `user-invocable` | `true` (default) / `false`. `false` hides from picker but callable via handoffs. |
| `target` | `vscode` / `github-copilot` / omitted → both. |
| `model` | `Claude Sonnet 4.5`, `gpt-4o`, etc. Optional, inherits default. |
| `disable-model-invocation` / `infer` | `disable-model-invocation: true` prevents auto-invocation, agent must be manually selected. |

Full list: `mcp-servers`, `handoffs`, `metadata`. `argument-hint` and `handoffs` are VS Code specific and ignored for `target: github-copilot`; they are kept for compatibility.

## Tools

GitHub Copilot coding agent tools (subset):

- `read` — read files
- `search` — grep/glob
- `edit` — edit files
- `execute` — run Bash
- `web` — fetch/search web (use for docs verification)
- `agent` — spawn subagents

For read-only review agents use `tools: [read, search, web]`. For read-write (like `test-creator`) use `[read, search, edit, execute, web]`.

## Watcher

Changes in `.github/agents/` are picked up on next assignment/branch. No restart needed, but agent is instantiated per task using the latest commit on the target branch.

## Example — read-only

```markdown
---
name: explore-nova
description: Fast read-only exploration for NovaNN. Use when searching files without edits.
tools: [read, search, web]
user-invocable: true
---

You are a read-only explorer. Search and summarize, don't edit.
```
