# OpenCode

`https://opencode.ai/docs/agents` — re-fetch before generating.

Project agents: `.opencode/agents/<name>.md`. Filename is the name. Frontmatter + body.

```markdown
---
description: Reviews code for quality and best practices
mode: subagent
permission:
  edit: deny
---

You are a reviewer. Focus on correctness and missing tests.
```

## Fields

| Field | Notes |
| --- | --- |
| `description` | Required, 1–1024 chars. When to delegate. |
| `mode` | `subagent` (typical), `primary`, `all` (default). |
| `permission` | `allow` / `ask` / `deny`. Supports globs for `bash` and similar. |
| `hidden` | `true` hides from `@` autocomplete, still callable via Task. |
| `temperature` `top_p` `model` `color` `steps` `disable` | Optional. See official docs. |

`permission` keys: `read`, `edit` (`write`/`edit`), `glob`, `grep`, `list`, `bash`, `task`, `external_directory`, `skill`, etc. Per-agent refines global `opencode.json`.

```yaml
permission:
  bash:
    "*": ask
    "git status *": allow
```

Last match wins — put `*` first.

## Example

```markdown
---
description: Reviews code for correctness and missing tests. Use after code changes.
mode: subagent
permission:
  edit: deny
---

You are a reviewer. Cite file:line for every finding.
```
