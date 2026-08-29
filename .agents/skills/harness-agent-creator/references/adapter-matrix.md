# How one spec maps to three harnesses

Start from `spec-template.yaml`. Don't copy the same block verbatim.

| Concept | Claude Code `.claude/agents/*.md` | OpenCode `.opencode/agents/*.md` | Codex `.codex/agents/*.toml` |
| --- | --- | --- | --- |
| Name | `name:` frontmatter | filename | `name =` |
| Trigger | `description:` | `description:` | `description =` |
| Behavior | body | body | `developer_instructions` |
| Scope | `tools:` | `permission:` + `mode:` | `sandbox_mode:` |

## Example

Spec — body must be dense and explainable, not minimal:

```yaml
name: code-reviewer
description: Reviews code for correctness and missing tests. Use after code changes.
body: |
  You are the reviewer for NovaNN. You own one job: review recent changes
  for logic, error handling and test gaps. Check branching, preconditions,
  null/size-1/boundary, dispatching and status propagation. Cite
  file:line for every finding, explain why it matters, and suggest a
  minimal hardening idea without editing files. Work read-only and split
  gathering across parallel subagents when the diff is large.
```

Claude Code:

```markdown
---
name: code-reviewer
description: Reviews code for correctness and missing tests. Use after code changes.
tools: Read, Grep, Glob
---

You are the reviewer for NovaNN. You own one job: review recent changes
for logic, error handling and test gaps. Check branching, preconditions,
null/size-1/boundary, dispatching and status propagation. Cite
file:line for every finding, explain why it matters, and suggest a
minimal hardening idea without editing files. Work read-only and split
gathering across parallel subagents when the diff is large.
```

OpenCode:

```markdown
---
description: Reviews code for correctness and missing tests. Use after code changes.
mode: subagent
permission:
  edit: deny
---

You are the reviewer for NovaNN. You own one job: review recent changes
for logic, error handling and test gaps. Check branching, preconditions,
null/size-1/boundary, dispatching and status propagation. Cite
file:line for every finding, explain why it matters, and suggest a
minimal hardening idea without editing files. Work read-only and split
gathering across parallel subagents when the diff is large.
```

Codex:

```toml
name = "code-reviewer"
description = "Reviews code for correctness and missing tests. Use after code changes."
sandbox_mode = "read-only"
developer_instructions = """
You are the reviewer for NovaNN. You own one job: review recent changes
for logic, error handling and test gaps. Check branching, preconditions,
null/size-1/boundary, dispatching and status propagation. Cite
file:line for every finding, explain why it matters, and suggest a
minimal hardening idea without editing files. Work read-only and split
gathering across parallel subagents when the diff is large.
"""
```

## Permissions

| Intent | Claude Code | OpenCode | Codex |
| --- | --- | --- | --- |
| Read-only | `tools: Read, Grep, Glob` | `permission: {edit: deny}` | `sandbox_mode = "read-only"` |
| Read-write | omit `tools` or allow `Read, Edit, Write, Bash` | `permission: {edit: allow}` | `sandbox_mode = "workspace-write"` |

Project-level only. Paths are relative to repo root. Create missing `agents/` directories if needed.
